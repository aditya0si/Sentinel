"""Sentinel API — FastAPI service exposing guardrails and agent as HTTP endpoints."""

import logging
import os
import time
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from guardrails.base import ValidationResult
from guardrails.engine import GuardrailEngine, GuardrailResult
from guardrails.safety import PIIDetector, ToxicityScanner
from guardrails.business_rules import Rule, RuleValidator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic request/response models
# ---------------------------------------------------------------------------


class ValidateRequest(BaseModel):
    """Request to run guardrails on arbitrary text."""

    output: str = Field(..., description="The text/output to validate.")
    context: dict[str, Any] = Field(
        default_factory=dict, description="Optional context for validators."
    )
    validators: list[str] = Field(
        default_factory=lambda: ["pii", "toxicity"],
        description="Which validators to enable: pii, toxicity, rules.",
    )
    rules: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Custom business rules to apply.",
    )


class ValidationDetail(BaseModel):
    """Single validator result."""

    validator_name: str
    passed: bool
    confidence: float
    details: dict[str, Any]


class ValidateResponse(BaseModel):
    """Response from the validate endpoint."""

    overall_pass: bool
    aggregate_confidence: float
    mode: str
    results: list[ValidationDetail]
    failure_summary: str


class AgentRequest(BaseModel):
    """Request to run the agent with guardrails."""

    input: str = Field(..., description="User input to the agent.")
    chat_history: list[dict[str, str]] = Field(
        default_factory=list,
        description="Previous messages as [{'role': 'user'/'assistant', 'content': '...'}].",
    )
    enable_guardrails: bool = Field(
        default=True, description="Whether to run guardrail validation."
    )


class AgentResponse(BaseModel):
    """Response from the agent endpoint."""

    output: str
    accepted: bool
    guardrail_result: ValidateResponse | None = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    uptime_seconds: float
    engine_config: dict[str, Any]


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

_start_time = time.time()


def create_app() -> FastAPI:
    """Build the FastAPI application with all routes."""
    app = FastAPI(
        title="Sentinel Guardrails API",
        description="Pluggable guardrails & quality-gate framework for agentic AI systems.",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Default engine (can be replaced with from_config in production)
    default_engine = GuardrailEngine(
        validators=[
            PIIDetector(use_presidio=False),
            ToxicityScanner(use_hf_pipeline=False),
        ],
        mode="all",
    )

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    @app.get("/health", response_model=HealthResponse)
    async def health_check() -> HealthResponse:
        """Health check endpoint."""
        return HealthResponse(
            status="healthy",
            version="0.1.0",
            uptime_seconds=time.time() - _start_time,
            engine_config={
                "mode": default_engine.mode,
                "validator_count": len(default_engine.validators),
                "validators": [v.name for v in default_engine.validators],
            },
        )

    @app.post("/validate", response_model=ValidateResponse)
    async def validate_output(request: ValidateRequest) -> ValidateResponse:
        """Run guardrail validation on arbitrary text output."""
        engine = _build_engine_from_request(request)

        guardrail_result = await engine.validate(
            output=request.output,
            context=request.context,
        )

        return ValidateResponse(
            overall_pass=guardrail_result.overall_pass,
            aggregate_confidence=guardrail_result.aggregate_confidence,
            mode=guardrail_result.mode,
            results=[
                ValidationDetail(
                    validator_name=r.validator_name,
                    passed=r.passed,
                    confidence=r.confidence,
                    details=r.details,
                )
                for r in guardrail_result.results
            ],
            failure_summary=guardrail_result.failure_summary,
        )

    @app.post("/agent", response_model=AgentResponse)
    async def run_agent(request: AgentRequest) -> AgentResponse:
        """Run the demo agent with optional guardrail validation."""
        try:
            agent = _get_guarded_agent(enable_guardrails=request.enable_guardrails)
            result = await agent.run(
                user_input=request.input,
                chat_history=request.chat_history,
            )

            guardrail_response = None
            if result.get("guardrail_result"):
                gr = result["guardrail_result"]
                guardrail_response = ValidateResponse(
                    overall_pass=gr.overall_pass,
                    aggregate_confidence=gr.aggregate_confidence,
                    mode=gr.mode,
                    results=[
                        ValidationDetail(
                            validator_name=r.validator_name,
                            passed=r.passed,
                            confidence=r.confidence,
                            details=r.details,
                        )
                        for r in gr.results
                    ],
                    failure_summary=gr.failure_summary,
                )

            return AgentResponse(
                output=result["output"],
                accepted=result["accepted"],
                guardrail_result=guardrail_response,
            )

        except ImportError as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Demo agent dependencies not available: {exc}. "
                "Install with: pip install langchain-openai langchain-chroma",
            )
        except Exception as exc:
            logger.error("Agent endpoint error: %s", exc, exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc))

    return app


def _build_engine_from_request(request: ValidateRequest) -> GuardrailEngine:
    """Build a GuardrailEngine based on the validate request parameters."""
    validators = []

    if "pii" in request.validators:
        validators.append(PIIDetector(use_presidio=False))

    if "toxicity" in request.validators:
        validators.append(ToxicityScanner(use_hf_pipeline=False))

    if "rules" in request.validators and request.rules:
        rules = [
            Rule(
                name=r.get("name", f"rule_{i}"),
                pattern=r["pattern"],
                description=r.get("description", ""),
                severity=r.get("severity", "error"),
            )
            for i, r in enumerate(request.rules)
        ]
        validators.append(RuleValidator(rules=rules))

    if not validators:
        validators.append(PIIDetector())  # fallback

    return GuardrailEngine(validators=validators, mode="all")


# Module-level app instance for uvicorn
app = create_app()


# ---------------------------------------------------------------------------
# Lazy singleton agents (built once, reused across /agent requests)
# ---------------------------------------------------------------------------

_guarded_agent_singleton: Any = None
_noop_agent_singleton: Any = None


def _get_guarded_agent(enable_guardrails: bool) -> Any:
    """Return (and lazily create) a singleton GuardedAgent instance."""
    global _guarded_agent_singleton, _noop_agent_singleton

    from demo.agent import GuardedAgent

    if enable_guardrails:
        if _guarded_agent_singleton is None:
            engine = GuardrailEngine(
                validators=[
                    PIIDetector(use_presidio=False),
                    ToxicityScanner(use_hf_pipeline=False),
                ],
                mode="all",
            )
            _guarded_agent_singleton = GuardedAgent(engine=engine)
        return _guarded_agent_singleton
    else:
        if _noop_agent_singleton is None:
            engine = GuardrailEngine(validators=[], mode="all")
            _noop_agent_singleton = GuardedAgent(engine=engine)
        return _noop_agent_singleton
