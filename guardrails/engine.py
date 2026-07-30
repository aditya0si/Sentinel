"""GuardrailEngine — orchestration layer that runs all validators and aggregates results."""

import logging
from typing import Any

from pydantic import BaseModel, Field

from guardrails.base import ValidationResult, Validator
from guardrails.hallucination import HallucinationChecker
from guardrails.safety import PIIDetector, ToxicityScanner
from guardrails.schema import SchemaValidator
from guardrails.business_rules import Rule, RuleValidator

logger = logging.getLogger(__name__)


class GuardrailResult(BaseModel):
    """Aggregated result of running multiple validators.

    Attributes:
        overall_pass: True if all (or sufficient) validators passed.
        results: Individual ValidationResult from each validator.
        aggregate_confidence: Mean confidence across all validators.
        failure_summary: Human-readable summary of failures.
        mode: The aggregation mode used ("all" or "threshold").
    """

    overall_pass: bool
    results: list[ValidationResult] = Field(default_factory=list)
    aggregate_confidence: float = Field(ge=0.0, le=1.0)
    failure_summary: str = ""
    mode: str = "all"

    @property
    def failed_validators(self) -> list[str]:
        """Names of validators that did not pass."""
        return [r.validator_name for r in self.results if not r.passed]

    @property
    def passed_validators(self) -> list[str]:
        """Names of validators that passed."""
        return [r.validator_name for r in self.results if r.passed]


class GuardrailEngine:
    """Orchestrates multiple validators and aggregates their results.

    Supports two modes:
    - **all**: Every validator must pass for overall_pass to be True.
    - **threshold**: overall_pass is True if the fraction of passing validators
      meets or exceeds the threshold.

    Usage:
        engine = GuardrailEngine(
            validators=[SchemaValidator(...), PIIDetector(), ToxicityScanner()],
            mode="all",
        )
        result = await engine.validate(output="...", context={...})
        if not result.overall_pass:
            print(result.failure_summary)
    """

    def __init__(
        self,
        validators: list[Validator],
        mode: str = "all",
        threshold: float = 0.5,
    ) -> None:
        """Initialize the guardrail engine.

        Args:
            validators: List of validators to run on each output.
            mode: Aggregation mode ("all" or "threshold").
            threshold: Minimum fraction of validators that must pass (threshold mode only).
        """
        if mode not in ("all", "threshold"):
            raise ValueError(f"Invalid mode '{mode}'. Expected 'all' or 'threshold'.")

        self.validators = validators
        self.mode = mode
        self.threshold = threshold

    async def validate(
        self,
        output: str,
        context: dict[str, Any] | None = None,
    ) -> GuardrailResult:
        """Run all validators against the output.

        Args:
            output: The text/output to validate.
            context: Optional context passed to each validator.

        Returns:
            Aggregated GuardrailResult.
        """
        results: list[ValidationResult] = []
        context = context or {}

        for validator in self.validators:
            try:
                result = await validator.validate(output, context)
                results.append(result)
                logger.debug(
                    "Validator %s: passed=%s confidence=%.2f",
                    validator.name,
                    result.passed,
                    result.confidence,
                )
            except Exception as exc:
                logger.error(
                    "Validator %s raised exception: %s", validator.name, exc, exc_info=True
                )
                results.append(
                    ValidationResult(
                        passed=False,
                        confidence=0.0,
                        details={"error": str(exc)},
                        validator_name=validator.name,
                    )
                )

        overall_pass = self._compute_overall_pass(results)
        aggregate_confidence = self._compute_aggregate_confidence(results)
        failure_summary = self._build_failure_summary(results)

        return GuardrailResult(
            overall_pass=overall_pass,
            results=results,
            aggregate_confidence=aggregate_confidence,
            failure_summary=failure_summary,
            mode=self.mode,
        )

    def _compute_overall_pass(self, results: list[ValidationResult]) -> bool:
        if not results:
            return True

        passed_count = sum(1 for r in results if r.passed)
        if self.mode == "all":
            return passed_count == len(results)
        else:
            return (passed_count / len(results)) >= self.threshold

    @staticmethod
    def _compute_aggregate_confidence(results: list[ValidationResult]) -> float:
        if not results:
            return 1.0
        return sum(r.confidence for r in results) / len(results)

    @staticmethod
    def _build_failure_summary(results: list[ValidationResult]) -> str:
        failures = [r for r in results if not r.passed]
        if not failures:
            return "All validators passed."

        lines = [f"{len(failures)} validator(s) failed:"]
        for f in failures:
            detail_preview = {k: v for k, v in f.details.items() if k != "error_info"}
            lines.append(f"  - {f.validator_name}: {detail_preview}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Factory: build engine from a config dict (YAML-compatible)
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "GuardrailEngine":
        """Build a GuardrailEngine from a configuration dictionary.

        Example config:
            {
                "mode": "all",
                "threshold": 0.8,
                "validators": {
                    "pii": {"enabled": true},
                    "toxicity": {"enabled": true, "use_hf_pipeline": false},
                    "hallucination": {"enabled": false},
                    "schema": {"enabled": false},
                    "rules": {"enabled": false},
                }
            }
        """
        validators: list[Validator] = []
        validator_configs = config.get("validators", {})

        if validator_configs.get("pii", {}).get("enabled", False):
            use_presidio = validator_configs["pii"].get("use_presidio", False)
            validators.append(PIIDetector(use_presidio=use_presidio))

        if validator_configs.get("toxicity", {}).get("enabled", False):
            tox_cfg = validator_configs["toxicity"]
            validators.append(
                ToxicityScanner(
                    use_hf_pipeline=tox_cfg.get("use_hf_pipeline", False),
                    threshold=tox_cfg.get("threshold", 0.7),
                )
            )

        if validator_configs.get("hallucination", {}).get("enabled", False):
            hal_cfg = validator_configs["hallucination"]
            validators.append(
                HallucinationChecker(
                    mode=hal_cfg.get("mode", "embedding"),
                    threshold=hal_cfg.get("threshold", 0.7),
                )
            )

        if validator_configs.get("schema", {}).get("enabled", False):
            schema_cfg = validator_configs["schema"]
            model_path = schema_cfg.get("model", None)
            if model_path:
                logger.error("SchemaValidator.from_config: dynamic model loading not yet implemented")
            # SchemaValidator requires a Pydantic model — skip if not provided

        if validator_configs.get("rules", {}).get("enabled", False):
            rules_cfg = validator_configs["rules"]
            rule_defs = rules_cfg.get("definitions", [])
            rules = [
                Rule(
                    name=r.get("name", f"rule_{i}"),
                    pattern=r["pattern"],
                    description=r.get("description", ""),
                    severity=r.get("severity", "error"),
                )
                for i, r in enumerate(rule_defs)
            ]
            if rules:
                validators.append(RuleValidator(rules=rules))

        mode = config.get("mode", "all")
        threshold = config.get("threshold", 0.5)

        logger.info(
            "Built GuardrailEngine from config: mode=%s, %d validators",
            mode,
            len(validators),
        )
        return cls(validators=validators, mode=mode, threshold=threshold)
