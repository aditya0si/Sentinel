"""OpenTelemetry tracing setup for Sentinel.

Provides:
- TracerProvider with OTLP and console exporters
- LangChain auto-instrumentation
- Custom span creation for guardrail validation and agent steps
- Helper to get a tracer for any component
"""

import logging
import os
from contextlib import asynccontextmanager, contextmanager
from typing import Any, Iterator

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
)
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.trace import Span, Status, StatusCode

logger = logging.getLogger(__name__)

# Global flag — set True to enable tracing
_TRACING_ENABLED = os.getenv("OTEL_TRACING_ENABLED", "true").lower() == "true"

# Cached tracer provider
_tracer_provider: TracerProvider | None = None


def init_tracing(
    service_name: str = "sentinel",
    otlp_endpoint: str | None = None,
    enable_console: bool = False,
) -> TracerProvider:
    """Initialize OpenTelemetry tracing.

    Args:
        service_name: Service name for the Resource.
        otlp_endpoint: OTLP gRPC endpoint (e.g. "http://localhost:4317").
        enable_console: Whether to also print spans to console.

    Returns:
        The configured TracerProvider.
    """
    global _tracer_provider

    resource = Resource(attributes={SERVICE_NAME: service_name})

    provider = TracerProvider(resource=resource)

    # Console exporter (useful for debugging)
    if enable_console:
        console_exporter = ConsoleSpanExporter()
        provider.add_span_processor(BatchSpanProcessor(console_exporter))

    # OTLP exporter
    otlp_endpoint = otlp_endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")
    if otlp_endpoint:
        try:
            otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
            provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
            logger.info("OTLP tracing enabled → %s", otlp_endpoint)
        except Exception as exc:
            logger.warning("Failed to configure OTLP exporter: %s", exc)

    trace.set_tracer_provider(provider)
    _tracer_provider = provider
    logger.info("Tracing initialized for service: %s", service_name)
    return provider


def get_tracer(name: str = "sentinel") -> trace.Tracer:
    """Get a tracer instance.

    Args:
        name: Instrumentation scope name.

    Returns:
        An OpenTelemetry Tracer.
    """
    return trace.get_tracer(name)


def instrument_langchain() -> bool:
    """Enable OpenTelemetry auto-instrumentation for LangChain.

    Registers the LangChain callback handler that automatically creates spans
    for LLM calls, tool invocations, chain executions, etc.

    Returns:
        True if instrumentation was applied successfully, False otherwise.
    """
    if not _TRACING_ENABLED:
        logger.debug("Tracing disabled — skipping LangChain instrumentation.")
        return False

    try:
        from opentelemetry.instrumentation.langchain import LangchainInstrumentor

        LangchainInstrumentor().instrument()
        logger.info("LangChain auto-instrumentation enabled.")
        return True
    except ImportError:
        logger.warning(
            "opentelemetry-instrumentation-langchain not installed. "
            "LangChain spans will not be auto-generated."
        )
        return False
    except Exception as exc:
        logger.error("Failed to instrument LangChain: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Context managers for custom spans
# ---------------------------------------------------------------------------


@contextmanager
def trace_guardrail_validation(
    validator_name: str,
    output_length: int = 0,
    context_size: int = 0,
) -> Iterator[Span]:
    """Create a span for guardrail validation.

    Usage:
        with trace_guardrail_validation("pii_detector", len(output)) as span:
            result = await validator.validate(output)
            span.set_attribute("passed", result.passed)
    """
    tracer = get_tracer("sentinel.guardrails")
    with tracer.start_as_current_span("guardrail.validate") as span:
        span.set_attribute("guardrail.validator_name", validator_name)
        span.set_attribute("guardrail.output_length", output_length)
        if context_size:
            span.set_attribute("guardrail.context_size", context_size)
        yield span


@contextmanager
def trace_agent_step(step_type: str, step_name: str = "") -> Iterator[Span]:
    """Create a span for an agent execution step.

    Step types: "invocation", "tool_call", "retrieval", "generation".

    Usage:
        with trace_agent_step("tool_call", "calculator"):
            result = calculator.invoke("2+2")
    """
    tracer = get_tracer("sentinel.agent")
    with tracer.start_as_current_span(f"agent.{step_type}") as span:
        span.set_attribute("agent.step_type", step_type)
        if step_name:
            span.set_attribute("agent.step_name", step_name)
        yield span


@asynccontextmanager
async def async_trace_guardrail_validation(
    validator_name: str,
    output_length: int = 0,
    context_size: int = 0,
):
    """Async version of trace_guardrail_validation."""
    tracer = get_tracer("sentinel.guardrails")
    with tracer.start_as_current_span("guardrail.validate") as span:
        span.set_attribute("guardrail.validator_name", validator_name)
        span.set_attribute("guardrail.output_length", output_length)
        if context_size:
            span.set_attribute("guardrail.context_size", context_size)
        yield span


def shutdown_tracing() -> None:
    """Gracefully shut down the tracer provider, flushing pending spans."""
    global _tracer_provider
    if _tracer_provider is not None:
        _tracer_provider.shutdown()
        logger.info("Tracing shut down.")
        _tracer_provider = None
