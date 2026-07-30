"""OpenTelemetry metrics for Sentinel — token usage, latency, error rates.

Tracks:
- agent.invocations: Counter of agent runs
- guardrail.validations: Counter of guardrail checks
- step.duration_ms: Histogram of step durations
- token.usage: Counter of token consumption by model
- error.count: Counter of errors by type

Exported via OTLP metrics exporter.
"""

import logging
import os
import time
from contextlib import contextmanager
from typing import Any, Iterator

from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Token pricing table (USD per 1K tokens) — approximate, update as needed
# ---------------------------------------------------------------------------

PRICING: dict[str, dict[str, float]] = {
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    "text-embedding-3-small": {"input": 0.00002, "output": 0.0},
    "text-embedding-3-large": {"input": 0.00013, "output": 0.0},
}

# Cached meter and meter provider
_meter: metrics.Meter | None = None
_meter_provider: MeterProvider | None = None

# Metric instruments (created lazily)
_agent_invocations: metrics.Counter | None = None
_guardrail_validations: metrics.Counter | None = None
_step_duration: metrics.Histogram | None = None
_token_usage: metrics.Counter | None = None
_error_count: metrics.Counter | None = None


def init_metrics(
    service_name: str = "sentinel",
    otlp_endpoint: str | None = None,
) -> MeterProvider:
    """Initialize OpenTelemetry metrics.

    Args:
        service_name: Service name for the Resource.
        otlp_endpoint: OTLP gRPC endpoint for metrics.

    Returns:
        The configured MeterProvider.
    """
    global _meter, _meter_provider
    global _agent_invocations, _guardrail_validations, _step_duration, _token_usage, _error_count

    resource = Resource(attributes={SERVICE_NAME: service_name})

    otlp_endpoint = otlp_endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")
    readers = []

    if otlp_endpoint:
        try:
            metric_exporter = OTLPMetricExporter(endpoint=otlp_endpoint, insecure=True)
            readers.append(PeriodicExportingMetricReader(metric_exporter))
            logger.info("OTLP metrics enabled → %s", otlp_endpoint)
        except Exception as exc:
            logger.warning("Failed to configure OTLP metric exporter: %s", exc)

    provider = MeterProvider(resource=resource, metric_readers=readers)
    metrics.set_meter_provider(provider)
    _meter_provider = provider

    _meter = metrics.get_meter("sentinel")

    _agent_invocations = _meter.create_counter(
        name="agent.invocations",
        description="Number of agent invocations",
        unit="1",
    )

    _guardrail_validations = _meter.create_counter(
        name="guardrail.validations",
        description="Number of guardrail validation checks",
        unit="1",
    )

    _step_duration = _meter.create_histogram(
        name="step.duration_ms",
        description="Duration of execution steps in milliseconds",
        unit="ms",
    )

    _token_usage = _meter.create_counter(
        name="token.usage",
        description="Number of tokens consumed",
        unit="tokens",
    )

    _error_count = _meter.create_counter(
        name="error.count",
        description="Number of errors by type",
        unit="1",
    )

    logger.info("Metrics initialized for service: %s", service_name)
    return provider


# ---------------------------------------------------------------------------
# Recording helpers
# ---------------------------------------------------------------------------


def record_agent_invocation(model: str = "unknown", accepted: bool = True) -> None:
    """Record an agent invocation."""
    if _agent_invocations is not None:
        _agent_invocations.add(
            1,
            attributes={
                "model": model,
                "accepted": str(accepted),
            },
        )


def record_guardrail_validation(
    validator_name: str,
    passed: bool,
    duration_ms: float = 0.0,
) -> None:
    """Record a guardrail validation check."""
    if _guardrail_validations is not None:
        _guardrail_validations.add(
            1,
            attributes={
                "validator": validator_name,
                "passed": str(passed),
            },
        )
    if _step_duration is not None:
        _step_duration.record(
            duration_ms,
            attributes={
                "step_type": "guardrail",
                "step_name": validator_name,
            },
        )


def record_step_duration(step_type: str, step_name: str, duration_ms: float) -> None:
    """Record a step's duration."""
    if _step_duration is not None:
        _step_duration.record(
            duration_ms,
            attributes={
                "step_type": step_type,
                "step_name": step_name,
            },
        )


def record_token_usage(
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> float:
    """Record token usage and return estimated cost in USD.

    Args:
        model: Model name (e.g. "gpt-4o-mini").
        input_tokens: Number of prompt tokens.
        output_tokens: Number of completion tokens.

    Returns:
        Estimated cost in USD.
    """
    if _token_usage is not None:
        _token_usage.add(input_tokens, attributes={"model": model, "direction": "input"})
        _token_usage.add(output_tokens, attributes={"model": model, "direction": "output"})

    pricing = PRICING.get(model, {"input": 0.0, "output": 0.0})
    cost = (input_tokens / 1000) * pricing["input"] + (output_tokens / 1000) * pricing["output"]
    return cost


def record_error(error_type: str, component: str = "unknown") -> None:
    """Record an error occurrence."""
    if _error_count is not None:
        _error_count.add(
            1,
            attributes={
                "error_type": error_type,
                "component": component,
            },
        )


# ---------------------------------------------------------------------------
# Context manager for timing steps
# ---------------------------------------------------------------------------


@contextmanager
def timed_step(step_type: str, step_name: str = "") -> Iterator[None]:
    """Context manager that records the duration of a code block.

    Usage:
        with timed_step("agent_tool", "calculator"):
            result = calculator.invoke("2+2")
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        duration_ms = (time.perf_counter() - start) * 1000
        record_step_duration(step_type, step_name, duration_ms)


def shutdown_metrics() -> None:
    """Gracefully shut down metrics, flushing pending data."""
    global _meter_provider
    if _meter_provider is not None:
        _meter_provider.shutdown()
        logger.info("Metrics shut down.")
        _meter_provider = None
