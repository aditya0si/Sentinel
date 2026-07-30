"""Observability exporters — configure OTLP and initialize tracing+metrics.

Usage:
    from observability.exporters import setup_observability
    setup_observability(service_name="sentinel")
"""

import logging
import os
from typing import Any

from observability.tracer import init_tracing, instrument_langchain
from observability.metrics import init_metrics

logger = logging.getLogger(__name__)


def setup_observability(
    service_name: str = "sentinel",
    otlp_endpoint: str | None = None,
    enable_console_tracing: bool = False,
    enable_langchain_instrumentation: bool = False,
) -> dict[str, Any]:
    """Initialize OpenTelemetry tracing and metrics.

    Call once at application startup.

    Args:
        service_name: Name for the service resource.
        otlp_endpoint: OTLP gRPC collector endpoint.
        enable_console_tracing: Print spans to stdout for debugging.
        enable_langchain_instrumentation: Auto-instrument LangChain.

    Returns:
        Configuration dict with status of each component.
    """
    otlp_endpoint = otlp_endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")

    status = {
        "tracing": "disabled",
        "metrics": "disabled",
        "langchain_instrumentation": False,
        "otlp_endpoint": otlp_endpoint or "not configured",
    }

    # Initialize tracing
    try:
        init_tracing(
            service_name=service_name,
            otlp_endpoint=otlp_endpoint or None,
            enable_console=enable_console_tracing,
        )
        status["tracing"] = "enabled"
    except Exception as exc:
        logger.error("Failed to initialize tracing: %s", exc)
        status["tracing"] = f"error: {exc}"

    # Initialize metrics
    try:
        init_metrics(
            service_name=service_name,
            otlp_endpoint=otlp_endpoint or None,
        )
        status["metrics"] = "enabled"
    except Exception as exc:
        logger.error("Failed to initialize metrics: %s", exc)
        status["metrics"] = f"error: {exc}"

    # LangChain instrumentation
    if enable_langchain_instrumentation:
        status["langchain_instrumentation"] = instrument_langchain()

    logger.info("Observability setup complete: %s", status)
    return status


def shutdown_observability() -> None:
    """Gracefully shut down all observability components."""
    from observability.tracer import shutdown_tracing
    from observability.metrics import shutdown_metrics

    shutdown_tracing()
    shutdown_metrics()
    logger.info("Observability shut down.")
