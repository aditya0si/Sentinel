"""Rolling window baseline computation for drift monitoring.

Computes mean and standard deviation of pass-rate and quality scores
over the last N runs. Used by the DegradationDetector to spot regressions.
"""

import logging
import math
from typing import Any

from monitoring.store import DriftStore

logger = logging.getLogger(__name__)


class RollingBaseline:
    """Maintains a rolling baseline of guardrail performance metrics.

    Computes statistics over a configurable window of recent runs:
    - pass_rate_mean / pass_rate_std
    - confidence_mean / confidence_std

    Usage:
        baseline = RollingBaseline(store=store, window_size=50)
        stats = baseline.compute(agent_name="my_agent")
        print(f"Pass rate: {stats['pass_rate_mean']:.2f} ± {stats['pass_rate_std']:.2f}")
    """

    def __init__(
        self,
        store: DriftStore,
        window_size: int = 100,
    ) -> None:
        """Initialize the rolling baseline.

        Args:
            store: DriftStore instance for querying historical data.
            window_size: Number of recent runs to include in the window.
        """
        self._store = store
        self._window_size = window_size

    def compute(self, agent_name: str = "default") -> dict[str, Any]:
        """Compute baseline statistics from the rolling window.

        Args:
            agent_name: Filter runs by agent name.

        Returns:
            Dict with keys:
                window_size: Actual number of runs used.
                pass_rate_mean, pass_rate_std: Pass rate statistics.
                confidence_mean, confidence_std: Aggregate confidence statistics.
                pass_rate_values: Per-run pass/fail values (for debugging).
                confidence_values: Per-run confidence values.
        """
        runs = self._store.get_distinct_runs(
            agent_name=agent_name,
            limit=self._window_size,
        )

        if not runs:
            logger.debug("No historical data for baseline (agent=%s).", agent_name)
            return {
                "window_size": 0,
                "pass_rate_mean": 0.0,
                "pass_rate_std": 0.0,
                "confidence_mean": 0.0,
                "confidence_std": 0.0,
                "pass_rate_values": [],
                "confidence_values": [],
            }

        pass_values = [1.0 if r["overall_pass"] else 0.0 for r in runs]
        conf_values = [r["aggregate_confidence"] for r in runs]

        n = len(pass_values)

        pass_mean = sum(pass_values) / n
        conf_mean = sum(conf_values) / n

        # Standard deviation (population formula)
        if n > 1:
            pass_std = math.sqrt(
                sum((x - pass_mean) ** 2 for x in pass_values) / n
            )
            conf_std = math.sqrt(
                sum((x - conf_mean) ** 2 for x in conf_values) / n
            )
        else:
            pass_std = 0.01  # small default so we don't divide by zero
            conf_std = 0.01

        # Ensure minimum std to avoid division-by-zero-like scenarios
        pass_std = max(pass_std, 0.01)
        conf_std = max(conf_std, 0.01)

        return {
            "window_size": n,
            "pass_rate_mean": pass_mean,
            "pass_rate_std": pass_std,
            "confidence_mean": conf_mean,
            "confidence_std": conf_std,
            "pass_rate_values": pass_values,
            "confidence_values": conf_values,
        }

    @property
    def window_size(self) -> int:
        return self._window_size
