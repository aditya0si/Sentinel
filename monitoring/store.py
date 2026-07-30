"""SQLite storage for historical guardrail results.

Stores per-validation records for drift monitoring, baseline computation,
and degradation detection.

Schema:
    CREATE TABLE guardrail_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        agent_name TEXT DEFAULT 'default',
        model_version TEXT DEFAULT 'unknown',
        overall_pass INTEGER NOT NULL,
        aggregate_confidence REAL NOT NULL,
        validator_name TEXT NOT NULL,
        validator_passed INTEGER NOT NULL,
        validator_confidence REAL NOT NULL,
        details_json TEXT,
        input_preview TEXT
    );
"""

import json
import logging
import os
import sqlite3
import time
from contextlib import contextmanager
from typing import Any, Iterator

from guardrails.engine import GuardrailResult

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "sentinel_drift.db")

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS guardrail_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    agent_name TEXT DEFAULT 'default',
    model_version TEXT DEFAULT 'unknown',
    overall_pass INTEGER NOT NULL,
    aggregate_confidence REAL NOT NULL,
    validator_name TEXT NOT NULL,
    validator_passed INTEGER NOT NULL,
    validator_confidence REAL NOT NULL,
    details_json TEXT,
    input_preview TEXT
);
"""

CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_timestamp
ON guardrail_results(timestamp);
"""


class DriftStore:
    """Manages SQLite storage for guardrail drift monitoring."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH) -> None:
        """Initialize the drift store.

        Args:
            db_path: Path to the SQLite database file.
        """
        self._db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """Create tables and indexes if they don't exist."""
        with self._get_conn() as conn:
            conn.execute(CREATE_TABLE_SQL)
            conn.execute(CREATE_INDEX_SQL)
            conn.commit()

    @contextmanager
    def _get_conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def record_result(
        self,
        result: GuardrailResult,
        agent_name: str = "default",
        model_version: str = "unknown",
        input_preview: str = "",
    ) -> None:
        """Record a GuardrailResult in the drift store.

        Each validator result is stored as a separate row for granular analysis.

        Args:
            result: The aggregated guardrail result to store.
            agent_name: Identifier for the agent that produced the output.
            model_version: Version of the model used.
            input_preview: First 200 chars of the user input (for debugging).
        """
        ts = time.time()
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ts)) + f".{int((ts % 1) * 1_000_000):06d}"
        input_preview = input_preview[:200] if input_preview else ""

        rows = []
        for vr in result.results:
            rows.append((
                timestamp,
                agent_name,
                model_version,
                int(result.overall_pass),
                result.aggregate_confidence,
                vr.validator_name,
                int(vr.passed),
                vr.confidence,
                json.dumps(vr.details, default=str),
                input_preview,
            ))

        try:
            with self._get_conn() as conn:
                conn.executemany(
                    """INSERT INTO guardrail_results
                       (timestamp, agent_name, model_version, overall_pass,
                        aggregate_confidence, validator_name, validator_passed,
                        validator_confidence, details_json, input_preview)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    rows,
                )
                conn.commit()
            logger.debug("Recorded %d validator results to drift store.", len(rows))
        except Exception as exc:
            logger.error("Failed to record drift data: %s", exc)

    def query_recent(
        self,
        agent_name: str = "default",
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Fetch recent guardrail results.

        Args:
            agent_name: Filter by agent name.
            limit: Max number of rows to return.

        Returns:
            List of result dicts, most recent first.
        """
        with self._get_conn() as conn:
            rows = conn.execute(
                """SELECT * FROM guardrail_results
                   WHERE agent_name = ?
                   ORDER BY timestamp DESC
                   LIMIT ?""",
                (agent_name, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def query_window(
        self,
        agent_name: str = "default",
        window_size: int = 100,
        before_timestamp: str | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch results within a rolling window.

        Args:
            agent_name: Filter by agent name.
            window_size: Number of most recent runs to include.
            before_timestamp: Optional ISO timestamp to query before.

        Returns:
            List of result dicts ordered by timestamp descending.
        """
        if before_timestamp:
            rows_sql = """SELECT * FROM guardrail_results
                           WHERE agent_name = ? AND timestamp < ?
                           ORDER BY timestamp DESC
                           LIMIT ?"""
            params = (agent_name, before_timestamp, window_size * 10)  # 10x for per-validator rows
        else:
            rows_sql = """SELECT * FROM guardrail_results
                           WHERE agent_name = ?
                           ORDER BY timestamp DESC
                           LIMIT ?"""
            params = (agent_name, window_size * 10)

        with self._get_conn() as conn:
            rows = conn.execute(rows_sql, params).fetchall()
        return [dict(row) for row in rows]

    def get_distinct_runs(
        self,
        agent_name: str = "default",
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get distinct run-level aggregates (one row per invocation).

        Args:
            agent_name: Filter by agent name.
            limit: Max number of runs.

        Returns:
            List of run dicts with aggregate_confidence and overall_pass.
        """
        with self._get_conn() as conn:
            rows = conn.execute(
                """SELECT timestamp, agent_name, model_version,
                          MAX(overall_pass) as overall_pass,
                          MAX(aggregate_confidence) as aggregate_confidence,
                          input_preview
                   FROM guardrail_results
                   WHERE agent_name = ?
                   GROUP BY timestamp
                   ORDER BY timestamp DESC
                   LIMIT ?""",
                (agent_name, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_pass_rate(
        self,
        agent_name: str = "default",
        limit: int = 100,
    ) -> float:
        """Get the recent pass rate (fraction of runs that passed overall).

        Args:
            agent_name: Filter by agent name.
            limit: Number of recent runs to consider.

        Returns:
            Pass rate as a float between 0.0 and 1.0.
        """
        runs = self.get_distinct_runs(agent_name=agent_name, limit=limit)
        if not runs:
            return 1.0
        passed = sum(1 for r in runs if r["overall_pass"])
        return passed / len(runs)

    def close(self) -> None:
        """No-op for SQLite (connections are closed per-operation)."""
        pass
