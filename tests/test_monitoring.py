"""Tests for drift monitoring (store, baseline, alerts)."""

import os
import tempfile

import pytest

from guardrails.engine import GuardrailEngine
from guardrails.base import ValidationResult, Validator
from monitoring.store import DriftStore
from monitoring.baseline import RollingBaseline
from monitoring.alerts import AlertLevel, DegradationDetector


# ---------------------------------------------------------------------------
# Fake validators for producing predictable GuardrailResults
# ---------------------------------------------------------------------------


class ConstPassValidator(Validator):
    """Always passes with given confidence."""

    def __init__(self, name_val: str = "const_pass", confidence: float = 0.9) -> None:
        self._name = name_val
        self._conf = confidence

    @property
    def name(self) -> str:
        return self._name

    async def validate(self, output: str, context: dict | None = None) -> ValidationResult:
        return ValidationResult(
            passed=True, confidence=self._conf,
            details={}, validator_name=self._name,
        )


class ConstFailValidator(Validator):
    """Always fails with given confidence."""

    def __init__(self, name_val: str = "const_fail", confidence: float = 0.3) -> None:
        self._name = name_val
        self._conf = confidence

    @property
    def name(self) -> str:
        return self._name

    async def validate(self, output: str, context: dict | None = None) -> ValidationResult:
        return ValidationResult(
            passed=False, confidence=self._conf,
            details={}, validator_name=self._name,
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_db():
    """Create a temporary SQLite database for testing."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    store = DriftStore(db_path=path)
    yield store
    os.unlink(path)


@pytest.fixture
def engine():
    return GuardrailEngine(
        validators=[ConstPassValidator(), ConstFailValidator()],
        mode="all",
    )


# ---------------------------------------------------------------------------
# DriftStore tests
# ---------------------------------------------------------------------------


class TestDriftStore:
    @pytest.mark.asyncio
    async def test_record_and_query(self, temp_db, engine):
        result = await engine.validate("test output")
        temp_db.record_result(result, agent_name="test_agent", input_preview="test")

        rows = temp_db.query_recent(agent_name="test_agent", limit=10)
        assert len(rows) == 2  # one per validator

    @pytest.mark.asyncio
    async def test_distinct_runs(self, temp_db, engine):
        result = await engine.validate("output 1")
        temp_db.record_result(result, agent_name="test_agent")
        result2 = await engine.validate("output 2")
        temp_db.record_result(result2, agent_name="test_agent")

        runs = temp_db.get_distinct_runs(agent_name="test_agent")
        assert len(runs) == 2

    @pytest.mark.asyncio
    async def test_pass_rate(self, temp_db, engine):
        # All-fail engine
        fail_engine = GuardrailEngine(
            validators=[ConstFailValidator()],
            mode="all",
        )
        result = await fail_engine.validate("test")
        temp_db.record_result(result, agent_name="test_agent")

        # All-pass engine
        pass_engine = GuardrailEngine(
            validators=[ConstPassValidator()],
            mode="all",
        )
        result2 = await pass_engine.validate("test")
        temp_db.record_result(result2, agent_name="test_agent")

        rate = temp_db.get_pass_rate(agent_name="test_agent", limit=10)
        assert 0.0 <= rate <= 1.0


# ---------------------------------------------------------------------------
# RollingBaseline tests
# ---------------------------------------------------------------------------


class TestRollingBaseline:
    @pytest.mark.asyncio
    async def test_empty_baseline(self, temp_db):
        baseline = RollingBaseline(store=temp_db, window_size=50)
        stats = baseline.compute(agent_name="nonexistent")
        assert stats["window_size"] == 0
        assert stats["pass_rate_mean"] == 0.0

    @pytest.mark.asyncio
    async def test_baseline_with_data(self, temp_db, engine):
        # Record several runs
        for i in range(10):
            result = await engine.validate(f"output {i}")
            temp_db.record_result(result, agent_name="test_agent")

        baseline = RollingBaseline(store=temp_db, window_size=50)
        stats = baseline.compute(agent_name="test_agent")

        assert stats["window_size"] == 10
        assert 0.0 <= stats["pass_rate_mean"] <= 1.0
        assert 0.0 <= stats["confidence_mean"] <= 1.0


# ---------------------------------------------------------------------------
# DegradationDetector tests
# ---------------------------------------------------------------------------


class TestDegradationDetector:
    @pytest.mark.asyncio
    async def test_no_baseline_returns_info(self, temp_db, engine):
        baseline = RollingBaseline(store=temp_db, window_size=50)
        detector = DegradationDetector(store=temp_db, baseline=baseline)

        result = await engine.validate("test")
        alert = detector.check(result, agent_name="nonexistent")
        assert alert.level == AlertLevel.INFO
        assert "Insufficient" in alert.message

    @pytest.mark.asyncio
    async def test_degradation_detection(self, temp_db, engine):
        # Build baseline with good results
        good_engine = GuardrailEngine(
            validators=[ConstPassValidator(confidence=0.95)],
            mode="all",
        )
        for i in range(20):
            result = await good_engine.validate(f"good {i}")
            temp_db.record_result(result, agent_name="test_agent")

        baseline = RollingBaseline(store=temp_db, window_size=50)
        detector = DegradationDetector(
            store=temp_db,
            baseline=baseline,
            confidence_threshold_warn=1.0,
            confidence_threshold_crit=2.0,
        )

        # Now submit a degraded result
        bad_result = await engine.validate("bad output")  # const_fail with 0.3 conf
        alert = detector.check(bad_result, agent_name="test_agent")

        # Should be WARNING or CRITICAL since confidence dropped to ~0.65
        assert alert.level in (AlertLevel.WARNING, AlertLevel.CRITICAL)

    @pytest.mark.asyncio
    async def test_normal_result_is_info(self, temp_db):
        # Build baseline
        engine = GuardrailEngine(
            validators=[ConstPassValidator(confidence=0.9)],
            mode="all",
        )
        for i in range(20):
            result = await engine.validate(f"output {i}")
            temp_db.record_result(result, agent_name="test_agent")

        baseline = RollingBaseline(store=temp_db, window_size=50)
        detector = DegradationDetector(store=temp_db, baseline=baseline)

        # Submit similar result
        result = await engine.validate("normal output")
        alert = detector.check(result, agent_name="test_agent")

        assert alert.level == AlertLevel.INFO
        assert "Normal" in alert.message

    @pytest.mark.asyncio
    async def test_alert_string_representation(self, temp_db, engine):
        baseline = RollingBaseline(store=temp_db, window_size=50)
        detector = DegradationDetector(store=temp_db, baseline=baseline)

        result = await engine.validate("test")
        alert = detector.check(result, agent_name="test")
        s = str(alert)
        assert "INFO" in s or "WARNING" in s or "CRITICAL" in s
