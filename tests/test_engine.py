"""Tests for the GuardrailEngine."""

import pytest
from guardrails.base import ValidationResult, Validator
from guardrails.engine import GuardrailEngine, GuardrailResult


class FakePassValidator(Validator):
    """A validator that always passes."""

    @property
    def name(self) -> str:
        return "fake_pass"

    async def validate(self, output: str, context: dict | None = None) -> ValidationResult:
        return ValidationResult(
            passed=True, confidence=1.0, details={}, validator_name=self.name
        )


class FakeFailValidator(Validator):
    """A validator that always fails."""

    @property
    def name(self) -> str:
        return "fake_fail"

    async def validate(self, output: str, context: dict | None = None) -> ValidationResult:
        return ValidationResult(
            passed=False, confidence=0.2, details={}, validator_name=self.name
        )


class FakeExceptionValidator(Validator):
    """A validator that raises an exception."""

    @property
    def name(self) -> str:
        return "fake_exception"

    async def validate(self, output: str, context: dict | None = None) -> ValidationResult:
        raise RuntimeError("Simulated failure")


class TestGuardrailEngine:
    @pytest.mark.asyncio
    async def test_all_pass(self):
        engine = GuardrailEngine(
            validators=[FakePassValidator(), FakePassValidator()],
            mode="all",
        )
        result = await engine.validate("hello")
        assert result.overall_pass is True
        assert result.aggregate_confidence == 1.0
        assert len(result.results) == 2
        assert result.failure_summary == "All validators passed."

    @pytest.mark.asyncio
    async def test_one_fail_all_mode(self):
        engine = GuardrailEngine(
            validators=[FakePassValidator(), FakeFailValidator()],
            mode="all",
        )
        result = await engine.validate("hello")
        assert result.overall_pass is False
        assert "fake_fail" in result.failed_validators

    @pytest.mark.asyncio
    async def test_threshold_mode(self):
        engine = GuardrailEngine(
            validators=[FakePassValidator(), FakePassValidator(), FakeFailValidator()],
            mode="threshold",
            threshold=0.6,
        )
        result = await engine.validate("hello")
        assert result.overall_pass is True  # 2/3 = 0.66 >= 0.6

    @pytest.mark.asyncio
    async def test_threshold_mode_fails(self):
        engine = GuardrailEngine(
            validators=[FakePassValidator(), FakeFailValidator(), FakeFailValidator()],
            mode="threshold",
            threshold=0.6,
        )
        result = await engine.validate("hello")
        assert result.overall_pass is False  # 1/3 = 0.33 < 0.6

    @pytest.mark.asyncio
    async def test_exception_handling(self):
        engine = GuardrailEngine(
            validators=[FakePassValidator(), FakeExceptionValidator()],
            mode="all",
        )
        result = await engine.validate("hello")
        assert result.overall_pass is False
        assert "fake_exception" in result.failed_validators
        assert any("Simulated failure" in r.details.get("error", "") for r in result.results)

    @pytest.mark.asyncio
    async def test_aggregate_confidence(self):
        engine = GuardrailEngine(
            validators=[FakePassValidator(), FakeFailValidator()],
            mode="all",
        )
        result = await engine.validate("hello")
        # Confidence: (1.0 + 0.2) / 2 = 0.6
        assert result.aggregate_confidence == pytest.approx(0.6)

    @pytest.mark.asyncio
    async def test_empty_validators(self):
        engine = GuardrailEngine(validators=[], mode="all")
        result = await engine.validate("hello")
        assert result.overall_pass is True

    @pytest.mark.asyncio
    async def test_invalid_mode(self):
        with pytest.raises(ValueError):
            GuardrailEngine(validators=[], mode="invalid_mode")

    @pytest.mark.asyncio
    async def test_failed_validators_property(self):
        engine = GuardrailEngine(
            validators=[FakePassValidator(), FakeFailValidator()],
            mode="all",
        )
        result = await engine.validate("hello")
        assert "fake_fail" in result.failed_validators
        assert "fake_pass" in result.passed_validators


class TestFromConfig:
    def test_from_config_basic(self):
        config = {
            "mode": "all",
            "validators": {
                "pii": {"enabled": True},
                "toxicity": {"enabled": True, "use_hf_pipeline": False},
                "hallucination": {"enabled": False},
                "schema": {"enabled": False},
                "rules": {"enabled": False},
            },
        }
        engine = GuardrailEngine.from_config(config)
        assert len(engine.validators) == 2
        assert engine.mode == "all"

    def test_from_config_with_rules(self):
        config = {
            "mode": "threshold",
            "threshold": 0.8,
            "validators": {
                "pii": {"enabled": False},
                "toxicity": {"enabled": False},
                "hallucination": {"enabled": False},
                "schema": {"enabled": False},
                "rules": {
                    "enabled": True,
                    "definitions": [
                        {"name": "no_spam", "pattern": r"buy now", "severity": "error"},
                        {"name": "check_tone", "pattern": r"(?i)please|thank", "severity": "warn"},
                    ],
                },
            },
        }
        engine = GuardrailEngine.from_config(config)
        assert len(engine.validators) == 1
        assert engine.mode == "threshold"
        assert engine.threshold == 0.8

    def test_from_config_empty(self):
        config = {"mode": "all", "validators": {}}
        engine = GuardrailEngine.from_config(config)
        assert len(engine.validators) == 0
