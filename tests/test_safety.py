"""Tests for the PIIDetector and ToxicityScanner."""

import pytest
from guardrails.safety import PIIDetector, ToxicityScanner


class TestPIIDetector:
    @pytest.mark.asyncio
    async def test_no_pii(self):
        detector = PIIDetector()
        result = await detector.validate("The weather is nice today.")
        assert result.passed is True
        assert result.confidence == 1.0
        assert len(result.details["findings"]) == 0

    @pytest.mark.asyncio
    async def test_email_detected(self):
        detector = PIIDetector()
        result = await detector.validate("Contact me at john.doe@example.com for details.")
        assert result.passed is False
        assert any(f["type"] == "email" for f in result.details["findings"])

    @pytest.mark.asyncio
    async def test_phone_detected(self):
        detector = PIIDetector()
        result = await detector.validate("Call me at 555-123-4567 or (555) 123-4567.")
        assert result.passed is False
        assert any(f["type"] == "phone_us" for f in result.details["findings"])

    @pytest.mark.asyncio
    async def test_ssn_detected(self):
        detector = PIIDetector()
        result = await detector.validate("SSN: 123-45-6789 was found.")
        assert result.passed is False
        assert any(f["type"] == "ssn" for f in result.details["findings"])

    @pytest.mark.asyncio
    async def test_multiple_pii_types(self):
        detector = PIIDetector()
        output = "Email: test@test.com, Phone: 555-123-7890, SSN: 111-22-3333"
        result = await detector.validate(output)
        assert result.passed is False
        assert result.details["detected_types"] >= 3

    @pytest.mark.asyncio
    async def test_custom_patterns(self):
        detector = PIIDetector(custom_patterns={"api_key": r"sk-[a-zA-Z0-9]{10,}"})
        result = await detector.validate("API key: sk-abcdefghij12345")
        assert result.passed is False
        assert any(f["type"] == "api_key" for f in result.details["findings"])

    @pytest.mark.asyncio
    async def test_validator_name(self):
        detector = PIIDetector()
        assert detector.name == "pii_detector"


class TestToxicityScanner:
    @pytest.mark.asyncio
    async def test_clean_text(self):
        scanner = ToxicityScanner(use_hf_pipeline=False)
        result = await scanner.validate("I hope you have a wonderful day.")
        assert result.passed is True
        assert result.confidence == 1.0

    @pytest.mark.asyncio
    async def test_toxic_word_detected(self):
        scanner = ToxicityScanner(use_hf_pipeline=False)
        result = await scanner.validate("You are a stupid idiot.")
        assert result.passed is False
        assert len(result.details["findings"]) > 0

    @pytest.mark.asyncio
    async def test_toxic_with_context(self):
        scanner = ToxicityScanner(use_hf_pipeline=False)
        # "kill" is in the blocklist
        result = await scanner.validate("I will kill the process if it hangs.")
        assert result.passed is False  # keyword match is naive

    @pytest.mark.asyncio
    async def test_hf_pipeline_fallback(self):
        """When HF pipeline is requested but not installed, should fallback."""
        scanner = ToxicityScanner(use_hf_pipeline=True, model_name="nonexistent")
        scanner._pipeline = None  # force unavailable
        result = await scanner.validate("You are stupid.")
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_validator_name(self):
        scanner = ToxicityScanner()
        assert scanner.name == "toxicity_scanner"
