"""Tests for the HallucinationChecker."""

import pytest
from guardrails.hallucination import HallucinationChecker


class TestHallucinationChecker:
    @pytest.mark.asyncio
    async def test_embedding_mode_no_context(self):
        """When no context is provided, the checker should skip and pass."""
        checker = HallucinationChecker(mode="embedding", threshold=0.7)
        result = await checker.validate("The sky is blue.")
        assert result.passed is True
        assert "No context documents provided" in result.details.get("warning", "")

    @pytest.mark.asyncio
    async def test_embedding_mode_model_not_available(self):
        """When sentence-transformers is not installed, should handle gracefully."""
        checker = HallucinationChecker(mode="embedding", threshold=0.7, model_name="nonexistent-model")
        checker._embedding_model = None  # force unavailable
        result = await checker.validate(
            "output text",
            context={"retrieved_documents": ["context document"]},
        )
        assert result.passed is True  # passes with warning
        assert result.confidence == 0.5

    @pytest.mark.asyncio
    async def test_llm_judge_no_api_key(self):
        """LLM judge mode without API key should fail gracefully."""
        checker = HallucinationChecker(mode="llm_judge", threshold=3, openai_api_key=None)
        result = await checker.validate(
            "output",
            context={"retrieved_documents": ["doc"]},
        )
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_unknown_mode(self):
        checker = HallucinationChecker(mode="unknown_mode")
        result = await checker.validate("output")
        assert result.passed is False
        assert "Unknown mode" in result.details.get("error", "")

    @pytest.mark.asyncio
    async def test_validator_name(self):
        checker = HallucinationChecker(mode="embedding")
        assert "hallucination:embedding" in checker.name
