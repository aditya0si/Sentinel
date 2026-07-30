"""Tests for the SchemaValidator."""

import pytest
from pydantic import BaseModel

from guardrails.schema import SchemaValidator


class SimpleOutput(BaseModel):
    answer: str
    confidence: float


class NestedOutput(BaseModel):
    summary: str
    items: list[str]
    metadata: dict


class TestSchemaValidator:
    @pytest.mark.asyncio
    async def test_valid_json_parsing(self):
        validator = SchemaValidator(schema=SimpleOutput, parse_mode="json")
        output = '{"answer": "hello world", "confidence": 0.95}'
        result = await validator.validate(output)

        assert result.passed is True
        assert result.confidence == 1.0
        assert result.details["parsed"]["answer"] == "hello world"
        assert result.details["parsed"]["confidence"] == 0.95

    @pytest.mark.asyncio
    async def test_invalid_json_missing_field(self):
        validator = SchemaValidator(schema=SimpleOutput, parse_mode="json")
        output = '{"answer": "hello"}'  # missing confidence
        result = await validator.validate(output)

        assert result.passed is False
        assert result.confidence == 0.0
        assert "error_info" in result.details

    @pytest.mark.asyncio
    async def test_invalid_json_wrong_type(self):
        validator = SchemaValidator(schema=SimpleOutput, parse_mode="json")
        output = '{"answer": 123, "confidence": "high"}'  # wrong types
        result = await validator.validate(output)

        assert result.passed is False

    @pytest.mark.asyncio
    async def test_malformed_json(self):
        validator = SchemaValidator(schema=SimpleOutput, parse_mode="json")
        output = "not valid json at all {{{"
        result = await validator.validate(output)

        assert result.passed is False
        assert "error" in result.details

    @pytest.mark.asyncio
    async def test_dict_parse_mode(self):
        validator = SchemaValidator(schema=SimpleOutput, parse_mode="dict")
        output = '{"answer": "test", "confidence": 0.5}'
        result = await validator.validate(output)

        assert result.passed is True

    @pytest.mark.asyncio
    async def test_nested_schema_valid(self):
        validator = SchemaValidator(schema=NestedOutput, parse_mode="json")
        output = '{"summary": "test", "items": ["a", "b"], "metadata": {"source": "web"}}'
        result = await validator.validate(output)

        assert result.passed is True

    @pytest.mark.asyncio
    async def test_validator_name(self):
        validator = SchemaValidator(schema=SimpleOutput)
        assert "SimpleOutput" in validator.name
