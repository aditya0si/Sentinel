"""SchemaValidator — validates LLM output against a Pydantic model."""

import json
import logging
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from guardrails.base import ValidationResult, Validator

logger = logging.getLogger(__name__)


class SchemaValidator(Validator):
    """Validates that LLM output conforms to a given Pydantic model schema.

    The validator attempts to parse the output (either a JSON string or a dict)
    into the provided Pydantic model. If parsing succeeds, the validator passes
    with high confidence. If it fails, details about missing/wrong fields are returned.

    Usage:
        class MyOutput(BaseModel):
            answer: str
            confidence: float

        validator = SchemaValidator(schema=MyOutput, parse_mode="json")
        result = await validator.validate('{"answer": "hello", "confidence": 0.9}')
    """

    def __init__(
        self,
        schema: type[BaseModel],
        parse_mode: Literal["json", "dict"] = "json",
    ) -> None:
        """Initialize the schema validator.

        Args:
            schema: A Pydantic BaseModel subclass defining the expected output schema.
            parse_mode: How to interpret the output string ("json" or "dict").
        """
        self._schema = schema
        self._parse_mode = parse_mode

    @property
    def name(self) -> str:
        return f"schema:{self._schema.__name__}"

    async def validate(
        self, output: str, context: dict[str, Any] | None = None
    ) -> ValidationResult:
        try:
            if self._parse_mode == "json":
                parsed = self._schema.model_validate_json(output)
            else:
                data = json.loads(output)
                parsed = self._schema.model_validate(data)

            return ValidationResult(
                passed=True,
                confidence=1.0,
                details={"parsed": parsed.model_dump()},
                validator_name=self.name,
            )

        except (json.JSONDecodeError, ValidationError) as exc:
            error_info = SchemaValidator._extract_error_info(exc)
            logger.debug("Schema validation failed for %s: %s", self.name, error_info)

            return ValidationResult(
                passed=False,
                confidence=0.0,
                details={"error": str(exc), "error_info": error_info},
                validator_name=self.name,
            )

        except Exception as exc:
            logger.error("Unexpected error in SchemaValidator %s: %s", self.name, exc)
            return ValidationResult(
                passed=False,
                confidence=0.0,
                details={"error": str(exc)},
                validator_name=self.name,
            )

    @staticmethod
    def _extract_error_info(exc: Exception) -> dict[str, Any]:
        """Extract structured error information from parsing/validation exceptions."""
        if isinstance(exc, ValidationError):
            return {
                "error_count": exc.error_count(),
                "errors": [
                    {"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]}
                    for e in exc.errors()
                ],
            }
        if isinstance(exc, json.JSONDecodeError):
            return {
                "error_count": 1,
                "errors": [
                    {
                        "loc": [f"line {exc.lineno}, col {exc.colno}"],
                        "msg": exc.msg,
                        "type": "json_decode_error",
                    }
                ],
            }
        return {"error_count": 1, "errors": [{"msg": str(exc)}]}
