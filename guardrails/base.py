"""Base classes for Sentinel guardrail validators."""

import logging
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ValidationResult(BaseModel):
    """Result of a single validator check.

    Attributes:
        passed: Whether the output passed this validator.
        confidence: Confidence score (0.0 - 1.0).
        details: Additional information about the validation (findings, errors, etc.).
        validator_name: Name of the validator that produced this result.
    """

    passed: bool
    confidence: float = Field(ge=0.0, le=1.0)
    details: dict[str, Any] = Field(default_factory=dict)
    validator_name: str = ""

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.validator_name} (confidence={self.confidence:.2f})"


class Validator(ABC):
    """Abstract base class for all Sentinel validators.

    Subclasses must implement:
        async def validate(self, output: str, context: dict | None = None) -> ValidationResult

    The `name` property must be set to a unique, human-readable identifier.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique name of this validator (e.g. 'schema', 'hallucination', 'pii')."""
        ...

    @abstractmethod
    async def validate(
        self, output: str, context: dict[str, Any] | None = None
    ) -> ValidationResult:
        """Validate the given output against this guardrail.

        Args:
            output: The text/output to validate.
            context: Optional context data (retrieved docs, metadata, etc.).

        Returns:
            A ValidationResult indicating pass/fail and confidence level.
        """
        ...

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name}>"
