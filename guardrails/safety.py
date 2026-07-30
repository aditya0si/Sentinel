"""Safety validators — PII detection and toxicity scanning."""

import logging
import re
from typing import Any

from guardrails.base import ValidationResult, Validator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pre-compiled regex patterns for common PII types
# ---------------------------------------------------------------------------
PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", re.IGNORECASE),
    "phone_us": re.compile(
        r"\b(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
    ),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
    "ip_address": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
}


class PIIDetector(Validator):
    """Detects Personally Identifiable Information (PII) in LLM output.

    Uses regex patterns for common PII types (emails, phones, SSNs, credit cards)
    and optionally integrates with Microsoft Presidio for advanced PII detection.

    The validator fails if *any* PII is detected. Confidence score is computed
    as 1.0 - (number of PII types detected / total types checked).

    Usage:
        detector = PIIDetector(use_presidio=False)
        result = await detector.validate("My email is john@example.com")
    """

    def __init__(
        self,
        use_presidio: bool = False,
        custom_patterns: dict[str, str] | None = None,
    ) -> None:
        """Initialize the PII detector.

        Args:
            use_presidio: Whether to use Microsoft Presidio for advanced detection.
            custom_patterns: Additional regex patterns {name: pattern_string}.
        """
        self._use_presidio = use_presidio
        self._patterns: dict[str, re.Pattern[str]] = dict(PII_PATTERNS)
        if custom_patterns:
            for name, pattern in custom_patterns.items():
                self._patterns[name] = re.compile(pattern)

        self._presidio_loaded = False
        if use_presidio:
            self._presidio_loaded = self._init_presidio()

    @property
    def name(self) -> str:
        return "pii_detector"

    def _init_presidio(self) -> bool:  # type: ignore[return]
        """Attempt to initialize Presidio. Returns True if available."""
        try:
            # Check if presidio is importable (don't actually load the model yet)
            import importlib.util

            spec = importlib.util.find_spec("presidio_analyzer")
            if spec is None:
                logger.warning("presidio_analyzer not installed. Using regex-only mode.")
                return False
            logger.debug("Presidio is available for advanced PII detection.")
            return True
        except Exception as exc:
            logger.warning("Presidio initialization skipped: %s", exc)
            return False

    async def validate(
        self, output: str, context: dict[str, Any] | None = None
    ) -> ValidationResult:
        findings: list[dict[str, Any]] = []
        total_types = len(self._patterns)
        detected_types = 0

        for pii_type, pattern in self._patterns.items():
            matches = pattern.findall(output)
            if matches:
                detected_types += 1
                findings.append(
                    {
                        "type": pii_type,
                        "count": len(matches),
                        "matches": matches[:5],  # limit to 5 examples
                    }
                )

        # Presidio analysis
        if self._use_presidio and self._presidio_loaded:
            presidio_findings = await self._run_presidio(output)
            findings.extend(presidio_findings)

        has_pii = detected_types > 0
        # Confidence reflects detection certainty: more types detected = higher confidence
        confidence = detected_types / max(total_types, 1) if has_pii else 1.0

        return ValidationResult(
            passed=not has_pii,
            confidence=confidence,
            details={
                "findings": findings,
                "total_types_checked": total_types,
                "detected_types": detected_types,
                "presidio_used": self._use_presidio and self._presidio_loaded,
            },
            validator_name=self.name,
        )

    async def _run_presidio(self, text: str) -> list[dict[str, Any]]:
        """Run Presidio analyzer on the text."""
        try:
            from presidio_analyzer import AnalyzerEngine

            analyzer = AnalyzerEngine()
            results = analyzer.analyze(text=text, language="en")
            return [
                {
                    "type": r.entity_type,
                    "score": r.score,
                    "start": r.start,
                    "end": r.end,
                    "text": text[r.start : r.end],
                    "source": "presidio",
                }
                for r in results
            ]
        except Exception as exc:
            logger.warning("Presidio analysis failed: %s", exc)
            return []


# ---------------------------------------------------------------------------
# Toxicity Scanner
# ---------------------------------------------------------------------------

# Minimal fallback word list (used when HF pipeline is unavailable)
_FALLBACK_TOXIC_WORDS: set[str] = {
    "kill", "murder", "bomb", "terrorist", "hate", "racist", "sexist",
    "stupid", "idiot", "moron", "asshole", "bastard", "fuck", "shit",
    "damn", "crap", "dumb", "ugly", "fat", "loser",
}


class ToxicityScanner(Validator):
    """Scans LLM output for toxic or harmful content.

    Supports two detection methods:
    1. **HuggingFace pipeline** — Use a pretrained toxicity classifier
       (default: ``unitary/toxic-bert``). Falls back gracefully if unavailable.
    2. **Rule-based fallback** — Checks for known toxic words as a lightweight alternative.

    Usage:
        scanner = ToxicityScanner(use_hf_pipeline=False)
        result = await scanner.validate("You are stupid!")
    """

    def __init__(
        self,
        use_hf_pipeline: bool = False,
        model_name: str = "unitary/toxic-bert",
        threshold: float = 0.7,
    ) -> None:
        """Initialize the toxicity scanner.

        Args:
            use_hf_pipeline: Whether to use a HuggingFace pipeline.
            model_name: HuggingFace model ID for toxicity classification.
            threshold: Minimum toxicity score to flag as toxic (0.0-1.0).
        """
        self._use_hf_pipeline = use_hf_pipeline
        self._model_name = model_name
        self._threshold = threshold
        self._pipeline = None
        if use_hf_pipeline:
            self._pipeline = self._load_pipeline()

    @property
    def name(self) -> str:
        return "toxicity_scanner"

    def _load_pipeline(self):  # type: ignore[no-untyped-def]
        """Attempt to load the HuggingFace toxicity pipeline."""
        try:
            from transformers import pipeline

            logger.debug("Loading toxicity pipeline: %s", self._model_name)
            return pipeline("text-classification", model=self._model_name)
        except ImportError:
            logger.warning(
                "transformers not installed. Using rule-based fallback for toxicity."
            )
            return None
        except Exception as exc:
            logger.warning(
                "Failed to load toxicity pipeline: %s. Using rule-based fallback.", exc
            )
            return None

    async def validate(
        self, output: str, context: dict[str, Any] | None = None
    ) -> ValidationResult:
        if self._use_hf_pipeline and self._pipeline is not None:
            return await self._validate_hf(output)
        else:
            return await self._validate_rule_based(output)

    async def _validate_hf(self, output: str) -> ValidationResult:
        """Validate using HuggingFace toxicity pipeline."""
        try:
            result = self._pipeline(output)[0]  # type: ignore[index]
            score = result["score"]
            is_toxic = result["label"].upper() == "TOXIC"
            passed = not is_toxic or score < self._threshold
            confidence = 1.0 - score if is_toxic else 1.0

            return ValidationResult(
                passed=passed,
                confidence=confidence,
                details={
                    "label": result["label"],
                    "score": score,
                    "threshold": self._threshold,
                    "mode": "huggingface",
                },
                validator_name=self.name,
            )
        except Exception as exc:
            logger.error("HF toxicity check failed: %s", exc)
            return await self._validate_rule_based(output)

    async def _validate_rule_based(self, output: str) -> ValidationResult:
        """Fallback rule-based toxicity check using keyword blocklist."""
        import re as _re

        output_lower = output.lower()
        # Strip punctuation so "stupid." matches "stupid"
        output_clean = _re.sub(r"[^\w\s]", "", output_lower)
        words = set(output_clean.split())
        matched = words & _FALLBACK_TOXIC_WORDS

        if matched:
            confidence = 1.0 - (len(matched) / max(len(words), 1))
            passed = False
        else:
            confidence = 1.0
            passed = True

        return ValidationResult(
            passed=passed,
            confidence=confidence,
            details={
                "findings": list(matched) if matched else [],
                "mode": "rule_based",
                "threshold": self._threshold,
            },
            validator_name=self.name,
        )
