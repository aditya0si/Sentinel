"""Business rule validator — regex patterns, keyword blocklists, and policy checks."""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Pattern

from guardrails.base import ValidationResult, Validator

logger = logging.getLogger(__name__)


@dataclass
class Rule:
    """A single business rule to check against output.

    Attributes:
        name: Human-readable rule identifier (e.g. "no_competitor_mentions").
        pattern: Regex pattern string or compiled regex.
        description: What the rule checks for.
        severity: "error" (fails validation) or "warn" (logs but passes).
    """

    name: str
    pattern: str | Pattern[str]
    description: str = ""
    severity: str = "error"

    def __post_init__(self) -> None:
        if isinstance(self.pattern, str):
            self.pattern = re.compile(self.pattern)

    def matches(self, text: str) -> list[dict[str, Any]]:
        """Find all matches of this rule's pattern in the text.

        Returns a list of match info dicts with 'match' and 'span'.
        """
        if self.severity == "warn":
            # For warn-level rules using patterns that might "match" broadly,
            # we just check if there are any matches.
            results = list(self.pattern.finditer(text))
        else:
            results = list(self.pattern.finditer(text))

        return [
            {"match": m.group(), "span": (m.start(), m.end())}
            for m in results
        ]


class RuleValidator(Validator):
    """Validates output against a list of business/policy rules.

    Each rule is a regex pattern (or a keyword to block). The validator reports
    which rules were triggered and with what content.

    Usage:
        rules = [
            Rule(name="no_competitors", pattern=r"CompetitorCorp", description="Don't mention competitors"),
            Rule(name="require_disclaimer", pattern=r"(?i)disclaimer", description="Must include disclaimer", severity="warn"),
        ]
        validator = RuleValidator(rules=rules)
        result = await validator.validate("We can beat CompetitorCorp...")
    """

    def __init__(self, rules: list[Rule]) -> None:
        """Initialize the rule validator.

        Args:
            rules: List of Rule objects to check.
        """
        self._rules = rules

    @property
    def name(self) -> str:
        return "business_rules"

    async def validate(
        self, output: str, context: dict[str, Any] | None = None
    ) -> ValidationResult:
        rule_results: list[dict[str, Any]] = []
        failed_rules: list[str] = []
        warned_rules: list[str] = []

        for rule in self._rules:
            matches = rule.matches(output)
            triggered = len(matches) > 0

            rule_result = {
                "rule_name": rule.name,
                "description": rule.description,
                "severity": rule.severity,
                "triggered": triggered,
                "match_count": len(matches),
                "matches": matches[:10],  # limit to 10 matches
            }
            rule_results.append(rule_result)

            if triggered:
                if rule.severity == "error":
                    failed_rules.append(rule.name)
                else:
                    warned_rules.append(rule.name)

        total_error_rules = sum(1 for r in self._rules if r.severity == "error")
        passed = len(failed_rules) == 0
        if total_error_rules > 0:
            confidence = 1.0 - (len(failed_rules) / total_error_rules)
        else:
            confidence = 1.0

        return ValidationResult(
            passed=passed,
            confidence=confidence,
            details={
                "rule_results": rule_results,
                "failed_rules": failed_rules,
                "warned_rules": warned_rules,
                "total_rules": len(self._rules),
            },
            validator_name=self.name,
        )
