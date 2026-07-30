"""Tests for the RuleValidator and business rule engine."""

import pytest
from guardrails.business_rules import Rule, RuleValidator


class TestRuleValidator:
    @pytest.mark.asyncio
    async def test_no_rules_triggered(self):
        rules = [Rule(name="no_bad_words", pattern=r"badword")]
        validator = RuleValidator(rules=rules)
        result = await validator.validate("This is clean text.")
        assert result.passed is True
        assert result.confidence == 1.0

    @pytest.mark.asyncio
    async def test_rule_triggered(self):
        rules = [Rule(name="no_competitors", pattern=r"CompetitorX")]
        validator = RuleValidator(rules=rules)
        result = await validator.validate("We are better than CompetitorX!")
        assert result.passed is False
        assert "no_competitors" in result.details["failed_rules"]

    @pytest.mark.asyncio
    async def test_warn_severity_passes(self):
        rules = [
            Rule(name="check_disclaimer", pattern=r"(?i)disclaimer", severity="warn")
        ]
        validator = RuleValidator(rules=rules)
        result = await validator.validate("No disclaimer in this text.")
        assert result.passed is True  # warn doesn't fail
        assert "check_disclaimer" in result.details["warned_rules"]

    @pytest.mark.asyncio
    async def test_warn_severity_when_triggered(self):
        rules = [
            Rule(name="check_disclaimer", pattern=r"(?i)disclaimer", severity="warn")
        ]
        validator = RuleValidator(rules=rules)
        result = await validator.validate("This text has a DISCLAIMER.")
        assert result.passed is True
        assert "check_disclaimer" in result.details["warned_rules"]

    @pytest.mark.asyncio
    async def test_multiple_rules(self):
        rules = [
            Rule(name="no_foul_language", pattern=r"crap"),
            Rule(name="require_greeting", pattern=r"(?i)hello|hi"),
        ]
        validator = RuleValidator(rules=rules)
        result = await validator.validate("This is crap! Hello there.")
        assert result.passed is False  # "crap" triggered
        assert "no_foul_language" in result.details["failed_rules"]

    @pytest.mark.asyncio
    async def test_confidence_calibration(self):
        rules = [
            Rule(name="rule_a", pattern=r"bad"),
            Rule(name="rule_b", pattern=r"terrible"),
        ]
        validator = RuleValidator(rules=rules)
        result = await validator.validate("This is bad.")
        assert result.confidence == 0.5  # 1 of 2 failed

        result2 = await validator.validate("This is bad and terrible.")
        assert result2.confidence == 0.0  # both failed

    @pytest.mark.asyncio
    async def test_match_details(self):
        rules = [Rule(name="find_numbers", pattern=r"\d+")]
        validator = RuleValidator(rules=rules)
        result = await validator.validate("There are 42 apples and 7 oranges.")
        assert result.passed is False
        rule_result = result.details["rule_results"][0]
        assert rule_result["match_count"] == 2

    @pytest.mark.asyncio
    async def test_validator_name(self):
        validator = RuleValidator(rules=[])
        assert validator.name == "business_rules"
