#!/usr/bin/env python3
"""Demo script showing Sentinel guardrails in action.

Compares agent behavior with and without guardrails:
1. PII detection: output containing an email → blocked
2. Toxicity: output containing toxic language → blocked
3. Schema enforcement: output that doesn't match expected format → blocked
4. Clean output: passes all checks

Usage:
    python demo/demo_script.py
"""

import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("demo")


async def demo_pii_detection():
    """Demonstrate PII detection blocking output."""
    from guardrails.engine import GuardrailEngine
    from guardrails.safety import PIIDetector

    print("\n" + "=" * 60)
    print(" DEMO 1: PII Detection")
    print("=" * 60)

    engine = GuardrailEngine(validators=[PIIDetector()], mode="all")

    # Clean output
    result_clean = await engine.validate("The meeting is scheduled for 3 PM tomorrow.")
    print(f"\n  Clean text: PASS={result_clean.overall_pass}, confidence={result_clean.aggregate_confidence:.2f}")

    # Output with PII
    result_pii = await engine.validate(
        "Please send the report to john.doe@company.com or call 555-123-4567."
    )
    print(f"  PII text: PASS={result_pii.overall_pass}, confidence={result_pii.aggregate_confidence:.2f}")
    print(f"  Findings: {result_pii.results[0].details.get('findings', [])}")

    return result_pii


async def demo_toxicity():
    """Demonstrate toxicity scanning."""
    from guardrails.engine import GuardrailEngine
    from guardrails.safety import ToxicityScanner

    print("\n" + "=" * 60)
    print(" DEMO 2: Toxicity Scanning")
    print("=" * 60)

    engine = GuardrailEngine(validators=[ToxicityScanner()], mode="all")

    # Clean output
    result_clean = await engine.validate("Thank you for your help today!")
    print(f"\n  Clean text: PASS={result_clean.overall_pass}, confidence={result_clean.aggregate_confidence:.2f}")

    # Toxic output
    result_toxic = await engine.validate("You are a complete idiot and a moron.")
    print(f"  Toxic text: PASS={result_toxic.overall_pass}, confidence={result_toxic.aggregate_confidence:.2f}")
    print(f"  Findings: {result_toxic.results[0].details.get('findings', [])}")

    return result_toxic


async def demo_schema_enforcement():
    """Demonstrate schema validation."""
    from guardrails.engine import GuardrailEngine
    from guardrails.schema import SchemaValidator
    from pydantic import BaseModel

    print("\n" + "=" * 60)
    print(" DEMO 3: Schema Enforcement")
    print("=" * 60)

    class AgentResponse(BaseModel):
        answer: str
        confidence: float
        sources: list[str] = []

    engine = GuardrailEngine(validators=[SchemaValidator(schema=AgentResponse)], mode="all")

    # Valid output
    valid_json = '{"answer": "Paris", "confidence": 0.95, "sources": ["wikipedia"]}'
    result_valid = await engine.validate(valid_json)
    print(f"\n  Valid schema: PASS={result_valid.overall_pass}")

    # Invalid output (wrong type)
    invalid_json = '{"answer": 42, "confidence": "high"}'
    result_invalid = await engine.validate(invalid_json)
    print(f"  Invalid schema: PASS={result_invalid.overall_pass}")
    print(f"  Errors: {result_invalid.results[0].details.get('error_info', {}).get('errors', [])}")

    return result_invalid


async def demo_business_rules():
    """Demonstrate business rule enforcement."""
    from guardrails.engine import GuardrailEngine
    from guardrails.business_rules import Rule, RuleValidator

    print("\n" + "=" * 60)
    print(" DEMO 4: Business Rules")
    print("=" * 60)

    rules = [
        Rule(name="no_competitors", pattern=r"CompetitorX", description="Never mention competitors"),
        Rule(name="require_disclaimer", pattern=r"(?i)disclaimer", severity="warn", description="Should include disclaimer"),
    ]
    engine = GuardrailEngine(validators=[RuleValidator(rules=rules)], mode="all")

    # Clean output
    result_clean = await engine.validate("Our product is the best in the market.")
    print(f"\n  Clean text: PASS={result_clean.overall_pass}")

    # Trigger a rule
    result_bad = await engine.validate("Our product is better than CompetitorX and we dominate.")
    print(f"  Competitor mention: PASS={result_bad.overall_pass}")
    print(f"  Failed rules: {result_bad.results[0].details.get('failed_rules', [])}")

    return result_bad


async def demo_full_engine():
    """Demonstrate the full engine with multiple validators."""
    from guardrails.engine import GuardrailEngine
    from guardrails.safety import PIIDetector, ToxicityScanner
    from guardrails.business_rules import Rule, RuleValidator

    print("\n" + "=" * 60)
    print(" DEMO 5: Full Engine (PII + Toxicity + Rules)")
    print("=" * 60)

    rules = [
        Rule(name="no_spam", pattern=r"(?i)buy now|click here", severity="error"),
        Rule(name="friendly_tone", pattern=r"(?i)thank you|please", severity="warn"),
    ]
    engine = GuardrailEngine(
        validators=[PIIDetector(), ToxicityScanner(), RuleValidator(rules=rules)],
        mode="all",
    )

    # Clean with rule match (friendly tone)
    result_clean = await engine.validate("Thank you for your inquiry. Our team will respond shortly.")
    print(f"\n  Clean + friendly: PASS={result_clean.overall_pass}")

    # All three problems
    result_bad = await engine.validate(
        "Click here now! Your stupid product leaked my email john@spam.com!"
    )
    print(f"  Bad output: PASS={result_bad.overall_pass}")
    print(f"  Failed validators: {result_bad.failed_validators}")

    return result_bad


async def main():
    print("=" * 60)
    print(" SENTINEL GUARDRAILS DEMO")
    print("=" * 60)
    print()
    print("This demo shows Sentinel blocking various types of problematic")
    print("AI agent outputs before they reach the user.")
    print()

    await demo_pii_detection()
    await demo_toxicity()
    await demo_schema_enforcement()
    await demo_business_rules()
    await demo_full_engine()

    print("\n" + "=" * 60)
    print(" DEMO COMPLETE")
    print("=" * 60)
    print()
    print("Sentinel successfully caught and blocked:")
    print("  1. PII (email, phone number)")
    print("  2. Toxic language")
    print("  3. Schema violations")
    print("  4. Business rule violations")
    print()
    print("Run the API server to try it interactively:")
    print("  uvicorn api.main:app --reload --port 8000")
    print()


if __name__ == "__main__":
    asyncio.run(main())
