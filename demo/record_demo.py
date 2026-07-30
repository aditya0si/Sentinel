#!/usr/bin/env python3
"""Demo script for recording Sentinel in action.

Run this script and record your terminal with asciinema or any screen recorder:
  asciinema rec -c "python demo/record_demo.py" demo.cast
  agg demo.cast demo.gif

Or use OBS/ScreenRec to capture the output and convert to GIF.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


async def main():
    from guardrails.engine import GuardrailEngine
    from guardrails.safety import PIIDetector, ToxicityScanner
    from guardrails.schema import SchemaValidator
    from guardrails.business_rules import Rule, RuleValidator
    from pydantic import BaseModel

    class AgentResponse(BaseModel):
        answer: str
        confidence: float

    print("=" * 60)
    print("  SENTINEL — Guardrails & Quality-Gate Demo")
    print("=" * 60)
    print()

    # --- Demo 1: PII Detection ---
    print("━" * 60)
    print("  DEMO 1: PII Detection")
    print("━" * 60)
    engine = GuardrailEngine(validators=[PIIDetector()], mode="all")

    result = await engine.validate("The meeting is at 3 PM.")
    print(f"  ✅ Clean text   → PASS (confidence={result.aggregate_confidence:.2f})")

    result = await engine.validate("Email john@acme.com or call 555-123-4567.")
    print(f"  🚫 PII detected → FAIL (confidence={result.aggregate_confidence:.2f})")
    print(f"     Findings: {result.results[0].details.get('findings', [])}")
    print()

    # --- Demo 2: Toxicity ---
    print("━" * 60)
    print("  DEMO 2: Toxicity Scanning")
    print("━" * 60)
    engine = GuardrailEngine(validators=[ToxicityScanner()], mode="all")

    result = await engine.validate("Thank you for your help!")
    print(f"  ✅ Clean text    → PASS (confidence={result.aggregate_confidence:.2f})")

    result = await engine.validate("You are a complete idiot.")
    print(f"  🚫 Toxic content → FAIL (confidence={result.aggregate_confidence:.2f})")
    print(f"     Findings: {result.results[0].details.get('findings', [])}")
    print()

    # --- Demo 3: Schema Enforcement ---
    print("━" * 60)
    print("  DEMO 3: Schema Enforcement")
    print("━" * 60)
    engine = GuardrailEngine(validators=[SchemaValidator(schema=AgentResponse)], mode="all")

    result = await engine.validate('{"answer": "Paris", "confidence": 0.95}')
    print(f"  ✅ Valid JSON    → PASS")

    result = await engine.validate('{"answer": 42, "confidence": "high"}')
    print(f"  🚫 Wrong types  → FAIL")
    print()

    # --- Demo 4: Business Rules ---
    print("━" * 60)
    print("  DEMO 4: Business Rules")
    print("━" * 60)
    rules = [
        Rule(name="no_competitors", pattern=r"CompetitorX", description="Never mention competitors"),
    ]
    engine = GuardrailEngine(validators=[RuleValidator(rules=rules)], mode="all")

    result = await engine.validate("Our product is the best.")
    print(f"  ✅ Clean text         → PASS")

    result = await engine.validate("We beat CompetitorX easily.")
    print(f"  🚫 Competitor mention → FAIL")
    print(f"     Failed rules: {result.results[0].details.get('failed_rules', [])}")
    print()

    # --- Demo 5: Full Engine ---
    print("━" * 60)
    print("  DEMO 5: Full Engine (PII + Toxicity + Rules)")
    print("━" * 60)
    engine = GuardrailEngine(
        validators=[
            PIIDetector(),
            ToxicityScanner(),
            RuleValidator(rules=[
                Rule(name="no_spam", pattern=r"(?i)buy now|click here", severity="error"),
            ]),
        ],
        mode="all",
    )

    result = await engine.validate("Thank you for your inquiry!")
    print(f"  ✅ Clean + friendly → PASS")

    result = await engine.validate("Click here now! Your stupid product leaked my email john@spam.com!")
    print(f"  🚫 Multiple issues  → FAIL")
    print(f"     Failed: {result.failed_validators}")
    print()

    # --- Summary ---
    print("=" * 60)
    print("  SENTINEL CAUGHT:")
    print("    • PII (email, phone number)")
    print("    • Toxic language")
    print("    • Schema violations")
    print("    • Business rule violations")
    print("=" * 60)
    print()
    print("  GitHub: https://github.com/aditya0si/Sentinal")
    print()


if __name__ == "__main__":
    asyncio.run(main())
