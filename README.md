# Sentinel — Guardrails & Quality-Gate Framework for Agentic AI

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Sentinel** is a pluggable guardrails and quality-gate framework for agentic AI systems. It validates every output from your AI agents against schema, safety, hallucination, and business-rule checks before it reaches users.

## Architecture

```
┌──────────────────────────────────────────────────┐
│                   Sentinel                        │
│                                                   │
│  ┌───────────┐   ┌───────────┐   ┌────────────┐  │
│  │ Guardrails │   │Observability│  │ Monitoring │  │
│  │  Engine    │   │  (OTel)    │   │  (Drift)   │  │
│  └─────┬─────┘   └─────┬──────┘  └─────┬──────┘  │
│        │               │               │          │
│  ┌─────┴───────────────┴───────────────┴─────┐   │
│  │              FastAPI Server               │   │
│  │        POST /validate  POST /agent        │   │
│  └───────────────────┬───────────────────────┘   │
│                      │                           │
│  ┌───────────────────┴───────────────────────┐   │
│  │  Demo Agent (LangChain) + Multi-Provider  │   │
│  │   Groq · Gemini · OpenAI · Ollama (local) │   │
│  └───────────────────────────────────────────┘   │
│                                                   │
│  ┌──────────────┐          ┌─────────────────┐   │
│  │ CI/CD Quality │          │  Docker Compose  │   │
│  │     Gate      │          │  (Jaeger+Grafana)│   │
│  └──────────────┘          └─────────────────┘   │
└──────────────────────────────────────────────────┘
```

### Component Map

| Component | Location | Responsibility |
|-----------|----------|----------------|
| **Guardrail Engine** | `guardrails/` | Pluggable validators, aggregation, config-driven setup |
| **Demo Agent** | `demo/` | LangChain agent with ChromaDB knowledge base |
| **Multi-Provider** | `demo/providers.py` | Auto-detects Groq → Gemini → OpenAI → Ollama |
| **API Server** | `api/` | FastAPI endpoints for validation and agent execution |
| **Observability** | `observability/` | OpenTelemetry tracing, metrics, OTLP exporters |
| **Drift Monitoring** | `monitoring/` | SQLite-based historical tracking, baselines, alerts |
| **CI/CD** | `.github/workflows/` | Quality gate on PRs with golden set evaluation |
| **Observability Stack** | `docker-compose.yml` | Jaeger + Prometheus + Grafana |

## Quick Start

### 1. Install

```bash
git clone https://github.com/aditya0si/Sentinal.git
cd Sentinal
pip install -e ".[dev]"
```

### 2. Set up an LLM provider (pick one — no API key required for Ollama)

```bash
cp .env.example .env
```

| Provider | Cost | Setup |
|----------|------|-------|
| **Groq** | Free tier | Get key at [console.groq.com](https://console.groq.com/keys), set `GROQ_API_KEY` |
| **Gemini** | Free tier | Get key at [aistudio.google.com](https://aistudio.google.com/apikey), set `GEMINI_API_KEY` |
| **OpenAI** | Paid | Set `OPENAI_API_KEY` |
| **Ollama** | Free (local) | `ollama serve && ollama pull llama3.2` — no key needed |

Sentinel auto-detects the first available provider: **Groq → Gemini → OpenAI → Ollama**.

### 3. Launch observability stack (optional)

```bash
docker-compose up -d
# Jaeger UI:   http://localhost:16686
# Grafana:      http://localhost:3000 (admin/admin)
# Prometheus:   http://localhost:9090
```

### 4. Run the demo

```bash
# Standalone guardrail demo (no LLM needed)
python demo/demo_script.py

# Full agent with guardrails
uvicorn api.main:app --reload --port 8000
```

### 5. Use the API

```bash
# Validate arbitrary text
curl -X POST http://localhost:8000/validate \
  -H "Content-Type: application/json" \
  -d '{"output": "Hello world! This is safe text."}'

# Run the agent with guardrails
curl -X POST http://localhost:8000/agent \
  -H "Content-Type: application/json" \
  -d '{"input": "What is Sentinel?", "enable_guardrails": true}'

# Health check
curl http://localhost:8000/health
```

## Usage Examples

### Programmatic Guardrail Validation

```python
import asyncio
from guardrails.engine import GuardrailEngine
from guardrails.safety import PIIDetector, ToxicityScanner

async def main():
    engine = GuardrailEngine(
        validators=[PIIDetector(), ToxicityScanner()],
        mode="all",
    )
    result = await engine.validate("My email is john@example.com")
    print(f"Passed: {result.overall_pass}")  # False (PII detected)
    print(f"Confidence: {result.aggregate_confidence:.2f}")

asyncio.run(main())
```

### Multi-Provider Agent

```python
from demo.providers import get_chat_model, get_provider_info

# Auto-detects best available provider
llm = get_chat_model()
print(get_provider_info())
# {'provider': 'groq', 'model': 'llama-3.3-70b-versatile'}

# Or explicitly choose a provider
llm = get_chat_model(model_name="llama3.2")  # forces Ollama
```

### YAML/Config-Driven Setup

```python
from guardrails.engine import GuardrailEngine

config = {
    "mode": "all",
    "validators": {
        "pii": {"enabled": True},
        "toxicity": {"enabled": True},
        "rules": {
            "enabled": True,
            "definitions": [
                {"name": "no_spam", "pattern": r"buy now", "severity": "error"},
            ],
        },
    },
}
engine = GuardrailEngine.from_config(config)
```

### Drift Monitoring

```python
from monitoring.store import DriftStore
from monitoring.baseline import RollingBaseline
from monitoring.alerts import DegradationDetector

store = DriftStore()
baseline = RollingBaseline(store=store, window_size=50)
detector = DegradationDetector(store=store, baseline=baseline)

# Record a result
store.record_result(guardrail_result, agent_name="my_agent")

# Check for degradation
alert = detector.check(guardrail_result, agent_name="my_agent")
if alert.level != "INFO":
    print(f"ALERT: {alert}")
```

## Running Tests

```bash
# All tests
python -m pytest tests/ -v

# Specific module
python -m pytest tests/test_engine.py -v

# With coverage
python -m pytest tests/ --cov=guardrails --cov=monitoring
```

## Running the Quality Gate

```bash
# Locally
python scripts/quality_gate.py --threshold 0.85 --output report.json

# In CI (triggered on PRs to main/master)
# See .github/workflows/quality-gate.yml
```

## Demo: Agent With vs. Without Guardrails

```bash
python demo/demo_script.py
```

This demonstrates:
1. An agent output that would leak PII → rejected by guardrails
2. Schema enforcement on structured outputs
3. Business rule violations → blocked before reaching the user

## CI/CD Quality Gate

The GitHub Actions workflow (`.github/workflows/quality-gate.yml`) runs on every PR:
- Executes a golden set of test prompts through the guardrail engine
- Scores outputs via LLM-as-judge rubric (correctness, relevance, safety)
- **Fails the build if aggregate score drops below threshold** — a real quality gate, not a toy test suite

## Validators

| Validator | Class | What it checks |
|-----------|-------|----------------|
| **Schema** | `SchemaValidator` | LLM output matches Pydantic model |
| **Hallucination** | `HallucinationChecker` | Output aligned with retrieved context |
| **PII** | `PIIDetector` | Emails, phones, SSNs, credit cards |
| **Toxicity** | `ToxicityScanner` | Toxic/harmful language |
| **Business Rules** | `RuleValidator` | Custom regex/policy checks |

## Configuration Reference

| Env Variable | Default | Description |
|-------------|---------|-------------|
| `GROQ_API_KEY` | — | Groq API key (free tier) |
| `GEMINI_API_KEY` | — | Google Gemini API key (free tier) |
| `OPENAI_API_KEY` | — | OpenAI API key (paid) |
| `OLLAMA_MODEL` | `llama3.2` | Ollama model name (local, free) |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | — | OTLP collector endpoint |
| `OTEL_TRACING_ENABLED` | `true` | Enable/disable tracing |

## License

MIT
