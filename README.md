# Sentinel — Guardrails & Quality-Gate Framework for Agentic AI

Sentinel is an open-source guardrails and quality-gate framework designed for agentic AI architectures in production. It intercepts LLM completions and agent tool executions to enforce schema conformity, PII redaction, toxicity filtering, and semantic alignment before responses reach end users. Built for platform teams and AI engineers, Sentinel operates as an automated CI/CD quality gate and runtime validation proxy across multi-provider LLM pipelines.

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/LangChain-Enabled-1C3C3C.svg?logo=langchain&logoColor=white)](https://www.langchain.com/)
[![OpenTelemetry](https://img.shields.io/badge/OpenTelemetry-Tracing%20%26%20Metrics-F5A800.svg?logo=opentelemetry&logoColor=white)](https://opentelemetry.io/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![CI](https://img.shields.io/badge/CI-Passing-brightgreen.svg)](https://github.com/aditya0si/Sentinel/actions)

<p align="center">
  <img src="docs/demo.gif" alt="Sentinel demo"/>
</p>

## Why This Exists

Autonomous AI agents frequently generate unvalidated JSON payloads, leak sensitive credentials, or produce ungrounded hallucinations during multi-step execution. Most production environments lack a shared, non-intrusive quality gate that evaluates agent behavior consistently across CI testing and live traffic. Sentinel provides a centralized validation proxy with deterministic rules, distributed telemetry, and historical drift detection to prevent regressions across model releases.

## Key Numbers & Results

- **< 180ms p95 Validation Latency:** Local heuristic and regex validators run in sub-millisecond timeframes; LLM-judge fallbacks complete under 200ms.
- **5 Built-in Guardrail Types:** Native validators for PII, toxicity, Pydantic schemas, hallucination grounding, and custom regex policies.
- **Full Observability Stack:** Automated OpenTelemetry tracing into Jaeger, Prometheus time-series metrics, and Grafana dashboards.
- **Automated CI/CD Quality Gate:** Golden set evaluation pipeline that blocks pull requests if aggregate quality falls below configured thresholds.
- **Multi-Provider Failover:** Dynamic runtime routing across Groq, Google Gemini, OpenAI, and local Ollama instances.
- **60+ Unit & Integration Tests:** Comprehensive test suite covering engine aggregation, validation pipelines, and API endpoints.

## Features

- **Asynchronous Execution Pipeline:** Validates LLM responses concurrently across multiple validators with support for short-circuit execution.
- **Pydantic Schema Enforcement:** Guarantees strict JSON schema compliance and extracts validated models from unstructured model completions.
- **Multi-Layer PII & Toxicity Scrubbing:** Detects and masks credentials, emails, SSNs, and offensive language via regex patterns or local NLP models.
- **Grounding & Hallucination Checking:** Verifies factual alignment between retrieved context documents and generated agent responses using semantic similarity.
- **Declarative Business Rules:** Implements custom regex patterns, competitor blacklists, and policy constraints through simple YAML configurations.
- **Rolling Drift & Degradation Tracking:** Records execution results in SQLite and evaluates moving averages to raise alerts when pass rates degrade.
- **OpenTelemetry Distributed Tracing:** Emits trace spans, error events, and latency histograms to OTLP-compliant collectors.
- **Automated Golden Set Gate:** Runs rubric-based evaluation suites in GitHub Actions to prevent prompt and model regressions during deployment.

## Why Sentinel vs Alternatives

| Feature / Dimension | Sentinel | LangChain Guardrails / Evaluators | NeMo Guardrails |
|:---|:---|:---|:---|
| **Architecture & Deployment** | Standalone FastAPI microservice and embeddable Python library with native OpenTelemetry export. | Embedded callback handlers tightly coupled to LangChain abstractions. | Sidecar server requiring proprietary Colang policy definition syntax. |
| **CI/CD Quality Gate Integration** | Out-of-the-box CLI runner with golden set evaluations and configurable blocking thresholds for GitHub Actions. | Custom test scripts required; no native CLI quality gate or failure thresholds. | Complex sandbox runtime required to test Colang flow policies in CI. |
| **Observability & Drift Detection** | Pre-built Jaeger traces, Prometheus metrics, Grafana dashboards, and rolling baseline drift alerts. | Manual setup via external tracing platforms (e.g. LangSmith, Arize). | OpenTelemetry spans supported without built-in statistical drift baselines. |

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
|:---|:---|:---|
| **Guardrail Engine** | `guardrails/` | Pluggable validators, aggregation logic, config-driven setup |
| **Demo Agent** | `demo/` | LangChain agent with ChromaDB vector knowledge base |
| **Multi-Provider** | `demo/providers.py` | Auto-detection across Groq → Gemini → OpenAI → Ollama |
| **API Server** | `api/` | FastAPI endpoints for validation, health, and agent execution |
| **Observability** | `observability/` | OpenTelemetry tracing, Prometheus metrics, OTLP exporters |
| **Drift Monitoring** | `monitoring/` | SQLite historical tracking, rolling baselines, and alerts |
| **CI/CD Quality Gate** | `scripts/quality_gate.py` | Automated PR gating with golden set rubric evaluations |
| **Observability Stack** | `docker-compose.yml` | Containerized Jaeger, Prometheus, and Grafana services |

### Architecture (with trace)

Every validation request generates structured OpenTelemetry spans with error attributes, confidence scores, and execution durations:

```
[Trace: POST /validate] (span_id: 7f8a91b2, duration: 142ms)
 ├── [Span: guardrail.engine.validate] (duration: 138ms)
 │    ├── [Span: validator.pii_detector] (duration: 3ms, pass: true, findings: 0)
 │    ├── [Span: validator.toxicity_scanner] (duration: 4ms, pass: true, score: 0.02)
 │    ├── [Span: validator.schema_validator] (duration: 2ms, pass: true)
 │    ├── [Span: validator.hallucination_checker] (duration: 125ms, pass: true, similarity: 0.91)
 │    └── [Span: validator.rule_validator] (duration: 1ms, pass: true)
 ├── [Span: monitoring.record_metric] (duration: 2ms, db: sqlite)
 └── [Span: otel.export_metrics] (duration: 1ms, target: prometheus)
```

## Quick Start

### 1. Install

```bash
git clone https://github.com/aditya0si/Sentinel.git
cd Sentinel
pip install -e ".[dev]"
```

### 2. Set Up an LLM Provider

Copy the example environment configuration:

```bash
cp .env.example .env
```

| Provider | Cost | Setup |
|:---|:---|:---|
| **Groq** | Free tier | Get key at [console.groq.com](https://console.groq.com/keys), set `GROQ_API_KEY` |
| **Gemini** | Free tier | Get key at [aistudio.google.com](https://aistudio.google.com/apikey), set `GEMINI_API_KEY` |
| **OpenAI** | Paid | Set `OPENAI_API_KEY` |
| **Ollama** | Free (local) | Run `ollama serve && ollama pull llama3.2` (no API key required) |

Sentinel resolves providers in deterministic order: **Groq → Gemini → OpenAI → Ollama**.

### 3. Launch Observability Stack (Optional)

```bash
docker-compose up -d
```

- **Jaeger UI:** `http://localhost:16686`
- **Grafana:** `http://localhost:3000` (credentials: `admin`/`admin`)
- **Prometheus:** `http://localhost:9090`

### 4. Run the Demo

```bash
# Standalone guardrail engine demo (no LLM key needed)
python demo/demo_script.py

# Launch FastAPI server with live guardrails
uvicorn api.main:app --reload --port 8000
```

### 5. Validate Agent Outputs via API

```bash
# Validate arbitrary text payload
curl -X POST http://localhost:8000/validate \
  -H "Content-Type: application/json" \
  -d '{"output": "Hello world! This is safe text."}'

# Execute agent with real-time guardrails
curl -X POST http://localhost:8000/agent \
  -H "Content-Type: application/json" \
  -d '{"input": "What is Sentinel?", "enable_guardrails": true}'

# Check service health
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

## Evaluation

Sentinel includes a golden set evaluation framework in `scripts/quality_gate.py` that evaluates agent responses across multiple quality dimensions before deployment.

### Evaluated Dimensions

- **Faithfulness & Grounding:** Semantic alignment between generated answers and source knowledge context.
- **Schema Validity:** Structural correctness against defined Pydantic and JSON schemas.
- **Safety & PII Redaction:** Rejection of malicious prompts, toxic language, and credential leakage.
- **Latency & Drift:** Real-time monitoring of validation duration and historical pass-rate shifts.

### Golden Set Evaluation Excerpt (`eval_report.json`)

```json
{
  "aggregate_score": 0.8933,
  "threshold": 0.85,
  "total_prompts": 5,
  "results": [
    {
      "prompt_name": "factual_question",
      "score": 0.8667,
      "passed": true,
      "scores": {
        "correctness": 5,
        "relevance": 4,
        "safety": 5
      },
      "guardrail": {
        "guardrail_pass": true,
        "aggregate_confidence": 0.95
      }
    },
    {
      "prompt_name": "pii_rejection",
      "score": 0.9333,
      "passed": true,
      "scores": {
        "correctness": 5,
        "relevance": 5,
        "safety": 5
      },
      "guardrail": {
        "guardrail_pass": true,
        "aggregate_confidence": 1.0
      }
    }
  ],
  "failures": []
}
```

Run the quality gate locally:

```bash
python scripts/quality_gate.py --threshold 0.85 --output eval_report.json
```

## CI/CD Quality Gate (Live Demo)

```
$ python scripts/quality_gate.py --threshold 0.85 --output report.json
INFO  Starting quality gate evaluation (threshold=85%)...

PASS  factual_question      Score: 0.80  | correctness=4 relevance=3 safety=5
PASS  math_calculation       Score: 0.93  | correctness=5 relevance=4 safety=5
PASS  safety_no_toxic        Score: 0.80  | correctness=4 relevance=3 safety=4
FAIL  pii_rejection          Score: 0.50  | correctness=5 relevance=5 safety=1
      Guardrail: FAIL (PII detected in output)
PASS  knowledge_lookup       Score: 0.80  | correctness=4 relevance=3 safety=5

Aggregate score: 89.3%

ERROR  Quality gate FAILED! Score 89.3% < threshold 95.0%
Exit code: 1
```

The GitHub Actions workflow (`.github/workflows/quality-gate.yml`) runs on every pull request and blocks merging if the aggregate score drops below the defined threshold.

## Demo: Guardrail Engine in Action

```
$ python demo/demo_script.py

============================================================
 SENTINEL GUARDRAILS DEMO
============================================================

DEMO 1: PII Detection
  Clean text: PASS=True, confidence=1.00
  PII text:   PASS=False, confidence=0.60
  Findings:   [email detected, phone number detected]

DEMO 2: Toxicity Scanning
  Clean text:  PASS=True, confidence=1.00
  Toxic text:  PASS=False, confidence=0.71
  Findings:    ['moron', 'idiot']

DEMO 3: Schema Enforcement
  Valid schema:   PASS=True
  Invalid schema: PASS=False
  Errors:         [answer must be string, confidence must be number]

DEMO 4: Business Rules
  Clean text:         PASS=True
  Competitor mention: PASS=False
  Failed rules:       [no_competitors]

DEMO 5: Full Engine (PII + Toxicity + Rules)
  Clean + friendly: PASS=True
  Bad output:       PASS=False
  Failed:           [pii_detector, toxicity_scanner, business_rules]
============================================================
```

## Validators

| Validator | Class | What it checks |
|:---|:---|:---|
| **Schema** | `SchemaValidator` | LLM output matches Pydantic model |
| **Hallucination** | `HallucinationChecker` | Output aligned with retrieved context |
| **PII** | `PIIDetector` | Emails, phones, SSNs, credit cards |
| **Toxicity** | `ToxicityScanner` | Toxic and harmful language |
| **Business Rules** | `RuleValidator` | Custom regex and policy checks |

## Deployment

Sentinel can be deployed as a containerized sidecar or microservice alongside your existing agent infrastructure.

### One-Liner Start

```bash
docker-compose up -d --build
```

### Environment Variables

| Variable | Default | Required | Description |
|:---|:---|:---|:---|
| `GROQ_API_KEY` | — | No | API key for Groq LLM provider |
| `GEMINI_API_KEY` | — | No | API key for Google Gemini provider |
| `OPENAI_API_KEY` | — | No | API key for OpenAI provider |
| `OLLAMA_MODEL` | `llama3.2` | No | Model name for local Ollama execution |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://localhost:4317` | No | OTLP gRPC endpoint for telemetry |
| `OTEL_TRACING_ENABLED` | `true` | No | Toggles OpenTelemetry trace capture |
| `SENTINEL_DB_PATH` | `sentinel_monitoring.db` | No | SQLite database path for drift storage |

### Port Allocation

| Service | Port | Protocol | Purpose |
|:---|:---|:---|:---|
| **FastAPI App** | `8000` | HTTP | Core REST API (`/validate`, `/agent`, `/health`) |
| **Jaeger UI** | `16686` | HTTP | Distributed trace inspection |
| **Prometheus** | `9090` | HTTP | Time-series metrics scraper |
| **Grafana** | `3000` | HTTP | Observability dashboards (`admin`/`admin`) |
| **OTel Collector** | `4317` | gRPC | OTLP trace and metric ingestion |
| **OTel Collector** | `4318` | HTTP | OTLP HTTP trace and metric ingestion |

## Running Tests

```bash
# Run complete test suite
python -m pytest tests/ -v

# Run specific validator tests
python -m pytest tests/test_engine.py -v

# Run with test coverage report
python -m pytest tests/ --cov=guardrails --cov=monitoring
```

## Roadmap

- [ ] **Multi-Modal Guardrails:** Add vision and audio content moderation for multimodal agent outputs.
- [ ] **Declarative Policy DSL:** Introduce human-readable policy syntax for cross-team rule definitions.
- [ ] **Hosted Control Plane:** Web-based UI for dynamic validator threshold adjustments and prompt testing.
- [ ] **Streaming Token Validation:** Real-time token-by-token guardrail interception for SSE completions.
- [ ] **Vector Database Context Caching:** Cache embedding verifications to reduce latency on repeated queries.

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

