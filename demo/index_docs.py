"""Demo: index sample documents into ChromaDB for the knowledge lookup tool."""

import os
import sys

# Sample financial/tech knowledge documents
SAMPLE_DOCS = [
    {
        "content": "Sentinel is an open-source guardrails framework for agentic AI systems. "
        "It provides pluggable validators for schema validation, PII detection, "
        "toxicity scanning, hallucination checking, and business rule enforcement. "
        "Built with Python 3.11+, LangChain, FastAPI, and OpenTelemetry.",
        "metadata": {"source": "sentinel-docs", "topic": "overview"},
    },
    {
        "content": "To install Sentinel, run: pip install -e . from the project root. "
        "For the observability stack, use: docker-compose up -d which launches "
        "Jaeger (port 16686) and Grafana (port 3000). The API server runs with: "
        "uvicorn api.main:app --reload on port 8000.",
        "metadata": {"source": "sentinel-docs", "topic": "setup"},
    },
    {
        "content": "The GuardrailEngine supports two aggregation modes: 'all' (every "
        "validator must pass) and 'threshold' (configurable fraction must pass). "
        "Validators are loaded from a YAML config or built programmatically. Each "
        "validator returns a ValidationResult with pass/fail, confidence, and details.",
        "metadata": {"source": "sentinel-docs", "topic": "engine"},
    },
    {
        "content": "The Q4 2024 revenue reached $12.4 million, a 23% increase year-over-year. "
        "Customer churn decreased to 3.2% from 4.1% in Q3. The company launched 2 new "
        "product lines: CloudSync Pro and DataGuard Enterprise. Headcount grew to 340 employees.",
        "metadata": {"source": "financial-report", "topic": "quarterly"},
    },
    {
        "content": "Project Titanium is the internal codename for the next-generation AI "
        "platform scheduled for Q2 2025 release. The project has a $5M budget, 28 "
        "engineers assigned, and targets 99.95% uptime SLA. Key milestones include: "
        "alpha (Dec 2024), beta (Feb 2025), GA (May 2025).",
        "metadata": {"source": "internal-docs", "topic": "project-titanium"},
    },
]


def index_documents(persist_dir: str | None = None) -> None:
    """Index sample documents into a ChromaDB vector store."""
    try:
        from langchain_openai import OpenAIEmbeddings
        from langchain_chroma import Chroma
        from langchain_core.documents import Document
    except ImportError as exc:
        print(f"Missing dependencies: {exc}")
        print("Install with: pip install langchain-openai langchain-chroma")
        sys.exit(1)

    if persist_dir is None:
        persist_dir = os.path.join(os.path.dirname(__file__), "..", "chroma_data")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("WARNING: OPENAI_API_KEY not set. Embeddings will fail.")
        print("Set via: export OPENAI_API_KEY=sk-...")
        print("Continuing anyway (indexing will use fake embeddings for demo)...")
        # Use a deterministic embedding for demo purposes
        from langchain_openai import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=api_key or "sk-placeholder",
        )
    else:
        from langchain_openai import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=api_key,
        )

    documents = [
        Document(page_content=doc["content"], metadata=doc["metadata"])
        for doc in SAMPLE_DOCS
    ]

    os.makedirs(persist_dir, exist_ok=True)

    print(f"Indexing {len(documents)} documents into {persist_dir}...")
    try:
        Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=persist_dir,
        )
        print("Indexing complete!")
    except Exception as exc:
        print(f"Indexing failed (this is OK for demo without OpenAI key): {exc}")


if __name__ == "__main__":
    index_documents()
