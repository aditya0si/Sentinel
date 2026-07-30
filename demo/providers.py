"""Multi-provider LLM factory for Sentinel demo agent.

Supports OpenAI, Groq (free tier), Ollama (local), and Google Gemini.
Configured via environment variables — falls back gracefully.

Priority: GROQ_API_KEY > GEMINI_API_KEY > OPENAI_API_KEY > Ollama (local)
"""

import logging
import os

from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)


def get_chat_model(
    model_name: str | None = None,
    temperature: float = 0.3,
) -> BaseChatModel:
    """Create a chat model from the best available provider.

    Provider detection order:
    1. GROQ_API_KEY  → Groq (free tier, fast, llama-3.3-70b-versatile)
    2. GEMINI_API_KEY → Google Gemini (free tier, gemini-2.0-flash)
    3. OPENAI_API_KEY → OpenAI (gpt-4o-mini)
    4. None → Ollama local (llama3.2, must be running locally)

    Args:
        model_name: Override the default model for the detected provider.
        temperature: Sampling temperature.

    Returns:
        A LangChain BaseChatModel instance.
    """
    # 1. Groq (free tier — generous limits, fast)
    if os.getenv("GROQ_API_KEY"):
        try:
            from langchain_groq import ChatGroq

            model = model_name or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
            logger.info("Using Groq provider: %s", model)
            return ChatGroq(model=model, temperature=temperature)
        except ImportError:
            logger.warning("langchain-groq not installed. pip install langchain-groq")

    # 2. Google Gemini (free tier — 15 RPM, 1M TPD)
    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            model = model_name or os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            logger.info("Using Google Gemini provider: %s", model)
            return ChatGoogleGenerativeAI(
                model=model, temperature=temperature, google_api_key=api_key
            )
        except ImportError:
            logger.warning(
                "langchain-google-genai not installed. pip install langchain-google-genai"
            )

    # 3. OpenAI
    if os.getenv("OPENAI_API_KEY"):
        try:
            from langchain_openai import ChatOpenAI

            model = model_name or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            logger.info("Using OpenAI provider: %s", model)
            return ChatOpenAI(model=model, temperature=temperature)
        except ImportError:
            logger.warning("langchain-openai not installed. pip install langchain-openai")

    # 4. Ollama (local, free, no API key needed)
    try:
        from langchain_ollama import ChatOllama

        model = model_name or os.getenv("OLLAMA_MODEL", "llama3.2")
        logger.info("Using Ollama local provider: %s (ensure Ollama is running)", model)
        return ChatOllama(model=model, temperature=temperature)
    except ImportError:
        logger.warning("langchain-ollama not installed. pip install langchain-ollama")
    except Exception as exc:
        logger.warning("Ollama connection failed: %s. Is Ollama running?", exc)

    raise RuntimeError(
        "No LLM provider available. Set one of: "
        "GROQ_API_KEY, GEMINI_API_KEY/GOOGLE_API_KEY, OPENAI_API_KEY, "
        "or install langchain-ollama and run Ollama locally."
    )


def get_embeddings_model(
    model_name: str | None = None,
):
    """Create an embeddings model from the best available provider.

    Priority: OPENAI_API_KEY → sentence-transformers (local, free)

    Returns:
        An embeddings model instance.
    """
    # OpenAI embeddings (if key available)
    if os.getenv("OPENAI_API_KEY"):
        try:
            from langchain_openai import OpenAIEmbeddings

            model = model_name or "text-embedding-3-small"
            logger.info("Using OpenAI embeddings: %s", model)
            return OpenAIEmbeddings(model=model)
        except ImportError:
            pass

    # Local sentence-transformers (free, no API key)
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings

        model = model_name or "all-MiniLM-L6-v2"
        logger.info("Using local HuggingFace embeddings: %s", model)
        return HuggingFaceEmbeddings(model_name=model)
    except ImportError:
        logger.warning(
            "sentence-transformers not installed. "
            "pip install sentence-transformers for local embeddings."
        )

    raise RuntimeError(
        "No embeddings provider available. Set OPENAI_API_KEY or "
        "install sentence-transformers for local embeddings."
    )


def get_provider_info() -> dict[str, str]:
    """Return info about which provider is currently active."""
    if os.getenv("GROQ_API_KEY"):
        return {"provider": "groq", "model": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")}
    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        return {"provider": "gemini", "model": os.getenv("GEMINI_MODEL", "gemini-2.0-flash")}
    if os.getenv("OPENAI_API_KEY"):
        return {"provider": "openai", "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini")}
    return {"provider": "ollama", "model": os.getenv("OLLAMA_MODEL", "llama3.2")}
