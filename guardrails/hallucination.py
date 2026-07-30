"""HallucinationChecker — detects when LLM output diverges from retrieved context."""

import logging
import os
from typing import Any

import numpy as np

from guardrails.base import ValidationResult, Validator

logger = logging.getLogger(__name__)


class HallucinationChecker(Validator):
    """Checks whether an LLM output is grounded in the provided context.

    Two modes are supported:

    - **embedding**: Uses sentence-transformers to compute cosine similarity
      between the output and each retrieved document. If the best similarity
      exceeds the threshold, the output passes.

    - **llm_judge**: Sends the output and context to another LLM (OpenAI) and
      asks it to rate factual alignment on a 1-5 scale. If the rating meets
      or exceeds the threshold (mapped to 1-5), the output passes.

    Usage:
        checker = HallucinationChecker(mode="embedding", threshold=0.7)
        result = await checker.validate(
            output="The sky is green",
            context={"retrieved_documents": ["The sky is blue."]}
        )
    """

    # Threshold mapping for LLM-judge mode: 1.0 confidence → rating 5, 0.0 → rating 1
    RATING_TO_CONFIDENCE = {1: 0.0, 2: 0.25, 3: 0.5, 4: 0.75, 5: 1.0}

    def __init__(
        self,
        mode: str = "embedding",
        threshold: float = 0.7,
        model_name: str = "all-MiniLM-L6-v2",
        openai_api_key: str | None = None,
    ) -> None:
        """Initialize the hallucination checker.

        Args:
            mode: "embedding" or "llm_judge".
            threshold: Minimum score to pass (0.0-1.0 for embedding, 1-5 for llm_judge).
            model_name: sentence-transformers model name (embedding mode) or
                        OpenAI model name (llm_judge mode, e.g. "gpt-4o-mini").
            openai_api_key: OpenAI API key for llm_judge mode.
        """
        self._mode = mode
        self._threshold = threshold
        self._model_name = model_name
        self._openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")

        self._embedding_model = None
        if self._mode == "embedding":
            self._embedding_model = self._load_embedding_model()

    @property
    def name(self) -> str:
        return f"hallucination:{self._mode}"

    def _load_embedding_model(self):  # type: ignore[no-untyped-def]
        """Lazy-load the sentence-transformers model."""
        try:
            from sentence_transformers import SentenceTransformer

            logger.debug("Loading sentence-transformers model: %s", self._model_name)
            return SentenceTransformer(self._model_name)
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. Embedding mode will be unavailable."
            )
            return None
        except Exception as exc:
            logger.error("Failed to load embedding model: %s", exc)
            return None

    async def validate(
        self, output: str, context: dict[str, Any] | None = None
    ) -> ValidationResult:
        if self._mode == "embedding":
            return await self._validate_embedding(output, context or {})
        elif self._mode == "llm_judge":
            return await self._validate_llm_judge(output, context or {})
        else:
            logger.error("Unknown mode: %s", self._mode)
            return ValidationResult(
                passed=False,
                confidence=0.0,
                details={"error": f"Unknown mode: {self._mode}"},
                validator_name=self.name,
            )

    async def _validate_embedding(
        self, output: str, context: dict[str, Any]
    ) -> ValidationResult:
        documents = context.get("retrieved_documents", [])
        if not documents:
            logger.debug("No retrieved documents in context; skipping hallucination check.")
            return ValidationResult(
                passed=True,
                confidence=1.0,
                details={"warning": "No context documents provided; check skipped."},
                validator_name=self.name,
            )

        if self._embedding_model is None:
            return ValidationResult(
                passed=True,
                confidence=0.5,
                details={"error": "Embedding model not available."},
                validator_name=self.name,
            )

        try:
            output_emb = self._embedding_model.encode(
                [output], convert_to_numpy=True, normalize_embeddings=True
            )
            doc_embs = self._embedding_model.encode(
                documents, convert_to_numpy=True, normalize_embeddings=True
            )

            similarities = np.dot(doc_embs, output_emb.T).flatten()
            best_score = float(np.max(similarities)) if len(similarities) > 0 else 0.0
            best_idx = int(np.argmax(similarities)) if len(similarities) > 0 else -1

            passed = best_score >= self._threshold

            return ValidationResult(
                passed=passed,
                confidence=float(best_score),
                details={
                    "best_similarity": best_score,
                    "best_doc_index": best_idx,
                    "threshold": self._threshold,
                    "all_similarities": similarities.tolist(),
                },
                validator_name=self.name,
            )
        except Exception as exc:
            logger.error("Embedding validation failed: %s", exc)
            return ValidationResult(
                passed=False,
                confidence=0.0,
                details={"error": str(exc)},
                validator_name=self.name,
            )

    async def _validate_llm_judge(
        self, output: str, context: dict[str, Any]
    ) -> ValidationResult:
        documents = context.get("retrieved_documents", [])
        context_text = "\n\n".join(documents) if documents else "No context provided."

        try:
            rating = await self._call_llm_judge(output, context_text)
            confidence = HallucinationChecker.RATING_TO_CONFIDENCE.get(
                rating, float(rating) / 5.0
            )
            threshold_mapped = (
                self._threshold / 5.0 if self._threshold > 1.0 else self._threshold
            )
            passed = confidence >= threshold_mapped

            return ValidationResult(
                passed=passed,
                confidence=confidence,
                details={
                    "llm_rating": rating,
                    "threshold": self._threshold,
                    "mode": "llm_judge",
                },
                validator_name=self.name,
            )
        except Exception as exc:
            logger.error("LLM judge validation failed: %s", exc)
            return ValidationResult(
                passed=False,
                confidence=0.0,
                details={"error": str(exc)},
                validator_name=self.name,
            )

    async def _call_llm_judge(self, output: str, context_text: str) -> int:
        """Call OpenAI to rate factual alignment on a 1-5 scale."""
        if not self._openai_api_key:
            raise ValueError("OPENAI_API_KEY not set for LLM judge mode.")

        prompt = (
            "You are a factual accuracy judge. Compare the following AI-generated output "
            "against the provided context documents. Rate how well the output is grounded "
            "in the context on a scale of 1 to 5:\n"
            "1 = Completely hallucinated / contradicts context\n"
            "2 = Mostly incorrect with minor alignment\n"
            "3 = Partially correct, mixed accuracy\n"
            "4 = Mostly correct, minor deviations\n"
            "5 = Perfectly aligned with context\n\n"
            f"CONTEXT DOCUMENTS:\n{context_text}\n\n"
            f"OUTPUT TO EVALUATE:\n{output}\n\n"
            'Respond with ONLY a single integer (1-5). Do not add any other text.'
        )

        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(api_key=self._openai_api_key)
            response = await client.chat.completions.create(
                model=self._model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=5,
            )
            rating_str = response.choices[0].message.content.strip()
            rating = int(rating_str)
            return max(1, min(5, rating))
        except ImportError:
            raise ImportError("openai package required for LLM judge mode. pip install openai")
        except ValueError:
            logger.warning("LLM judge returned non-integer rating: %r", rating_str)
            return 3  # neutral fallback
