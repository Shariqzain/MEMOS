"""
MEMOS — Embedding Service
Generates dense vector embeddings for semantic memory retrieval.
Supports sentence-transformers and Ollama embedding backends.
"""
from __future__ import annotations

import logging
from typing import List, Optional

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_model = None
_model_name: Optional[str] = None


def _load_sentence_transformer_model():
    global _model, _model_name
    if _model is not None:
        return _model
    try:
        from sentence_transformers import SentenceTransformer
        model_name = settings.embedding_model
        logger.info("Loading embedding model: %s", model_name)
        _model = SentenceTransformer(model_name)
        _model_name = model_name
        logger.info("Embedding model loaded. Dimension: %d", _model.get_sentence_embedding_dimension())
        return _model
    except Exception as e:
        logger.error("Failed to load SentenceTransformer: %s", e)
        raise


def embed_text(text: str) -> List[float]:
    """Generate embedding for a single text string."""
    if not text or not text.strip():
        raise ValueError("Cannot embed empty text.")

    provider = settings.embedding_provider

    if provider == "sentence_transformers":
        model = _load_sentence_transformer_model()
        embedding = model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    elif provider == "ollama":
        import httpx
        resp = httpx.post(
            f"{settings.llm_base_url}/api/embeddings",
            json={"model": settings.llm_model, "prompt": text},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["embedding"]

    else:
        raise ValueError(f"Unknown embedding provider: {provider}")


def embed_batch(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for multiple texts efficiently."""
    if not texts:
        return []

    provider = settings.embedding_provider

    if provider == "sentence_transformers":
        model = _load_sentence_transformer_model()
        embeddings = model.encode(texts, normalize_embeddings=True, batch_size=32)
        return [e.tolist() for e in embeddings]

    else:
        # Fall back to individual embedding for non-batch providers
        return [embed_text(t) for t in texts]


def get_embedding_dimension() -> int:
    """Return dimension of embedding vectors."""
    if settings.embedding_provider == "sentence_transformers":
        model = _load_sentence_transformer_model()
        return model.get_sentence_embedding_dimension()
    return 384  # Default for MiniLM
