"""Embedder wrapper around intfloat/multilingual-e5-base."""
from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

_MODEL_NAME = "intfloat/multilingual-e5-base"


class Embedder:
    """Lazy-loading wrapper for multilingual-e5-base."""

    def __init__(self) -> None:
        self._model: SentenceTransformer | None = None

    def _load(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(_MODEL_NAME)
        return self._model

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        """Embed a list of passages with 'passage: ' prefix. Returns (N, 768) float32 L2-normalized."""
        prefixed = [f"passage: {t}" for t in texts]
        model = self._load()
        embeddings = model.encode(
            prefixed,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query with 'query: ' prefix. Returns (768,) float32 L2-normalized."""
        model = self._load()
        embedding = model.encode(
            f"query: {text}",
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embedding.astype(np.float32)
