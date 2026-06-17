"""LanceDB-backed hybrid retriever for Bang! chunks."""
from __future__ import annotations

import lancedb
import numpy as np
from lancedb.rerankers import RRFReranker

from src.embedding.embedder import Embedder
from src.retrieval.slovak_text import preprocess_for_fts

_TABLE_NAME = "bang_chunks"


def _normalize_chunk(chunk: dict, text_lemmatized: str, vector: list[float]) -> dict:
    return {
        "id": chunk["id"],
        "type": chunk.get("type", ""),
        "name_sk": chunk.get("name_sk", ""),
        "sk_name": chunk.get("sk_name", ""),
        "alt_names": chunk.get("alt_names", []),
        "name_orig": chunk.get("name_orig", ""),
        "category": chunk.get("category", ""),
        "expansion": chunk.get("expansion", ""),
        "image_path": chunk.get("image_path", ""),
        "section_title": chunk.get("section_title", ""),
        "source": chunk.get("source", ""),
        "text": chunk["text"],
        "text_lemmatized": text_lemmatized,
        "vector": vector,
    }


def build_table(chunks: list[dict], embeddings: np.ndarray, db_path: str) -> None:
    """Build (or overwrite) the LanceDB table with FTS index from chunks + embeddings."""
    db = lancedb.connect(db_path)

    records = []
    for chunk, embedding in zip(chunks, embeddings):
        if chunk.get("type") == "card":
            # Include sk_name and alt_names in the repeated name block so BM25 boosts
            # all known names for this card (original, Slovak, alternate language variants)
            name_part = " ".join(filter(None, [
                chunk.get("name_sk"),
                chunk.get("sk_name"),
                chunk.get("name_orig"),
                *chunk.get("alt_names", []),
            ]))
            fts_source = " ".join([name_part, name_part, name_part, chunk["text"]])
        else:
            name_part = " ".join(filter(None, [
                chunk.get("name_sk"),
                chunk.get("name_orig"),
                chunk.get("section_title"),
            ]))
            fts_source = " ".join(filter(None, [name_part, chunk["text"]]))
        text_lemmatized = preprocess_for_fts(fts_source)
        records.append(_normalize_chunk(chunk, text_lemmatized, embedding.tolist()))

    table = db.create_table(_TABLE_NAME, data=records, mode="overwrite")
    table.create_fts_index("text_lemmatized", replace=True)


def _exact_name_first(results: list[dict], query: str) -> list[dict]:
    """Promote chunks whose card name matches the full query or any query token."""
    q_full = query.lower().strip()
    # Individual tokens catch "čo robí Pivo" → token "pivo" matches card "Pivo"
    q_tokens = {t.lower() for t in query.split() if len(t) > 1}
    top, rest = [], []
    for r in results:
        names = {
            (r.get("name_sk") or "").lower(),
            (r.get("sk_name") or "").lower(),
            (r.get("name_orig") or "").lower(),
            *(n.lower() for n in (r.get("alt_names") or [])),
        }
        # Full query matches multi-word names ("Posledné pivo");
        # token match catches single-word names inside a sentence ("Pivo" in "čo robí Pivo")
        if q_full in names or bool(names & q_tokens):
            top.append(r)
        else:
            rest.append(r)
    return top + rest


class HybridRetriever:
    """Hybrid dense+sparse retriever using LanceDB native RRF fusion."""

    def __init__(self, db_path: str) -> None:
        self._db = lancedb.connect(db_path)
        self._table = self._db.open_table(_TABLE_NAME)
        self._embedder = Embedder()
        self._reranker = RRFReranker(K=60, return_score="all")

    def search(self, query: str, k: int = 5, variant: str = "sparse") -> list[dict]:
        """Return top-k chunks. variant: 'dense' | 'sparse' | 'hybrid'."""
        query_vector = self._embedder.embed_query(query)
        query_lemmatized = preprocess_for_fts(query)

        if variant == "dense":
            raw = self._table.search(query_vector).limit(k).to_list()
            results = []
            for r in raw:
                d = dict(r)
                d["_dense_score"] = d.pop("_distance", 0.0)
                d["_rrf_score"] = 0.0
                d["_sparse_score"] = None
                results.append(d)
            return results

        if variant == "sparse":
            if not query_lemmatized:
                return []
            try:
                raw = self._table.search(query_lemmatized, query_type="fts").limit(k).to_list()
            except Exception:
                return []
            results = []
            for r in raw:
                d = dict(r)
                d["_sparse_score"] = d.pop("_score", 0.0)
                d["_rrf_score"] = 0.0
                d["_dense_score"] = 0.0
                results.append(d)
            return results

        # hybrid (default)
        if query_lemmatized:
            raw = (
                self._table.search(query_type="hybrid")
                .vector(query_vector)
                .text(query_lemmatized)
                .limit(max(k * 3, 15))
                .rerank(self._reranker)
                .to_list()
            )
        else:
            raw = self._table.search(query_vector).limit(max(k * 3, 15)).to_list()

        results = []
        for r in raw:
            d = dict(r)
            d["_rrf_score"] = d.pop("_relevance_score", 0.0)
            d["_dense_score"] = d.pop("_distance", 0.0)
            d["_sparse_score"] = d.pop("_score", None)
            results.append(d)
        return _exact_name_first(results, query)[:k]
