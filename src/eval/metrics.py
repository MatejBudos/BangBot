"""Retrieval evaluation metrics."""
from __future__ import annotations


def recall_at_k(retrieved_ids: list[str], gold_ids: list[str], k: int) -> float:
    """1.0 if any gold ID appears in top-k retrieved, 0.0 otherwise."""
    if not gold_ids:
        return 0.0
    return 1.0 if any(g in retrieved_ids[:k] for g in gold_ids) else 0.0


def mrr(retrieved_ids: list[str], gold_ids: list[str]) -> float:
    """Reciprocal rank of first gold ID hit; 0.0 if none found."""
    if not gold_ids:
        return 0.0
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in gold_ids:
            return 1.0 / rank
    return 0.0
