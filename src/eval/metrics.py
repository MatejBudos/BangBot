"""Retrieval evaluation metrics."""
from __future__ import annotations

from src.schemas import JudgeRefusalScores, JudgeScores


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


def case_score(scores: JudgeScores | JudgeRefusalScores) -> float:
    """Normalized [0,1] quality score for one eval case."""
    if isinstance(scores, JudgeScores):
        return (scores.faithfulness / 2 + scores.correctness / 2 + scores.cites_sources + scores.in_slovak) / 4
    return (scores.refused_correctly + scores.in_slovak) / 2


def run_score(results: list) -> float:
    """Mean case_score across all results, [0,1]."""
    if not results:
        return 0.0
    return sum(case_score(r.judge_scores) for r in results) / len(results)
