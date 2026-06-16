"""Ablation runner: dense / sparse / hybrid retrieval eval.

Usage:
    python -m src.eval.run_eval --variant all
    python -m src.eval.run_eval --variant hybrid
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.eval.metrics import mrr, recall_at_k
from src.retrieval.lancedb_store import HybridRetriever

_QA_PATH = Path("eval/qa.jsonl")
_DB_PATH = "artifacts/.lance"
_RESULTS_PATH = Path("eval/results.md")
# Hybrid RRF scores are typically in [0, ~0.033]; below this threshold the
# retriever is considered to have found no relevant context (refusal signal).
_REFUSAL_THRESHOLD = 0.01
_KS = [1, 3, 5]


def _load_qa(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _eval_variant(retriever: HybridRetriever, rows: list[dict], variant: str) -> dict:
    non_refusal = [r for r in rows if not r.get("expected_refusal", False)]
    refusal_rows = [r for r in rows if r.get("expected_refusal", False)]

    recalls: dict[int, list[float]] = {k: [] for k in _KS}
    mrr_scores: list[float] = []

    for row in non_refusal:
        results = retriever.search(row.get("question", ""), k=max(_KS), variant=variant)
        retrieved = [r["id"] for r in results]
        gold = row["gold_chunk_ids"]
        for k in _KS:
            recalls[k].append(recall_at_k(retrieved, gold, k))
        mrr_scores.append(mrr(retrieved, gold))

    # Refusal accuracy only meaningful for hybrid (uses RRF score as confidence)
    refusal_acc: float | None = None
    if refusal_rows and variant == "hybrid":
        hits = 0
        for row in refusal_rows:
            results = retriever.search(row.get("query") or row.get("question", ""), k=1, variant="hybrid")
            score = results[0].get("_rrf_score", 1.0) if results else 0.0
            if score < _REFUSAL_THRESHOLD:
                hits += 1
        refusal_acc = hits / len(refusal_rows)

    def _avg(lst: list[float]) -> float:
        return sum(lst) / len(lst) if lst else 0.0

    return {
        "variant": variant,
        "R@1": _avg(recalls[1]),
        "R@3": _avg(recalls[3]),
        "R@5": _avg(recalls[5]),
        "MRR": _avg(mrr_scores),
        "refusal_acc": refusal_acc,
        "n_retrieval": len(non_refusal),
        "n_refusal": len(refusal_rows),
    }


def _format_table(results: list[dict]) -> str:
    header = "| Varianta | R@1  | R@3  | R@5  | MRR  | Refusal Acc |"
    separator = "|----------|------|------|------|------|-------------|"
    lines = [header, separator]
    for r in results:
        ref = f"{r['refusal_acc']:.2f}" if r["refusal_acc"] is not None else "—   "
        lines.append(
            f"| {r['variant']:<8} "
            f"| {r['R@1']:.2f} "
            f"| {r['R@3']:.2f} "
            f"| {r['R@5']:.2f} "
            f"| {r['MRR']:.2f} "
            f"| {ref}        |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="BangRag retrieval ablation eval")
    parser.add_argument(
        "--variant",
        choices=["dense", "sparse", "hybrid", "all"],
        default="all",
        help="Which retrieval variant(s) to evaluate",
    )
    parser.add_argument("--db", default=_DB_PATH, help="Path to LanceDB artifacts")
    parser.add_argument("--qa", default=str(_QA_PATH), help="Path to eval/qa.jsonl")
    args = parser.parse_args()

    qa_path = Path(args.qa)
    if not qa_path.exists():
        raise FileNotFoundError(f"Eval set not found: {qa_path}. Run manual curation first.")

    rows = _load_qa(qa_path)
    print(f"Loaded {len(rows)} eval rows ({sum(1 for r in rows if not r.get('expected_refusal'))} retrieval, "
          f"{sum(1 for r in rows if r.get('expected_refusal'))} refusal)")

    retriever = HybridRetriever(args.db)

    variants = ["dense", "sparse", "hybrid"] if args.variant == "all" else [args.variant]
    eval_results = []
    for v in variants:
        n_ret = sum(1 for r in rows if not r.get("expected_refusal"))
        print(f"  {v}: evaluating {n_ret} queries...", end=" ", flush=True)
        metrics = _eval_variant(retriever, rows, v)
        eval_results.append(metrics)
        print(f"R@5={metrics['R@5']:.2f} MRR={metrics['MRR']:.2f}")

    table = _format_table(eval_results)
    print("\n" + table + "\n")

    _RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    n_ret = eval_results[0]["n_retrieval"]
    n_ref = eval_results[0]["n_refusal"]
    with open(_RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write("# Ablation results\n\n")
        f.write(table + "\n\n")
        f.write(f"_Eval set: {n_ret} retrieval queries + {n_ref} refusal cases_\n")
    print(f"Saved to {_RESULTS_PATH}")


if __name__ == "__main__":
    main()
