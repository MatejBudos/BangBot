"""Agentic generation eval with LLM-as-judge.

Usage:
    python -m src.eval.run_gen_eval
    python -m src.eval.run_gen_eval --input eval/gen_qa.jsonl --sample 5
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

from openai import OpenAI

from src.generation.agent import BangAgent
from src.generation.openai_client import LLMUnavailable, OpenAIClient
from src.generation.prompts import format_context
from src.retrieval.lancedb_store import HybridRetriever

_GEN_QA_PATH = Path("eval/gen_qa.jsonl")
_DB_PATH = "artifacts/.lance"
_RESULTS_JSONL = Path("eval/gen_results.jsonl")
_RESULTS_MD = Path("eval/gen_results.md")

_JUDGE_SYSTEM = (
    "Si nestranný sudca kvality odpovedí asistenta pre Bang! pravidlá. "
    "Hodnoť VÝLUČNE na základe dodaných chunkov — nie vlastných znalostí. "
    "Vráť JSON s poľami: faithfulness (int 0-2), correctness (int 0-2), "
    "cites_sources (int 0-1), in_slovak (int 0-1), reasoning (string)."
)

_JUDGE_SYSTEM_REFUSAL = (
    "Si nestranný sudca kvality odpovedí asistenta pre Bang! pravidlá. "
    "Otázka je MIMO Bang! pravidlá — správna odpoveď je odmietnutie. "
    "Vráť JSON s poľami: refused_correctly (int 0-1; 1 ak odpoveď odmietla bez "
    "potvrdenia pravidiel), in_slovak (int 0-1), reasoning (string)."
)


def _load_qa(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Parse error on line {i}: {exc}\n  repr: {repr(line[:120])}") from exc
    return rows


def _judge(
    client: OpenAI,
    question: str,
    chunks: list[dict],
    generated: str,
    gold_answer: str,
    is_refusal: bool,
) -> dict:
    if is_refusal:
        system = _JUDGE_SYSTEM_REFUSAL
        user = f"Otázka: {question}\n\nVygenerovaná odpoveď: {generated}"
    else:
        system = _JUDGE_SYSTEM
        user = (
            f"Otázka: {question}\n\n"
            f"Dostupné chunky (agent vybral):\n{format_context(chunks)}\n\n"
            f"Referenčná odpoveď: {gold_answer}\n\n"
            f"Vygenerovaná odpoveď: {generated}"
        )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
        temperature=0,
        seed=42,
    )
    return json.loads(response.choices[0].message.content)


def _eval_case(
    row: dict,
    agent: BangAgent,
    gen_client: OpenAIClient,
    judge_client: OpenAI,
) -> dict | None:
    question = row["question"]
    gold_chunk_ids: list[str] = row.get("gold_chunk_ids", [])
    gold_answer: str = row.get("gold_answer", "")
    is_refusal: bool = row.get("expected_refusal", False)

    t0 = time.monotonic()
    try:
        selected_chunks = agent.collect_context(question)
    except LLMUnavailable as exc:
        print(f"    SKIP (agent unavailable): {exc}")
        return None

    selected_ids = [c.get("id", "") for c in selected_chunks]
    # gold_retrieved is None for refusal cases (no gold chunks expected)
    gold_retrieved: bool | None = None
    if not is_refusal:
        gold_retrieved = bool(gold_chunk_ids) and any(gid in selected_ids for gid in gold_chunk_ids)

    try:
        generated = "".join(gen_client.stream_answer(question, selected_chunks))
    except LLMUnavailable as exc:
        print(f"    SKIP (gen unavailable): {exc}")
        return None

    gen_latency_ms = round((time.monotonic() - t0) * 1000)

    try:
        judge_scores = _judge(judge_client, question, selected_chunks, generated, gold_answer, is_refusal)
    except Exception as exc:
        print(f"    SKIP (judge error): {exc}")
        return None

    return {
        "id": row["id"],
        "category": row.get("category", "unknown"),
        "question": question,
        "agent_tool_calls": agent.tool_calls_log,
        "agent_selected_chunk_ids": selected_ids,
        "gold_retrieved": gold_retrieved,
        "generated_answer": generated,
        "judge_scores": judge_scores,
        "n_search_calls": len(agent.tool_calls_log),
        "n_selected_chunks": len(selected_ids),
        "tool_token_count": agent.last_tool_token_count,
        "gen_latency_ms": gen_latency_ms,
    }


def _avg(lst: list[float]) -> float:
    return sum(lst) / len(lst) if lst else 0.0


def _format_table(by_category: dict[str, list[dict]], all_results: list[dict]) -> str:
    header = "| Kategória   | N  | Gold@sel | Faith. | Correct. | Cites | Slovak | Avg searches |"
    sep    = "|-------------|----|---------:|-------:|---------:|------:|-------:|-------------:|"
    lines = [header, sep]

    def _row(cat: str, results: list[dict]) -> str:
        n = len(results)
        is_refusal_cat = cat == "refusal"

        slovak = _avg([r["judge_scores"].get("in_slovak", 0) for r in results])
        avg_searches = _avg([r["n_search_calls"] for r in results])

        if is_refusal_cat:
            refused = _avg([r["judge_scores"].get("refused_correctly", 0) for r in results])
            return (
                f"| {cat:<11} | {n:<2} | —        | —      | —        | —     "
                f"| {slovak*100:>5.0f}% | refused: {refused*100:.0f}%  |"
            )

        gold_pct = _avg([1.0 if r["gold_retrieved"] else 0.0 for r in results]) * 100
        faith = _avg([r["judge_scores"].get("faithfulness", 0) for r in results])
        correct = _avg([r["judge_scores"].get("correctness", 0) for r in results])
        cites = _avg([r["judge_scores"].get("cites_sources", 0) for r in results]) * 100
        return (
            f"| {cat:<11} | {n:<2} | {gold_pct:>7.0f}% "
            f"| {faith:.1f}/2  "
            f"| {correct:.1f}/2     "
            f"| {cites:>4.0f}% "
            f"| {slovak*100:>5.0f}% "
            f"| {avg_searches:.1f}          |"
        )

    for cat in sorted(by_category):
        lines.append(_row(cat, by_category[cat]))

    non_refusal = [r for r in all_results if r.get("category") != "refusal"]
    if non_refusal:
        n = len(all_results)
        gold_pct = _avg([1.0 if r["gold_retrieved"] else 0.0 for r in non_refusal]) * 100
        faith = _avg([r["judge_scores"].get("faithfulness", 0) for r in non_refusal])
        correct = _avg([r["judge_scores"].get("correctness", 0) for r in non_refusal])
        cites = _avg([r["judge_scores"].get("cites_sources", 0) for r in non_refusal]) * 100
        slovak = _avg([r["judge_scores"].get("in_slovak", 0) for r in all_results]) * 100
        avg_searches = _avg([r["n_search_calls"] for r in all_results])
        lines.append(
            f"| **TOTAL**   | {n:<2} | {gold_pct:>7.0f}% "
            f"| {faith:.1f}/2  "
            f"| {correct:.1f}/2     "
            f"| {cites:>4.0f}% "
            f"| {slovak:>5.0f}% "
            f"| {avg_searches:.1f}          |"
        )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="BangRag agentic generation eval")
    parser.add_argument("--input", default=str(_GEN_QA_PATH), help="Path to gen_qa.jsonl")
    parser.add_argument("--db", default=_DB_PATH, help="Path to LanceDB artifacts")
    parser.add_argument("--sample", type=int, default=None, help="Evaluate N random cases")
    args = parser.parse_args()

    qa_path = Path(args.input)
    if not qa_path.exists():
        raise FileNotFoundError(
            f"Gen eval set not found: {qa_path}. Construct eval/gen_qa.jsonl first."
        )

    rows = _load_qa(qa_path)
    if args.sample:
        random.seed(42)
        rows = random.sample(rows, min(args.sample, len(rows)))

    n_refusal = sum(1 for r in rows if r.get("expected_refusal"))
    print(f"Loaded {len(rows)} gen eval rows ({len(rows) - n_refusal} generation, {n_refusal} refusal)")

    retriever = HybridRetriever(args.db)
    agent = BangAgent(retriever)
    gen_client = OpenAIClient()
    judge_client = OpenAI(api_key=os.environ.get("OPENAI_KEY"))

    results: list[dict] = []
    _RESULTS_JSONL.parent.mkdir(parents=True, exist_ok=True)

    with open(_RESULTS_JSONL, "w", encoding="utf-8") as out_f:
        for i, row in enumerate(rows, 1):
            print(f"[{i}/{len(rows)}] {row['id']} ({row.get('category', '?')})...", flush=True)
            result = _eval_case(row, agent, gen_client, judge_client)
            if result is None:
                continue
            scores = result["judge_scores"]
            if row.get("expected_refusal"):
                print(
                    f"    refused_correctly={scores.get('refused_correctly')} "
                    f"in_slovak={scores.get('in_slovak')}"
                )
            else:
                print(
                    f"    gold={result['gold_retrieved']} "
                    f"faith={scores.get('faithfulness')}/2 "
                    f"correct={scores.get('correctness')}/2 "
                    f"cites={scores.get('cites_sources')} "
                    f"searches={result['n_search_calls']}"
                )
            results.append(result)
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_f.flush()

    if not results:
        print("No results to aggregate.")
        return

    by_category: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_category[r["category"]].append(r)

    table = _format_table(by_category, results)
    print("\n" + table + "\n")

    _RESULTS_MD.parent.mkdir(parents=True, exist_ok=True)
    with open(_RESULTS_MD, "w", encoding="utf-8") as f:
        f.write("# Generation eval results (LLM-as-judge)\n\n")
        f.write(table + "\n\n")
        f.write(
            f"_Eval set: {len(results)} prípadov. "
            "Judge: gpt-4o-mini (same-model bias — interpretuj konzervatívne)._\n"
        )
    print(f"Saved {_RESULTS_JSONL} + {_RESULTS_MD}")


if __name__ == "__main__":
    main()
