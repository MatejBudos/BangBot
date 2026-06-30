"""Agentic generation eval with LLM-as-judge.

Usage:
    python -m src.eval.run_gen_eval
    python -m src.eval.run_gen_eval --input eval/gen_qa.jsonl --sample 5
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from openai import OpenAI

from src.eval.metrics import run_score
from src.generation.agent import (
    _AGENT_SYSTEM_PROMPT,
    _MAX_ITERATIONS,
    _MAX_K_PER_CALL,
    _MAX_TOTAL_CHUNKS,
    _MODEL_ID as _AGENT_MODEL_ID,
)
from src.generation.openai_client import LLMUnavailable, OpenAIClient
from src.generation.openai_client import _MODEL_ID as _GEN_MODEL_ID
from src.generation.prompts import SYSTEM_PROMPT, format_context
from src.retrieval.lancedb_store import HybridRetriever
from src.schemas import (
    EvalResult,
    GenQARow,
    JudgeRefusalScores,
    JudgeScores,
    RunConfig,
    ToolCallLog,
)

from src.generation.agent import BangAgent

_GEN_QA_PATH = Path("eval/gen_qa.jsonl")
_DB_PATH = "artifacts/.lance"
_RUNS_DIR = Path("eval/runs")

_JUDGE_MODEL = "gpt-4o-mini"

_PROMPTS_DIR = Path(__file__).parent.parent.parent / "config" / "prompts"
_JUDGE_SYSTEM: str = (_PROMPTS_DIR / "judge.md").read_text(encoding="utf-8").strip()
_JUDGE_SYSTEM_REFUSAL: str = (_PROMPTS_DIR / "judge_refusal.md").read_text(encoding="utf-8").strip()


def _md5(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:8]


def _render_toml(cfg: RunConfig) -> str:
    lines = [
        f'timestamp = "{cfg.timestamp}"',
        f'agent_model = "{cfg.agent_model}"',
        f'gen_model = "{cfg.gen_model}"',
        f'judge_model = "{cfg.judge_model}"',
        f'agent_prompt_md5 = "{cfg.agent_prompt_md5}"',
        f'gen_prompt_md5 = "{cfg.gen_prompt_md5}"',
        f'judge_prompt_md5 = "{cfg.judge_prompt_md5}"',
        "",
        "[agent]",
        f"max_iterations = {cfg.agent_max_iterations}",
        f"max_k_per_call = {cfg.agent_max_k_per_call}",
        f"max_total_chunks = {cfg.agent_max_total_chunks}",
    ]
    return "\n".join(lines) + "\n"


def _append_index(run_dir: Path, cfg: RunConfig, score: float, results: list[EvalResult]) -> None:
    index_path = _RUNS_DIR / "_index.toml"
    non_refusal = [r for r in results if r.category != "refusal"]

    faith_avg = _avg([
        r.judge_scores.faithfulness if isinstance(r.judge_scores, JudgeScores) else 0
        for r in non_refusal
    ])
    correct_avg = _avg([
        r.judge_scores.correctness if isinstance(r.judge_scores, JudgeScores) else 0
        for r in non_refusal
    ])
    cites_pct = _avg([
        r.judge_scores.cites_sources if isinstance(r.judge_scores, JudgeScores) else 0
        for r in non_refusal
    ]) * 100
    gold_pct = _avg([1.0 if r.gold_retrieved else 0.0 for r in non_refusal]) * 100

    entry = "\n".join([
        "[[runs]]",
        f'timestamp = "{cfg.timestamp}"',
        f'dir = "{run_dir.name}"',
        f"run_score = {score:.1f}",
        f"n_cases = {len(results)}",
        f'agent_model = "{cfg.agent_model}"',
        f'gen_model = "{cfg.gen_model}"',
        f'agent_prompt_md5 = "{cfg.agent_prompt_md5}"',
        f'gen_prompt_md5 = "{cfg.gen_prompt_md5}"',
        f"faithfulness_avg = {faith_avg:.2f}",
        f"correctness_avg = {correct_avg:.2f}",
        f"cites_sources_pct = {cites_pct:.1f}",
        f"gold_retrieved_pct = {gold_pct:.1f}",
    ]) + "\n"

    need_separator = index_path.exists() and index_path.stat().st_size > 0
    with open(index_path, "a", encoding="utf-8") as f:
        if need_separator:
            f.write("\n")
        f.write(entry)


def _load_qa(path: Path) -> list[GenQARow]:
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    rows.append(GenQARow.model_validate(json.loads(line)))
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
) -> JudgeScores | JudgeRefusalScores:
    if is_refusal:
        system = _JUDGE_SYSTEM_REFUSAL
        user = f"Otázka: {question}\n\nVygenerovaná odpoveď: {generated}"
        schema = JudgeRefusalScores
    else:
        system = _JUDGE_SYSTEM
        user = (
            f"Otázka: {question}\n\n"
            f"Dostupné chunky (agent vybral):\n{format_context(chunks)}\n\n"
            f"Referenčná odpoveď: {gold_answer}\n\n"
            f"Vygenerovaná odpoveď: {generated}"
        )
        schema = JudgeScores
    response = client.beta.chat.completions.parse(
        model=_JUDGE_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format=schema,
        temperature=0,
        seed=42,
    )
    parsed = response.choices[0].message.parsed
    if parsed is None:
        raise ValueError("Structured output parsing failed (model refused or content filtered)")
    return parsed


def _eval_case(
    row: GenQARow,
    agent: BangAgent,
    gen_client: OpenAIClient,
    judge_client: OpenAI,
) -> EvalResult | None:
    question = row.question
    gold_chunk_ids = row.gold_chunk_ids
    gold_answer = row.gold_answer
    is_refusal = row.expected_refusal

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

    return EvalResult(
        id=row.id,
        category=row.category,
        question=question,
        gold_answer=gold_answer,
        gold_chunk_ids=gold_chunk_ids,
        agent_tool_calls=agent.tool_calls_log,
        agent_selected_chunk_ids=selected_ids,
        gold_retrieved=gold_retrieved,
        generated_answer=generated,
        judge_scores=judge_scores,
        n_search_calls=len(agent.tool_calls_log),
        n_selected_chunks=len(selected_ids),
        tool_token_count=agent.last_tool_token_count,
        gen_latency_ms=gen_latency_ms,
    )


def _avg(lst: list[float]) -> float:
    return sum(lst) / len(lst) if lst else 0.0


def _format_table(by_category: dict[str, list[EvalResult]], all_results: list[EvalResult]) -> str:
    header = "| Kategória   | N  | Gold@sel | Faith. | Correct. | Cites | Slovak | Avg searches |"
    sep    = "|-------------|----|---------:|-------:|---------:|------:|-------:|-------------:|"
    lines = [header, sep]

    def _row(cat: str, results: list[EvalResult]) -> str:
        n = len(results)
        is_refusal_cat = cat == "refusal"

        slovak = _avg([r.judge_scores.in_slovak for r in results])
        avg_searches = _avg([r.n_search_calls for r in results])

        if is_refusal_cat:
            refused = _avg([
                r.judge_scores.refused_correctly
                if isinstance(r.judge_scores, JudgeRefusalScores) else 0
                for r in results
            ])
            return (
                f"| {cat:<11} | {n:<2} | —        | —      | —        | —     "
                f"| {slovak*100:>5.0f}% | refused: {refused*100:.0f}%  |"
            )

        gold_pct = _avg([1.0 if r.gold_retrieved else 0.0 for r in results]) * 100
        faith = _avg([
            r.judge_scores.faithfulness if isinstance(r.judge_scores, JudgeScores) else 0
            for r in results
        ])
        correct = _avg([
            r.judge_scores.correctness if isinstance(r.judge_scores, JudgeScores) else 0
            for r in results
        ])
        cites = _avg([
            r.judge_scores.cites_sources if isinstance(r.judge_scores, JudgeScores) else 0
            for r in results
        ]) * 100
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

    non_refusal = [r for r in all_results if r.category != "refusal"]
    if non_refusal:
        n = len(all_results)
        gold_pct = _avg([1.0 if r.gold_retrieved else 0.0 for r in non_refusal]) * 100
        faith = _avg([
            r.judge_scores.faithfulness if isinstance(r.judge_scores, JudgeScores) else 0
            for r in non_refusal
        ])
        correct = _avg([
            r.judge_scores.correctness if isinstance(r.judge_scores, JudgeScores) else 0
            for r in non_refusal
        ])
        cites = _avg([
            r.judge_scores.cites_sources if isinstance(r.judge_scores, JudgeScores) else 0
            for r in non_refusal
        ]) * 100
        slovak = _avg([r.judge_scores.in_slovak for r in all_results]) * 100
        avg_searches = _avg([r.n_search_calls for r in all_results])
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

    n_refusal = sum(1 for r in rows if r.expected_refusal)
    print(f"Loaded {len(rows)} gen eval rows ({len(rows) - n_refusal} generation, {n_refusal} refusal)")

    # Build run directory and config snapshot
    ts = datetime.now()
    run_dir = _RUNS_DIR / f"{ts.strftime('%Y%m%d_%H%M%S')}_{_AGENT_MODEL_ID}"
    run_dir.mkdir(parents=True, exist_ok=True)

    run_config = RunConfig(
        timestamp=ts.isoformat(timespec="seconds"),
        agent_model=_AGENT_MODEL_ID,
        gen_model=_GEN_MODEL_ID,
        judge_model=_JUDGE_MODEL,
        agent_prompt_md5=_md5(_AGENT_SYSTEM_PROMPT),
        gen_prompt_md5=_md5(SYSTEM_PROMPT),
        judge_prompt_md5=_md5(_JUDGE_SYSTEM),
        agent_max_iterations=_MAX_ITERATIONS,
        agent_max_k_per_call=_MAX_K_PER_CALL,
        agent_max_total_chunks=_MAX_TOTAL_CHUNKS,
    )
    (run_dir / "config.toml").write_text(_render_toml(run_config), encoding="utf-8")
    print(f"Run dir: {run_dir}")

    retriever = HybridRetriever(args.db)
    agent = BangAgent(retriever)
    gen_client = OpenAIClient()
    judge_client = OpenAI(api_key=os.environ.get("OPENAI_KEY"))

    results: list[EvalResult] = []
    results_jsonl = run_dir / "gen_results.jsonl"
    results_md = run_dir / "gen_results.md"

    with open(results_jsonl, "w", encoding="utf-8") as out_f:
        for i, row in enumerate(rows, 1):
            print(f"[{i}/{len(rows)}] {row.id} ({row.category})...", flush=True)
            result = _eval_case(row, agent, gen_client, judge_client)
            if result is None:
                continue
            scores = result.judge_scores
            if row.expected_refusal:
                print(
                    f"    refused_correctly={scores.refused_correctly if isinstance(scores, JudgeRefusalScores) else '?'} "
                    f"in_slovak={scores.in_slovak}"
                )
            else:
                print(
                    f"    gold={result.gold_retrieved} "
                    f"faith={scores.faithfulness if isinstance(scores, JudgeScores) else '?'}/2 "
                    f"correct={scores.correctness if isinstance(scores, JudgeScores) else '?'}/2 "
                    f"cites={scores.cites_sources if isinstance(scores, JudgeScores) else '?'} "
                    f"searches={result.n_search_calls}"
                )
            results.append(result)
            out_f.write(result.model_dump_json() + "\n")
            out_f.flush()

    if not results:
        print("No results to aggregate.")
        return

    by_category: dict[str, list[EvalResult]] = defaultdict(list)
    for r in results:
        by_category[r.category].append(r)

    score = run_score(results) * 100
    print(f"\nRunScore: {score:.1f} / 100")

    table = _format_table(by_category, results)
    print("\n" + table + "\n")

    with open(results_md, "w", encoding="utf-8") as f:
        f.write("# Generation eval results (LLM-as-judge)\n\n")
        f.write(f"**RunScore: {score:.1f} / 100**\n\n")
        f.write(table + "\n\n")
        f.write(
            f"_Eval set: {len(results)} prípadov. "
            f"Agent: {run_config.agent_model} | Gen: {run_config.gen_model} | "
            f"Judge: {run_config.judge_model} (same-model bias — interpretuj konzervatívne)._\n"
        )

    _append_index(run_dir, run_config, score, results)
    print(f"Saved {results_jsonl} + {results_md}")
    print(f"Index updated: {_RUNS_DIR / '_index.toml'}")


if __name__ == "__main__":
    main()
