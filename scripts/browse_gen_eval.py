"""Streamlit viewer pre eval/runs/*/gen_results.jsonl.

Spustenie:
    streamlit run scripts/browse_gen_eval.py
"""
from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from src.eval.metrics import run_score
from src.schemas import EvalResult, JudgeRefusalScores, JudgeScores

_RUNS_DIR = Path("eval/runs")

st.set_page_config(page_title="BangRag Gen Eval", layout="wide")


def _get_run_dirs() -> list[Path]:
    if not _RUNS_DIR.exists():
        return []
    dirs = sorted(
        [d for d in _RUNS_DIR.iterdir() if d.is_dir() and d.name[0].isdigit()],
        reverse=True,
    )
    return [d for d in dirs if (d / "gen_results.jsonl").exists()]


@st.cache_data
def load_results(path: str) -> list[EvalResult]:
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(EvalResult.model_validate(json.loads(line)))
    return rows


@st.cache_data
def load_config(path: str) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def _score_color(score: int, max_score: int) -> str:
    ratio = score / max_score if max_score else 0
    if ratio >= 1.0:
        return "green"
    if ratio >= 0.5:
        return "orange"
    return "red"


def _score_md(label: str, score: int, max_score: int) -> str:
    color = _score_color(score, max_score)
    return f"**{label}:** :{color}[{score}/{max_score}]"


def _bool_md(label: str, value: int) -> str:
    icon = ":green[✓]" if value else ":red[✗]"
    return f"**{label}:** {icon}"


# ── Run selector ───────────────────────────────────────────────────────────────

run_dirs = _get_run_dirs()

with st.sidebar:
    st.title("BangRag Gen Eval")

    if not run_dirs:
        st.error("Žiadne runy v eval/runs/. Spusti `python -m src.eval.run_gen_eval` najprv.")
        st.stop()

    sel_run = st.selectbox("Run", run_dirs, format_func=lambda p: p.name)

    config_path = sel_run / "config.toml"
    if config_path.exists():
        cfg = load_config(str(config_path))
        with st.expander("Konfigurácia"):
            st.code(config_path.read_text(encoding="utf-8"), language="toml")

    st.divider()

# ── Load data ──────────────────────────────────────────────────────────────────

all_rows = load_results(str(sel_run / "gen_results.jsonl"))

# ── Sidebar metrics ────────────────────────────────────────────────────────────

with st.sidebar:
    all_cats = sorted({r.category for r in all_rows})
    sel_cats = st.multiselect("Kategória", all_cats, default=all_cats)

    min_correct = st.slider("Min. correctness", 0, 2, 0)

    st.divider()

    non_refusal = [r for r in all_rows if r.category != "refusal"]
    score = run_score(all_rows) * 100
    st.metric("RunScore", f"{score:.1f} / 100")
    if non_refusal:
        avg_faith = sum(
            r.judge_scores.faithfulness if isinstance(r.judge_scores, JudgeScores) else 0
            for r in non_refusal
        ) / len(non_refusal)
        avg_correct = sum(
            r.judge_scores.correctness if isinstance(r.judge_scores, JudgeScores) else 0
            for r in non_refusal
        ) / len(non_refusal)
        gold_pct = sum(1 for r in non_refusal if r.gold_retrieved) / len(non_refusal) * 100
        st.metric("Prípadov celkom", len(all_rows))
        st.metric("Avg faithfulness", f"{avg_faith:.2f} / 2")
        st.metric("Avg correctness", f"{avg_correct:.2f} / 2")
        st.metric("Gold@sel", f"{gold_pct:.0f}%")

# ── Filter rows ────────────────────────────────────────────────────────────────

rows = [
    r for r in all_rows
    if r.category in sel_cats
    and (r.judge_scores.correctness if isinstance(r.judge_scores, JudgeScores) else 0) >= min_correct
]

if not rows:
    st.warning("Žiadne výsledky pre aktuálne filtre.")
    st.stop()

# ── Selectbox ──────────────────────────────────────────────────────────────────

def _label(r: EvalResult) -> str:
    cat = r.category
    gold_icon = "" if cat == "refusal" else (" ✓" if r.gold_retrieved else " ✗")
    q = r.question[:80] + ("..." if len(r.question) > 80 else "")
    return f"[{r.id}] {q}  ({cat}{gold_icon})"


selected = st.selectbox("Prípad", rows, format_func=_label)

if selected is None:
    st.stop()

scores = selected.judge_scores
is_refusal = selected.category == "refusal"

st.divider()

# ── Sekcia 1: Odpovede ─────────────────────────────────────────────────────────

col_gold, col_gen = st.columns(2)
with col_gold:
    st.subheader("Referenčná odpoveď")
    st.write(selected.gold_answer or "_bez referenčnej odpovede_")
    if selected.gold_chunk_ids:
        st.caption("Gold chunks: " + ", ".join(f"`{i}`" for i in selected.gold_chunk_ids))

with col_gen:
    st.subheader("Vygenerovaná odpoveď")
    st.write(selected.generated_answer)

st.divider()

# ── Sekcia 2: Hodnotenie judge ─────────────────────────────────────────────────

st.subheader("Hodnotenie judge")

if isinstance(scores, JudgeRefusalScores):
    m1, m2 = st.columns(2)
    with m1:
        st.markdown(_bool_md("Refused correctly", scores.refused_correctly))
    with m2:
        st.markdown(_bool_md("Po slovensky", scores.in_slovak))
else:
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(_score_md("Faithfulness", scores.faithfulness, 2))
    with m2:
        st.markdown(_score_md("Correctness", scores.correctness, 2))
    with m3:
        st.markdown(_bool_md("Cites [Z1]", scores.cites_sources))
    with m4:
        st.markdown(_bool_md("Po slovensky", scores.in_slovak))

if scores.reasoning:
    st.info(scores.reasoning)

st.divider()

# ── Sekcia 3: Agent kroky ──────────────────────────────────────────────────────

st.subheader("Agent kroky")

for i, call in enumerate(selected.agent_tool_calls, 1):
    with st.expander(f"Krok {i} — search_rules(query={call.query!r}, variant={call.variant}, k={call.k})"):
        st.write(f"Nájdené ({call.n_new} nových z {len(call.retrieved_ids)}):")
        for rid in call.retrieved_ids:
            st.code(rid, language=None)

with st.expander(f"Záverečný select_chunks ({len(selected.agent_selected_chunk_ids)} chunkov)"):
    for sid in selected.agent_selected_chunk_ids:
        gold_mark = " ← gold" if sid in selected.gold_chunk_ids else ""
        st.code(sid + gold_mark, language=None)

parts = [
    f"**{selected.n_search_calls}** search volaní",
    f"**{selected.n_selected_chunks}** vybraných chunkov",
    f"**{selected.tool_token_count}** tokenov (tool loop)" if selected.tool_token_count else None,
    f"**{selected.gen_latency_ms} ms** celková latencia",
]
st.caption("  |  ".join(p for p in parts if p))
