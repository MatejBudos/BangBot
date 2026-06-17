"""Tests for src/eval/run_gen_eval.py — _load_qa, _judge prompt construction, _eval_case output, _format_table."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.eval.run_gen_eval import (
    _JUDGE_SYSTEM,
    _JUDGE_SYSTEM_REFUSAL,
    _avg,
    _eval_case,
    _format_table,
    _judge,
    _load_qa,
)
from src.generation.openai_client import LLMUnavailable
from src.schemas import (
    EvalResult,
    GenQARow,
    JudgeRefusalScores,
    JudgeScores,
    ToolCallLog,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_judge_client(scores: JudgeScores | JudgeRefusalScores) -> MagicMock:
    client = MagicMock()
    msg = MagicMock()
    msg.parsed = scores
    client.beta.chat.completions.parse.return_value.choices = [MagicMock(message=msg)]
    return client


def _make_agent(chunks: list[dict], tool_calls: list[ToolCallLog] | None = None) -> MagicMock:
    agent = MagicMock()
    agent.collect_context.return_value = chunks
    agent.tool_calls_log = tool_calls or []
    agent.last_tool_token_count = 300
    return agent


def _make_gen_client(answer: str = "Testovacia odpoveď [Z1].") -> MagicMock:
    client = MagicMock()
    client.stream_answer.return_value = iter([answer])
    return client


_CHUNK = {"id": "card_hnede_pivo", "type": "card", "name_sk": "Pivo", "sk_name": "Pivo",
          "alt_names": [], "name_orig": "pivo", "category": "hneda", "expansion": "base",
          "image_path": "", "section_title": "", "source": "", "text": "Doplní 1 život."}

_NORMAL_ROW = GenQARow(
    id="gq001",
    question="Čo robí karta Pivo?",
    gold_answer="Zahraním karty Pivo si hráč doplní 1 život.",
    gold_chunk_ids=["card_hnede_pivo"],
    category="lookup",
    expected_refusal=False,
)

_REFUSAL_ROW = GenQARow(
    id="gq099",
    question="Kto vyhral majstrovstvá sveta vo futbale 2022?",
    gold_answer="",
    gold_chunk_ids=[],
    category="refusal",
    expected_refusal=True,
)

_NORMAL_SCORES = JudgeScores(faithfulness=2, correctness=2, cites_sources=1, in_slovak=1, reasoning="ok")
_REFUSAL_SCORES = JudgeRefusalScores(refused_correctly=1, in_slovak=1, reasoning="ok")


# ── _load_qa ───────────────────────────────────────────────────────────────────

def test_load_qa_valid(tmp_path: Path) -> None:
    p = tmp_path / "qa.jsonl"
    p.write_text('{"id": "q1", "question": "test"}\n', encoding="utf-8")
    rows = _load_qa(p)
    assert len(rows) == 1
    assert rows[0].id == "q1"


def test_load_qa_returns_gen_qa_rows(tmp_path: Path) -> None:
    p = tmp_path / "qa.jsonl"
    p.write_text('{"id": "q1", "question": "test"}\n', encoding="utf-8")
    rows = _load_qa(p)
    assert isinstance(rows[0], GenQARow)


def test_load_qa_skips_empty_lines(tmp_path: Path) -> None:
    p = tmp_path / "qa.jsonl"
    p.write_text('{"id": "q1", "question": "a"}\n\n{"id": "q2", "question": "b"}\n', encoding="utf-8")
    rows = _load_qa(p)
    assert len(rows) == 2


def test_load_qa_handles_bom(tmp_path: Path) -> None:
    p = tmp_path / "qa.jsonl"
    p.write_bytes(b'\xef\xbb\xbf{"id": "q1", "question": "test"}\n')
    rows = _load_qa(p)
    assert rows[0].id == "q1"


def test_load_qa_invalid_json_raises_value_error(tmp_path: Path) -> None:
    p = tmp_path / "qa.jsonl"
    p.write_text('{"id": "q1", "question": "ok"}\nNOT JSON\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        _load_qa(p)


# ── _judge — prompt construction ───────────────────────────────────────────────

def test_judge_normal_uses_standard_system() -> None:
    client = _make_judge_client(_NORMAL_SCORES)
    _judge(client, "Čo robí Pivo?", [_CHUNK], "Doplní život.", "Doplní 1 život.", is_refusal=False)
    messages = client.beta.chat.completions.parse.call_args.kwargs["messages"]
    assert messages[0]["content"] == _JUDGE_SYSTEM


def test_judge_refusal_uses_refusal_system() -> None:
    client = _make_judge_client(_REFUSAL_SCORES)
    _judge(client, "Kto vyhral MS?", [], "Nemám informáciu.", "", is_refusal=True)
    messages = client.beta.chat.completions.parse.call_args.kwargs["messages"]
    assert messages[0]["content"] == _JUDGE_SYSTEM_REFUSAL


def test_judge_normal_user_prompt_contains_gold_answer() -> None:
    client = _make_judge_client(_NORMAL_SCORES)
    _judge(client, "Čo robí Pivo?", [_CHUNK], "Doplní život.", "Referenčná odpoveď XYZ.", is_refusal=False)
    user_msg = client.beta.chat.completions.parse.call_args.kwargs["messages"][1]["content"]
    assert "Referenčná odpoveď XYZ." in user_msg


def test_judge_normal_user_prompt_contains_generated_answer() -> None:
    client = _make_judge_client(_NORMAL_SCORES)
    _judge(client, "Čo robí Pivo?", [_CHUNK], "VYGENEROVANA_ODPOVED", "gold", is_refusal=False)
    user_msg = client.beta.chat.completions.parse.call_args.kwargs["messages"][1]["content"]
    assert "VYGENEROVANA_ODPOVED" in user_msg


def test_judge_refusal_user_prompt_excludes_gold_answer() -> None:
    client = _make_judge_client(_REFUSAL_SCORES)
    _judge(client, "Otázka?", [], "Odmietnutie.", "GOLD_NESMIE_BYT_TU", is_refusal=True)
    user_msg = client.beta.chat.completions.parse.call_args.kwargs["messages"][1]["content"]
    assert "GOLD_NESMIE_BYT_TU" not in user_msg


def test_judge_normal_passes_judge_scores_schema() -> None:
    client = _make_judge_client(_NORMAL_SCORES)
    _judge(client, "q", [], "a", "g", is_refusal=False)
    assert client.beta.chat.completions.parse.call_args.kwargs["response_format"] is JudgeScores


def test_judge_refusal_passes_refusal_scores_schema() -> None:
    client = _make_judge_client(_REFUSAL_SCORES)
    _judge(client, "q", [], "a", "g", is_refusal=True)
    assert client.beta.chat.completions.parse.call_args.kwargs["response_format"] is JudgeRefusalScores


def test_judge_returns_pydantic_model() -> None:
    client = _make_judge_client(_NORMAL_SCORES)
    result = _judge(client, "q", [], "a", "g", is_refusal=False)
    assert isinstance(result, JudgeScores)
    assert result.faithfulness == 2


# ── _eval_case — output ────────────────────────────────────────────────────────

def test_eval_case_returns_eval_result() -> None:
    result = _eval_case(_NORMAL_ROW, _make_agent([_CHUNK]), _make_gen_client(), _make_judge_client(_NORMAL_SCORES))
    assert isinstance(result, EvalResult)


def test_eval_case_judge_scores_is_pydantic_model() -> None:
    result = _eval_case(_NORMAL_ROW, _make_agent([_CHUNK]), _make_gen_client(), _make_judge_client(_NORMAL_SCORES))
    assert isinstance(result.judge_scores, JudgeScores)


def test_eval_case_gold_retrieved_true_when_gold_in_selected() -> None:
    result = _eval_case(_NORMAL_ROW, _make_agent([_CHUNK]), _make_gen_client(), _make_judge_client(_NORMAL_SCORES))
    assert result.gold_retrieved is True


def test_eval_case_gold_retrieved_false_when_gold_missing() -> None:
    other_chunk = {**_CHUNK, "id": "card_hnede_bang"}
    result = _eval_case(_NORMAL_ROW, _make_agent([other_chunk]), _make_gen_client(), _make_judge_client(_NORMAL_SCORES))
    assert result.gold_retrieved is False


def test_eval_case_gold_retrieved_none_for_refusal() -> None:
    result = _eval_case(
        _REFUSAL_ROW,
        _make_agent([]),
        _make_gen_client("Nemám k tomu v pravidlách informáciu."),
        _make_judge_client(_REFUSAL_SCORES),
    )
    assert result is not None
    assert result.gold_retrieved is None


def test_eval_case_preserves_gold_answer() -> None:
    result = _eval_case(_NORMAL_ROW, _make_agent([_CHUNK]), _make_gen_client(), _make_judge_client(_NORMAL_SCORES))
    assert result.gold_answer == _NORMAL_ROW.gold_answer


def test_eval_case_preserves_gold_chunk_ids() -> None:
    result = _eval_case(_NORMAL_ROW, _make_agent([_CHUNK]), _make_gen_client(), _make_judge_client(_NORMAL_SCORES))
    assert result.gold_chunk_ids == ["card_hnede_pivo"]


def test_eval_case_returns_none_on_agent_llm_unavailable() -> None:
    agent = MagicMock()
    agent.collect_context.side_effect = LLMUnavailable("quota")
    assert _eval_case(_NORMAL_ROW, agent, _make_gen_client(), _make_judge_client(_NORMAL_SCORES)) is None


def test_eval_case_returns_none_on_gen_llm_unavailable() -> None:
    gen_client = MagicMock()
    gen_client.stream_answer.side_effect = LLMUnavailable("quota")
    assert _eval_case(_NORMAL_ROW, _make_agent([_CHUNK]), gen_client, _make_judge_client(_NORMAL_SCORES)) is None


# ── _avg ───────────────────────────────────────────────────────────────────────

def test_avg_empty_returns_zero() -> None:
    assert _avg([]) == 0.0


def test_avg_correct() -> None:
    assert _avg([1.0, 2.0, 3.0]) == pytest.approx(2.0)


# ── _format_table ──────────────────────────────────────────────────────────────

def _make_eval_result(
    cat: str, gold: bool,
    faith: int = 2, correct: int = 2, cites: int = 1, slovak: int = 1, searches: int = 1,
) -> EvalResult:
    return EvalResult(
        id="gq001", category=cat, question="test?", gold_answer="answer",
        gold_chunk_ids=["card_test"],
        agent_tool_calls=[ToolCallLog(query="q", variant="sparse", k=3, retrieved_ids=[], n_new=0)] * searches,
        agent_selected_chunk_ids=["card_test"],
        gold_retrieved=gold,
        generated_answer="generated",
        judge_scores=JudgeScores(faithfulness=faith, correctness=correct, cites_sources=cites, in_slovak=slovak, reasoning="ok"),
        n_search_calls=searches, n_selected_chunks=1, tool_token_count=300, gen_latency_ms=1000,
    )


def _make_refusal_eval_result(refused: int = 1, slovak: int = 1) -> EvalResult:
    return EvalResult(
        id="gq099", category="refusal", question="test?", gold_answer="",
        gold_chunk_ids=[], agent_tool_calls=[], agent_selected_chunk_ids=[],
        gold_retrieved=None, generated_answer="Nemám informáciu.",
        judge_scores=JudgeRefusalScores(refused_correctly=refused, in_slovak=slovak, reasoning="ok"),
        n_search_calls=1, n_selected_chunks=0, tool_token_count=None, gen_latency_ms=500,
    )


def test_format_table_contains_header() -> None:
    results = [_make_eval_result("lookup", True)]
    table = _format_table({"lookup": results}, results)
    assert "Kategória" in table
    assert "Gold@sel" in table


def test_format_table_has_total_row() -> None:
    results = [_make_eval_result("lookup", True), _make_eval_result("lookup", False)]
    table = _format_table({"lookup": results}, results)
    assert "TOTAL" in table


def test_format_table_no_total_row_for_refusal_only() -> None:
    results = [_make_refusal_eval_result()]
    table = _format_table({"refusal": results}, results)
    assert "TOTAL" not in table


def test_format_table_refusal_row_shows_refused_pct() -> None:
    results = [_make_refusal_eval_result(refused=1), _make_refusal_eval_result(refused=0)]
    table = _format_table({"refusal": results}, results)
    assert "refused:" in table
    assert "50%" in table


def test_format_table_lookup_row_shows_gold_pct() -> None:
    results = [_make_eval_result("lookup", True), _make_eval_result("lookup", True), _make_eval_result("lookup", False)]
    table = _format_table({"lookup": results}, results)
    assert "67%" in table
