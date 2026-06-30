"""Shared Pydantic schemas for structured data across the project."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class JudgeScores(BaseModel):
    """LLM judge output for normal (non-refusal) eval cases."""
    faithfulness: Literal[0, 1, 2]
    correctness: Literal[0, 1, 2]
    cites_sources: Literal[0, 1]
    in_slovak: Literal[0, 1]
    reasoning: str


class JudgeRefusalScores(BaseModel):
    """LLM judge output for expected-refusal eval cases."""
    refused_correctly: Literal[0, 1]
    in_slovak: Literal[0, 1]
    reasoning: str


class GenQARow(BaseModel):
    """One row in eval/gen_qa.jsonl."""
    id: str
    question: str
    gold_answer: str = ""
    gold_chunk_ids: list[str] = Field(default_factory=list)
    category: Literal["lookup", "paraphrase", "interaction", "refusal"] = "lookup"
    expected_refusal: bool = False


class ToolCallLog(BaseModel):
    """One search_rules call recorded by BangAgent."""
    query: str
    variant: Literal["sparse", "dense", "hybrid"]
    k: int
    retrieved_ids: list[str]
    n_new: int


class EvalResult(BaseModel):
    """One completed eval case written to gen_results.jsonl."""
    id: str
    category: str
    question: str
    gold_answer: str
    gold_chunk_ids: list[str]
    agent_tool_calls: list[ToolCallLog]
    agent_selected_chunk_ids: list[str]
    gold_retrieved: bool | None
    generated_answer: str
    judge_scores: JudgeScores | JudgeRefusalScores
    n_search_calls: int
    n_selected_chunks: int
    tool_token_count: int | None
    gen_latency_ms: int


class RunConfig(BaseModel):
    """Snapshot of the configuration used for one eval run."""
    timestamp: str
    agent_model: str
    gen_model: str
    judge_model: str
    agent_prompt_md5: str
    gen_prompt_md5: str
    judge_prompt_md5: str
    agent_max_iterations: int
    agent_max_k_per_call: int
    agent_max_total_chunks: int
