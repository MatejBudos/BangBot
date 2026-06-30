"""Agentic RAG loop — iterative retrieval via OpenAI tool-calling."""
from __future__ import annotations

import json
import os
import tomllib
from pathlib import Path
from typing import Any

from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError

from src.generation.openai_client import LLMUnavailable
from src.retrieval.lancedb_store import HybridRetriever
from src.schemas import ToolCallLog

_CONFIG_DIR = Path(__file__).parent.parent.parent / "config"

with (_CONFIG_DIR / "agent.toml").open("rb") as _f:
    _cfg = tomllib.load(_f)

_MODEL_ID: str = _cfg["agent_model_id"]
_MAX_ITERATIONS: int = _cfg["max_iterations"]
_MAX_K_PER_CALL: int = _cfg["max_k_per_call"]
_MAX_TOTAL_CHUNKS: int = _cfg["max_total_chunks"]
_TEMPERATURE: float = _cfg["temperature"]
_SEED: int = _cfg["seed"]

_AGENT_SYSTEM_PROMPT: str = (_CONFIG_DIR / "prompts" / "agent.md").read_text(encoding="utf-8").strip()

_SELECT_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "select_chunks",
        "description": (
            "POVINNÝ záverečný krok. Vyber relevantné chunks podľa ID — ostatné budú zahodené. "
            "Zavolaj VŽDY po dokončení všetkých search_rules volaní, pred tým ako odpovieš."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Zoznam chunk ID (z hranatých zátvoriek vo výsledkoch) ktoré chceš zachovať.",
                }
            },
            "required": ["ids"],
        },
    },
}

_SEARCH_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "search_rules",
        "description": (
            "Vyhľadaj v pravidlách kartovej hry Bang!. "
            "Použi 'sparse' (predvolené, najlepší R@1=0.88) pre presné mená kariet "
            "a kľúčové slová. "
            "Použi 'dense' (R@1=0.40) pre sémantické/pojmové otázky bez konkrétneho mena. "
            "Použi 'hybrid' (R@1=0.61) ak si neistý alebo kombinuješ oba prístupy."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Vyhľadávací dotaz v slovenčine.",
                },
                "variant": {
                    "type": "string",
                    "enum": ["sparse", "dense", "hybrid"],
                    "description": "Typ vyhľadávania (predvolene 'sparse').",
                },
                "k": {
                    "type": "integer",
                    "description": "Počet výsledkov (1–5, predvolene 3).",
                    "minimum": 1,
                    "maximum": 5,
                },
            },
            "required": ["query"],
        },
    },
}


class BangAgent:
    """Decides what to retrieve iteratively via tool-calling, then returns collected chunks."""

    def __init__(self, retriever: HybridRetriever, api_key: str | None = None) -> None:
        key = api_key or os.environ.get("OPENAI_KEY")
        if not key:
            raise ValueError("OPENAI_KEY not set")
        self._client = OpenAI(api_key=key)
        self._retriever = retriever
        self.tool_calls_log: list[ToolCallLog] = []
        self.last_tool_token_count: int | None = None
        self._kept_ids: list[str] | None = None

    def collect_context(self, query: str) -> list[dict]:
        """Run tool-calling loop and return deduplicated chunks (max _MAX_TOTAL_CHUNKS).

        Populates self.tool_calls_log and self.last_tool_token_count as side effects.
        Raises LLMUnavailable on API errors.
        """
        self.tool_calls_log: list[ToolCallLog] = []
        self.last_tool_token_count = None
        self._kept_ids = None

        seen_ids: set[str] = set()
        all_chunks: list[dict] = []

        messages: list[dict] = [
            {"role": "system", "content": _AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]

        for _ in range(_MAX_ITERATIONS):
            try:
                response = self._client.chat.completions.create(
                    model=_MODEL_ID,
                    messages=messages,
                    tools=[_SEARCH_TOOL, _SELECT_TOOL],
                    tool_choice="auto",
                    temperature=_TEMPERATURE,
                    parallel_tool_calls=False,
                    seed=_SEED,
                )
            except (RateLimitError, APIStatusError, APIConnectionError) as exc:
                raise LLMUnavailable(str(exc)) from exc

            choice = response.choices[0]
            msg = choice.message

            if response.usage:
                self.last_tool_token_count = (
                    (self.last_tool_token_count or 0) + response.usage.total_tokens
                )

            # Required by OpenAI protocol: append the full assistant message
            messages.append(msg.model_dump(exclude_unset=False))

            if choice.finish_reason == "stop" or not msg.tool_calls:
                break

            done = False
            for tool_call in msg.tool_calls:
                if tool_call.function.name == "select_chunks":
                    content = self._execute_select_call(tool_call)
                    done = True
                else:
                    content = self._execute_tool_call(tool_call, seen_ids, all_chunks)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": content,
                })
            if done:
                break

        if self._kept_ids is not None:
            kept_set = set(self._kept_ids)
            return [c for c in all_chunks if c.get("id") in kept_set]
        return all_chunks[:_MAX_TOTAL_CHUNKS]

    def _execute_tool_call(
        self,
        tool_call: Any,
        seen_ids: set[str],
        all_chunks: list[dict],
    ) -> str:
        """Execute one search_rules call, update seen_ids + all_chunks, return result summary."""
        try:
            args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError:
            self.tool_calls_log.append(ToolCallLog(
                query="<parse error>", variant="sparse", k=0, retrieved_ids=[], n_new=0,
            ))
            return "Chyba: nepodarilo sa spracovať argumenty nástroja."

        query = args.get("query", "")
        variant = args.get("variant", "sparse")
        if variant not in ("sparse", "dense", "hybrid"):
            variant = "sparse"
        k = min(int(args.get("k", 3)), _MAX_K_PER_CALL)

        chunks = self._retriever.search(query, k=k, variant=variant)
        retrieved_ids = [c.get("id", "") for c in chunks]

        new_chunks = [c for c in chunks if c.get("id") not in seen_ids]
        for c in new_chunks:
            seen_ids.add(c.get("id", ""))
            all_chunks.append(c)

        self.tool_calls_log.append(ToolCallLog(
            query=query, variant=variant, k=k,
            retrieved_ids=retrieved_ids, n_new=len(new_chunks),
        ))

        if not chunks:
            return f"Nájdené: 0 výsledkov pre dotaz '{query}' [{variant}]."

        lines = [f"Nájdené: {len(new_chunks)} nových, {len(chunks) - len(new_chunks)} už známych."]
        for c in new_chunks:
            name = c.get("name_sk") or c.get("section_title") or c.get("type", "")
            lines.append(f"[{c.get('id', '')}] {name}:\n{c.get('text', '')}")
        return "\n\n".join(lines)

    def _execute_select_call(self, tool_call: Any) -> str:
        """Process select_chunks call — store kept IDs, terminate loop."""
        try:
            args = json.loads(tool_call.function.arguments)
            ids = args.get("ids", [])
            self._kept_ids = [str(i) for i in ids] if isinstance(ids, list) else []
        except json.JSONDecodeError:
            self._kept_ids = []
        return f"Výber uložený: {len(self._kept_ids)} chunks."
