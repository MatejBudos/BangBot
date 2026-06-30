"""OpenAI streaming client for Bang! RAG answers."""
from __future__ import annotations

import os
import time
import tomllib
from pathlib import Path
from typing import Iterator

from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError

from src.generation.prompts import SYSTEM_PROMPT, format_context

_CONFIG_DIR = Path(__file__).parent.parent.parent / "config"
with (_CONFIG_DIR / "agent.toml").open("rb") as _f:
    _cfg = tomllib.load(_f)

_MODEL_ID: str = _cfg["gen_model_id"]


class LLMUnavailable(Exception):
    """Raised when OpenAI API is unavailable or rate-limited."""


class OpenAIClient:
    """Streaming OpenAI client for Bang! RAG answers."""

    def __init__(self, api_key: str | None = None) -> None:
        key = api_key or os.environ.get("OPENAI_KEY")
        if not key:
            raise ValueError("OPENAI_KEY not set")
        self._client = OpenAI(api_key=key)
        self.last_token_count: int | None = None
        self.last_latency_ms: float | None = None

    def stream_answer(self, query: str, chunks: list[dict]) -> Iterator[str]:
        """Stream an answer for query given retrieved chunks.

        Yields text fragments as they arrive. Raises LLMUnavailable on quota/network errors.
        """
        context = format_context(chunks)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{context}\n\nOtázka: {query}"},
        ]

        t0 = time.monotonic()
        try:
            stream = self._client.chat.completions.create(
                model=_MODEL_ID,
                messages=messages,
                stream=True,
                stream_options={"include_usage": True},
            )
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                if chunk.usage:
                    self.last_token_count = chunk.usage.total_tokens
        except (RateLimitError, APIStatusError, APIConnectionError) as exc:
            raise LLMUnavailable(str(exc)) from exc
        finally:
            self.last_latency_ms = (time.monotonic() - t0) * 1000
