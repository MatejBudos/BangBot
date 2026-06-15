"""Gemini 2.0 Flash client with streaming and rate-limit handling."""
from __future__ import annotations

import os
import time
from typing import Iterator

from google import genai
from google.genai import types

from src.generation.prompts import SYSTEM_PROMPT, format_context

_MODEL_ID = "gemini-2.0-flash"


class LLMUnavailable(Exception):
    """Raised when Gemini API is unavailable or rate-limited."""


class GeminiClient:
    """Streaming Gemini client for Bang! RAG answers."""

    def __init__(self, api_key: str | None = None) -> None:
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY not set")
        self._client = genai.Client(api_key=key)
        # Populated after each call for logging
        self.last_token_count: int | None = None
        self.last_latency_ms: float | None = None

    def stream_answer(self, query: str, chunks: list[dict]) -> Iterator[str]:
        """Stream an answer for query given retrieved chunks.

        Yields text fragments as they arrive. Raises LLMUnavailable on quota/network errors.
        """
        context = format_context(chunks)
        prompt = f"{context}\n\nOtázka: {query}"

        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
        )

        t0 = time.monotonic()
        try:
            response = self._client.models.generate_content_stream(
                model=_MODEL_ID,
                contents=prompt,
                config=config,
            )
            token_count = 0
            for chunk in response:
                text = chunk.text if chunk.text else ""
                if text:
                    yield text
            self.last_token_count = token_count
        except Exception as exc:
            msg = str(exc).lower()
            if any(kw in msg for kw in ("quota", "rate", "429", "503", "unavailable", "resource exhausted")):
                raise LLMUnavailable(str(exc)) from exc
            raise
        finally:
            self.last_latency_ms = (time.monotonic() - t0) * 1000
