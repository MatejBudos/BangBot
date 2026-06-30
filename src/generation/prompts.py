"""System prompt and context formatting for Bang! RAG."""
from __future__ import annotations

from pathlib import Path

_PROMPT_PATH = Path(__file__).parent.parent.parent / "config" / "prompts" / "gen.md"
SYSTEM_PROMPT: str = _PROMPT_PATH.read_text(encoding="utf-8").strip()


def _chunk_label(chunk: dict) -> str:
    """Human-readable label for a chunk used in context header."""
    if chunk.get("type") == "card":
        caption_name = chunk.get("caption_sk", "")
        sk_name = chunk.get("sk_name", "")
        alt_names = chunk.get("alt_names") or []
        name_orig = chunk.get("name_orig", "")
        category = chunk.get("category", "")
        expansion = chunk.get("expansion", "")

        # Collect all unique name variants so LLM recognises the card by any alias
        seen: set[str] = {sk_name.lower()}
        extras: list[str] = []
        for n in [sk_name, name_orig, *alt_names, caption_name]:
            if n and n.lower() not in seen:
                seen.add(n.lower())
                extras.append(n)

        names = f'"{sk_name}"'
        if extras:
            names += f' (tiež: {", ".join(extras)})'
        return f"Karta {names} [typ karty: {category}, rozšírenie: {expansion}]"
    if chunk.get("type") == "rule_section":
        return f'Sekcia "{chunk.get("section_title", "")}" ({chunk.get("source", "")})'
    return "Vysvetlivky"


def format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks as numbered context block for the LLM prompt."""
    parts: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        label = _chunk_label(chunk)
        text = chunk.get("text", "")
        parts.append(f"[Z{i}] {label}: {text}")
    return "\n\n".join(parts)
