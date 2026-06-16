"""Generate eval Q&A draft from chunks.jsonl via OpenAI structured outputs.

Output: eval/qa_draft.jsonl
Each line: {"chunk_id": "...", "question": "...", "category": "lookup|paraphrase", "gold_chunk_ids": ["..."]}
Run: python scripts/generate_eval.py
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).parent.parent))

import openai
from pydantic import BaseModel

_CHUNKS_PATH = Path(__file__).parent.parent / "artifacts" / "chunks.jsonl"
_OUTPUT_PATH = Path(__file__).parent.parent / "eval" / "qa_draft.jsonl"
_MODEL = "gpt-4o-mini"
_BATCH_SIZE = 5
_SLEEP_BETWEEN_BATCHES = 1.0
_MAX_RETRIES = 4


# ---------------------------------------------------------------------------
# Response schema
# ---------------------------------------------------------------------------

class EvalItem(BaseModel):
    chunk_id: str
    question: str
    category: Literal["lookup", "paraphrase"]


class EvalBatch(BaseModel):
    items: list[EvalItem]


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM = (
    "Si generátor eval dát pre RAG systém nad pravidlami kartovej hry Bang!. "
    "Existuju rôzne typy kariet ktoré sú označené v poli 'category': "
    "Hneda = hrá sa z ruky a efekt vstupuje do platnosti po zahratí. "
    "Zelená = vykladá sa pred hráča a dá sa zahrať aj mimo svojho kola a potom funguje ako hnedá karta. "
    "Modrá = vykladá sa pred hráča - pasívny efekt karty. "
    "Rozšírenia = Wild west, fistful, highnoon - karta sa vyloží na stôl a platí pre všetkých hráčov pokiaľ sa nevyloží nová karta. "
    "Postavy = každý hráč hrá za nejakú postavu ktorej schopnosť má k dispozícií počas celej hry ak to neobmedzí nejaká karta/postava vyslovene."
)

_USER_TEMPLATE = """\
Pre každý chunk vygeneruj presne 2 otázky po slovensky na základe typu karty.
Ak je k dispozícií slovenský názov karty tak uprednosti ten pri dotaze; ak neexistuje, použi český alebo pôvodný taliansky (neprekladaj).

Typy otázok (vždy 1 od každého na chunk):
- lookup: priama faktická otázka na efekt/pravidlo v chunku (môže použiť názov karty/sekcie)
- paraphrase: preformulovaná otázka na tú istú informáciu bez doslovného mena karty/sekcie

Chunky:
{chunks_json}

Vráť presne 2×N položiek (pre {n} chunkov) v poli `items`."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_chunks() -> list[dict]:
    chunks = []
    with open(_CHUNKS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def _load_done_ids(output_path: Path) -> set[str]:
    done: set[str] = set()
    if not output_path.exists():
        return done
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    done.add(json.loads(line)["chunk_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return done


def _parse_retry_after(exc: Exception) -> float:
    m = re.search(r"retry after (\d+)", str(exc), re.IGNORECASE)
    return float(m.group(1)) + 2.0 if m else 30.0


def _call_openai(client: openai.OpenAI, batch: list[dict]) -> list[EvalItem]:
    user_msg = _USER_TEMPLATE.format(
        chunks_json=json.dumps(batch, ensure_ascii=False, indent=2),
        n=len(batch),
    )

    for attempt in range(_MAX_RETRIES):
        try:
            resp = client.beta.chat.completions.parse(
                model=_MODEL,
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.7,
                response_format=EvalBatch,
            )
            break
        except openai.RateLimitError as exc:
            if attempt < _MAX_RETRIES - 1:
                wait = _parse_retry_after(exc)
                print(f"    429 rate limit — waiting {wait:.0f}s (retry {attempt + 2}/{_MAX_RETRIES})…", flush=True)
                time.sleep(wait)
            else:
                raise

    result = resp.choices[0].message.parsed
    if result is None:
        # refusal or parse failure — surface the raw content for debugging
        raw = resp.choices[0].message.content or ""
        print(f"  WARN: parsed=None, raw={raw[:200]!r}", file=sys.stderr)
        return []
    return result.items


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("OPENAI_KEY")
    if not api_key:
        print("ERROR: OPENAI_KEY not set in environment.", file=sys.stderr)
        return 1

    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    chunks = _load_chunks()
    done_ids = _load_done_ids(_OUTPUT_PATH)

    remaining = [c for c in chunks if c["id"] not in done_ids]
    total = len(chunks)
    print(f"Chunks: {total} total, {total - len(remaining)} already done, {len(remaining)} to process.")

    if not remaining:
        print("Nothing to do.")
        return 0

    client = openai.OpenAI(api_key=api_key)

    batches = [remaining[i : i + _BATCH_SIZE] for i in range(0, len(remaining), _BATCH_SIZE)]
    generated = 0
    errors = 0

    with open(_OUTPUT_PATH, "w", encoding="utf-8") as out_f:
        for batch_idx, batch in enumerate(batches):
            batch_ids = [c["id"] for c in batch]
            print(f"  Batch {batch_idx + 1}/{len(batches)}: {batch_ids}", flush=True)

            try:
                items = _call_openai(client, batch)
            except Exception as exc:
                print(f"  ERROR on batch {batch_idx + 1}: {exc}", file=sys.stderr)
                errors += 1
                time.sleep(_SLEEP_BETWEEN_BATCHES * 3)
                continue

            if not items:
                print("  WARN: no items returned for this batch", file=sys.stderr)
                continue

            print(f"  Returned {len(items)} items")
            for item in items:
                if item.chunk_id not in {c["id"] for c in batch}:
                    print(f"  SKIP unknown chunk_id: {item.chunk_id!r}", file=sys.stderr)
                    continue
                record = {
                    "chunk_id": item.chunk_id,
                    "question": item.question.strip(),
                    "category": item.category,
                    "gold_chunk_ids": [item.chunk_id],
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()
                generated += 1

            if batch_idx < len(batches) - 1:
                time.sleep(_SLEEP_BETWEEN_BATCHES)

    print(f"Done. Generated {generated} questions, {errors} batch errors.")
    print(f"Output: {_OUTPUT_PATH}")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
