"""Parse Bang! LaTeX corpus into structured chunks."""
from __future__ import annotations

import re
import json
from pathlib import Path
from typing import Iterator

# Maps source filename stem → (category, expansion)
_FILE_META: dict[str, tuple[str, str]] = {
    "hnede":    ("hneda",    "base"),
    "modre":    ("modra",    "base"),
    "zelene":   ("zelena",   "base"),
    "postavy":  ("postava",  "base"),
    "fistful":  ("fistful",  "fistful"),
    "highnoon": ("highnoon", "highnoon"),
    "wildwest": ("wildwest", "wildwest"),
}


def _strip_comments(text: str) -> str:
    """Remove \\begin{comment}...\\end{comment} blocks."""
    return re.sub(
        r"\\begin\{comment\}.*?\\end\{comment\}",
        "",
        text,
        flags=re.DOTALL,
    )


def _clean_latex(text: str) -> str:
    """Strip common LaTeX markup, preserve Slovak text."""
    # Remove \\textbf{...}, \\textit{...}, \\emph{...} → keep inner text
    text = re.sub(r"\\text(?:bf|it|rm|sf|tt)\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\emph\{([^}]*)\}", r"\1", text)
    # Remove remaining simple commands
    text = re.sub(r"\\[a-zA-Z]+\*?\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\[a-zA-Z]+\*?\s*", "", text)
    # Remove leftover braces
    text = re.sub(r"[{}]", "", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _name_orig_from_path(image_path: str) -> str:
    """Extract name_orig from image path like 'Hnede/01_bang.png' → 'bang'."""
    stem = Path(image_path).stem          # e.g. "01_bang"
    parts = stem.split("_", 1)
    return parts[1] if len(parts) == 2 else stem


def _slugify(text: str) -> str:
    """Convert text to lowercase slug for use in IDs."""
    text = text.lower()
    text = re.sub(r"[áàâä]", "a", text)
    text = re.sub(r"[čć]", "c", text)
    text = re.sub(r"[ď]", "d", text)
    text = re.sub(r"[éèêë]", "e", text)
    text = re.sub(r"[íìîï]", "i", text)
    text = re.sub(r"[ľĺ]", "l", text)
    text = re.sub(r"[ňń]", "n", text)
    text = re.sub(r"[óòôö]", "o", text)
    text = re.sub(r"[ŕř]", "r", text)
    text = re.sub(r"[šś]", "s", text)
    text = re.sub(r"[ťţ]", "t", text)
    text = re.sub(r"[úùûü]", "u", text)
    text = re.sub(r"[ýÿ]", "y", text)
    text = re.sub(r"[žź]", "z", text)
    text = re.sub(r"[^a-z0-9]+", "", text)
    return text


# ---------------------------------------------------------------------------
# Card chunk parser
# ---------------------------------------------------------------------------

_MINIPAGE_RE = re.compile(
    r"\\begin\{minipage\}.*?"          # opening
    r"\\includegraphics\[.*?\]\{([^}]+)\}"  # image path
    r".*?"
    r"\\caption\[([^\]]*)\]\{(.*?)\}"  # caption[label]{text}
    r".*?"
    r"\\end\{minipage\}",
    re.DOTALL,
)


def _parse_cards(source: str, content: str) -> list[dict]:
    """Extract card chunks from a card .tex file content."""
    category, expansion = _FILE_META[source]
    content = _strip_comments(content)
    chunks: list[dict] = []
    seen_ids: set[str] = set()

    for m in _MINIPAGE_RE.finditer(content):
        image_path = m.group(1).strip()
        name_sk_raw = m.group(2).strip()
        text_raw = m.group(3).strip()

        if not image_path or not name_sk_raw:
            continue

        name_orig = _name_orig_from_path(image_path)
        name_sk = _clean_latex(name_sk_raw)
        text = _clean_latex(text_raw)

        if not text:
            continue

        base_id = f"card_{category}_{_slugify(name_orig)}"
        chunk_id = base_id
        # Handle rare duplicates (same card, different figures)
        if chunk_id in seen_ids:
            continue
        seen_ids.add(chunk_id)

        chunks.append({
            "id": chunk_id,
            "type": "card",
            "name_sk": name_sk,
            "name_orig": name_orig,
            "category": category,
            "expansion": expansion,
            "image_path": image_path,
            "text": text,
        })

    return chunks


# ---------------------------------------------------------------------------
# Rule section parser
# ---------------------------------------------------------------------------

_SECTION_RE = re.compile(
    r"\\section\*\{([^}]+)\}(.*?)(?=\\section\*\{|\\chapter\{|\\chapter\*\{|$)",
    re.DOTALL,
)


def _parse_rule_sections(source_key: str, content: str) -> list[dict]:
    """Extract rule_section chunks from a .tex file."""
    content = _strip_comments(content)
    chunks: list[dict] = []
    seen_ids: set[str] = set()

    for m in _SECTION_RE.finditer(content):
        title_raw = m.group(1).strip()
        body_raw = m.group(2).strip()

        title = _clean_latex(title_raw)
        text = _clean_latex(body_raw)

        if not text:
            continue

        slug = _slugify(title)
        chunk_id = f"rule_{source_key}_{slug}"
        if chunk_id in seen_ids:
            continue
        seen_ids.add(chunk_id)

        chunks.append({
            "id": chunk_id,
            "type": "rule_section",
            "section_title": title,
            "source": source_key,
            "text": text,
        })

    return chunks


# ---------------------------------------------------------------------------
# Glossary parser
# ---------------------------------------------------------------------------

def _parse_glossary(content: str) -> list[dict]:
    """Extract a single glossary chunk from vysvetlivky.tex."""
    content = _strip_comments(content)
    # Strip the chapter heading
    content = re.sub(r"\\chapter\*?\{[^}]*\}", "", content)
    text = _clean_latex(content)
    if not text:
        return []
    return [{
        "id": "glossary_vysvetlivky",
        "type": "glossary",
        "text": text,
    }]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_CARD_FILES = {"hnede", "modre", "zelene", "postavy", "fistful", "highnoon", "wildwest"}


def parse_corpus(corpus_dir: str | Path) -> list[dict]:
    """Parse all .tex files in corpus_dir and return list of chunks."""
    corpus_dir = Path(corpus_dir)
    chunks: list[dict] = []

    for tex_file in sorted(corpus_dir.glob("*.tex")):
        stem = tex_file.stem
        content = tex_file.read_text(encoding="utf-8")

        if stem in _CARD_FILES:
            chunks.extend(_parse_cards(stem, content))
            # Also extract Dohoda section from hnede.tex
            if stem == "hnede":
                chunks.extend(_parse_rule_sections("hnede_dohoda", content))

        elif stem == "general_rules":
            chunks.extend(_parse_rule_sections("general", content))

        elif stem == "vysvetlivky":
            chunks.extend(_parse_glossary(content))

    return chunks


def write_chunks_jsonl(chunks: list[dict], out_path: str | Path) -> None:
    """Write chunks to JSONL file."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
