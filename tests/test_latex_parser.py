"""Unit tests for the LaTeX parser (tasks #1, #2)."""
import pytest
from pathlib import Path

from src.ingestion.latex_parser import (
    _parse_cards,
    _parse_rule_sections,
    _parse_glossary,
    _name_orig_from_path,
    _strip_comments,
    _clean_latex,
    parse_corpus,
)

CORPUS_DIR = Path(__file__).parent.parent / "data" / "corpus"


# ---------------------------------------------------------------------------
# Unit tests on helper functions
# ---------------------------------------------------------------------------

def test_name_orig_from_numbered_path():
    assert _name_orig_from_path("Hnede/01_mancato.png") == "mancato"


def test_name_orig_from_simple_path():
    assert _name_orig_from_path("postavy/01_bartcassidy.png") == "bartcassidy"


def test_strip_comments_removes_block():
    src = r"before\begin{comment}hidden content\end{comment}after"
    result = _strip_comments(src)
    assert "hidden" not in result
    assert "before" in result
    assert "after" in result


def test_clean_latex_removes_textbf():
    assert _clean_latex(r"\textbf{Bang!}") == "Bang!"


def test_clean_latex_preserves_slovak_diacritics():
    text = r"\caption[Karta]{Hráč má životov čšťľ}"
    result = _clean_latex(text)
    assert "čšťľ" in result
    assert "Hráč" in result


def test_parse_cards_hnede_extracts_bang():
    content = (CORPUS_DIR / "hnede.tex").read_text(encoding="utf-8")
    chunks = _parse_cards("hnede", content)
    ids = [c["id"] for c in chunks]
    assert "card_hneda_bang" in ids


def test_parse_cards_name_sk_from_caption_label():
    content = (CORPUS_DIR / "hnede.tex").read_text(encoding="utf-8")
    chunks = _parse_cards("hnede", content)
    pivo = next(c for c in chunks if c["id"] == "card_hneda_birra")
    assert pivo["name_sk"] == "Pivo"


def test_parse_cards_category_all_files():
    """Every card file produces chunks with the correct category."""
    expected = {
        "hnede": "hneda",
        "modre": "modra",
        "zelene": "zelena",
        "postavy": "postava",
        "fistful": "fistful",
        "highnoon": "highnoon",
        "wildwest": "wildwest",
    }
    for stem, expected_cat in expected.items():
        content = (CORPUS_DIR / f"{stem}.tex").read_text(encoding="utf-8")
        chunks = _parse_cards(stem, content)
        assert len(chunks) > 0, f"No chunks from {stem}.tex"
        for c in chunks:
            assert c["category"] == expected_cat, f"Wrong category in {stem}.tex: {c}"


def test_parse_cards_postavy_skips_comment_template():
    """postavy.tex comment block has empty captions — must be skipped."""
    content = (CORPUS_DIR / "postavy.tex").read_text(encoding="utf-8")
    chunks = _parse_cards("postavy", content)
    for c in chunks:
        assert c["name_sk"] != "", f"Empty name_sk: {c}"
        assert c["text"] != "", f"Empty text: {c}"


def test_parse_rule_sections_general_rules():
    content = (CORPUS_DIR / "general_rules.tex").read_text(encoding="utf-8")
    chunks = _parse_rule_sections("general", content)
    titles = [c["section_title"] for c in chunks]
    assert "Priebeh ťahu" in titles


def test_parse_rule_sections_dohoda_in_hnede():
    """Dohoda section from hnede.tex extracted as rule_section."""
    content = (CORPUS_DIR / "hnede.tex").read_text(encoding="utf-8")
    chunks = _parse_rule_sections("hnede_dohoda", content)
    assert any("dohoda" in c["id"].lower() for c in chunks), "Dohoda chunk missing"


def test_parse_corpus_no_empty_text_or_id():
    chunks = parse_corpus(CORPUS_DIR)
    for c in chunks:
        assert c["id"], f"Empty id: {c}"
        assert c["text"], f"Empty text: {c}"


def test_parse_corpus_ids_unique():
    chunks = parse_corpus(CORPUS_DIR)
    ids = [c["id"] for c in chunks]
    assert len(ids) == len(set(ids)), "Duplicate IDs found"


def test_parse_corpus_roughly_100_chunks():
    chunks = parse_corpus(CORPUS_DIR)
    assert 60 <= len(chunks) <= 200, f"Unexpected chunk count: {len(chunks)}"


def test_parse_glossary_single_chunk():
    content = (CORPUS_DIR / "vysvetlivky.tex").read_text(encoding="utf-8")
    chunks = _parse_glossary(content)
    assert len(chunks) == 1
    assert chunks[0]["id"] == "glossary_vysvetlivky"
    assert chunks[0]["type"] == "glossary"
    assert len(chunks[0]["text"]) > 50
