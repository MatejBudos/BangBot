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


def test_parse_cards_caption_name_from_caption_label():
    content = (CORPUS_DIR / "hnede.tex").read_text(encoding="utf-8")
    chunks = _parse_cards("hnede", content)
    pivo = next(c for c in chunks if c["id"] == "card_hneda_birra")
    assert pivo["caption_name"] == "Pivo"


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
        assert c["caption_name"] != "", f"Empty caption_name: {c}"
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


def test_parse_glossary_per_term():
    content = (CORPUS_DIR / "vysvetlivky.tex").read_text(encoding="utf-8")
    chunks = _parse_glossary(content)
    ids = {c["id"] for c in chunks}
    assert len(chunks) >= 5
    assert all(c["type"] == "glossary" for c in chunks)
    assert all("section_title" in c and "text" in c for c in chunks)
    # "Ťahať kartu" a "Potiahnuť kartu" musia byť samostatné entries
    assert "glossary_tahatkartu" in ids
    assert "glossary_potiahnutkartu" in ids
    # Každý text musí obsahovať aj názov pojmu (prefixovaný)
    by_id = {c["id"]: c for c in chunks}
    tc = by_id["glossary_tahatkartu"]
    assert "Ťahať kartu" in tc["text"]
    assert "Potiahnuť kartu" not in tc["text"]


# ---------------------------------------------------------------------------
# \skname and \altnames tag tests
# ---------------------------------------------------------------------------

_MINIPAGE_TEMPLATE = r"""
\begin{{minipage}}[t]{{\minipageSize}}\centering
  \includegraphics[width=\cardSize\linewidth]{{hnede/01_{stem}.png}}
  \caption[{label}]{{{text}}}
  {extra}
\end{{minipage}}
"""


def _make_card_tex(stem: str, label: str, text: str, extra: str = "") -> str:
    return _MINIPAGE_TEMPLATE.format(stem=stem, label=label, text=text, extra=extra)


def test_skname_absent_falls_back_to_caption_label():
    """When \\skname is missing, sk_name must equal caption_name (caption label)."""
    tex = _make_card_tex("bang", "Bang!", "Vyber hráča a vystrel naň.")
    chunks = _parse_cards("hnede", tex)
    assert len(chunks) == 1
    assert chunks[0]["sk_name"] == "Bang!"
    assert chunks[0]["sk_name"] == chunks[0]["caption_name"]


def test_altnames_absent_yields_empty_list():
    """When \\altnames is missing, alt_names must be an empty list."""
    tex = _make_card_tex("bang", "Bang!", "Vyber hráča a vystrel naň.")
    chunks = _parse_cards("hnede", tex)
    assert chunks[0]["alt_names"] == []


def test_skname_extracted_when_present():
    """\\skname overrides the caption label as sk_name."""
    tex = _make_card_tex("agguato", "Agguato", "Vzdialenosť je 1.", r"\skname{Léčka}")
    chunks = _parse_cards("hnede", tex)
    assert chunks[0]["sk_name"] == "Léčka"


def test_altnames_single_value_parsed_as_list():
    """\\altnames with one value produces a one-element list."""
    tex = _make_card_tex("agguato", "Agguato", "Vzdialenosť je 1.", r"\altnames{Ambush}")
    chunks = _parse_cards("hnede", tex)
    assert chunks[0]["alt_names"] == ["Ambush"]


def test_altnames_multiple_values_parsed_correctly():
    """\\altnames with comma-separated values produces multi-element list."""
    tex = _make_card_tex(
        "fratellidisangue",
        "Fratelli di Sangue",
        "Daruj jeden život.",
        r"\altnames{Blood Brothers, Pokrvní Bratři}",
    )
    chunks = _parse_cards("hnede", tex)
    assert chunks[0]["alt_names"] == ["Blood Brothers", "Pokrvní Bratři"]


def test_skname_and_altnames_both_present():
    """Both tags present: sk_name and alt_names are set independently."""
    tex = _make_card_tex(
        "agguato",
        "Agguato",
        "Vzdialenosť je 1.",
        "\\skname{Léčka}\n  \\altnames{Ambush}",
    )
    chunks = _parse_cards("hnede", tex)
    assert chunks[0]["sk_name"] == "Léčka"
    assert chunks[0]["alt_names"] == ["Ambush"]
    assert chunks[0]["caption_name"] == "Agguato"


def test_skname_absent_altnames_present():
    """\\skname absent but \\altnames present: sk_name falls back, alt_names populated."""
    tex = _make_card_tex(
        "birra",
        "Pivo",
        "Hráč získa jeden život.",
        r"\altnames{Birra, Beer}",
    )
    chunks = _parse_cards("hnede", tex)
    assert chunks[0]["sk_name"] == "Pivo"
    assert chunks[0]["alt_names"] == ["Birra", "Beer"]


# Integration tests against real corpus files

def test_fistful_agguato_skname_and_altnames():
    content = (CORPUS_DIR / "fistful.tex").read_text(encoding="utf-8")
    chunks = _parse_cards("fistful", content)
    agguato = next((c for c in chunks if c["name_orig"] == "agguato"), None)
    assert agguato is not None, "agguato chunk missing from fistful.tex"
    assert agguato["sk_name"] == "Léčka"
    assert agguato["alt_names"] == ["Ambush"]


def test_fistful_fratellidisangue_multiple_altnames():
    content = (CORPUS_DIR / "fistful.tex").read_text(encoding="utf-8")
    chunks = _parse_cards("fistful", content)
    card = next((c for c in chunks if c["name_orig"] == "fratellidisangue"), None)
    assert card is not None
    assert card["sk_name"] == "Pokrvní bratia"
    assert "Blood Brothers" in card["alt_names"]
    assert len(card["alt_names"]) >= 2


def test_fistful_ranch_only_skname_no_altnames():
    """ranch card has \\skname but no \\altnames."""
    content = (CORPUS_DIR / "fistful.tex").read_text(encoding="utf-8")
    chunks = _parse_cards("fistful", content)
    card = next((c for c in chunks if c["name_orig"] == "ranch"), None)
    assert card is not None
    assert card["sk_name"] == "Ranč"
    assert card["alt_names"] == []


def test_hnede_birra_altnames_no_skname():
    """birra has \\altnames but no \\skname — sk_name must fall back to caption_name."""
    content = (CORPUS_DIR / "hnede.tex").read_text(encoding="utf-8")
    chunks = _parse_cards("hnede", content)
    card = next((c for c in chunks if c["name_orig"] == "birra"), None)
    assert card is not None
    assert card["alt_names"] == ["Birra", "Beer", "pivo"]
    assert card["sk_name"] == card["caption_name"]


def test_all_card_chunks_have_sk_name_and_alt_names_fields():
    """Every card chunk must have sk_name (non-empty str) and alt_names (list)."""
    chunks = [c for c in parse_corpus(CORPUS_DIR) if c["type"] == "card"]
    for c in chunks:
        assert isinstance(c.get("sk_name"), str) and c["sk_name"], \
            f"sk_name missing or empty: {c['id']}"
        assert isinstance(c.get("alt_names"), list), \
            f"alt_names not a list: {c['id']}"
