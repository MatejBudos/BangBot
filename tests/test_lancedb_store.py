"""Tests for lancedb_store: schema normalization and FTS boost logic."""
import numpy as np
import pytest

from src.retrieval.lancedb_store import _normalize_chunk, build_table

_FAKE_VEC = [0.0] * 768


# ---------------------------------------------------------------------------
# _normalize_chunk schema
# ---------------------------------------------------------------------------

def test_normalize_chunk_includes_sk_name():
    chunk = {"id": "x", "type": "card", "text": "t", "sk_name": "Léčka", "alt_names": []}
    record = _normalize_chunk(chunk, "t", _FAKE_VEC)
    assert record["sk_name"] == "Léčka"


def test_normalize_chunk_includes_alt_names_list():
    chunk = {"id": "x", "type": "card", "text": "t", "sk_name": "", "alt_names": ["Ambush", "Léčka"]}
    record = _normalize_chunk(chunk, "t", _FAKE_VEC)
    assert record["alt_names"] == ["Ambush", "Léčka"]


def test_normalize_chunk_alt_names_defaults_to_empty_list():
    chunk = {"id": "x", "type": "card", "text": "t"}
    record = _normalize_chunk(chunk, "t", _FAKE_VEC)
    assert record["alt_names"] == []


def test_normalize_chunk_sk_name_defaults_to_empty_string():
    chunk = {"id": "x", "type": "card", "text": "t"}
    record = _normalize_chunk(chunk, "t", _FAKE_VEC)
    assert record["sk_name"] == ""


# ---------------------------------------------------------------------------
# build_table: text_lemmatized contains boosted name variants
# ---------------------------------------------------------------------------

def _single_card(sk_name="", alt_names=None, name_sk="TestCard", name_orig="testcard"):
    return {
        "id": f"card_test_{name_orig}",
        "type": "card",
        "name_sk": name_sk,
        "sk_name": sk_name or name_sk,
        "alt_names": alt_names or [],
        "name_orig": name_orig,
        "category": "hneda",
        "expansion": "base",
        "image_path": "",
        "text": "Popis karty.",
    }


def _build_and_fetch(tmp_path, chunks):
    embeddings = np.zeros((len(chunks), 768), dtype=np.float32)
    build_table(chunks, embeddings, str(tmp_path))
    import lancedb
    table = lancedb.connect(str(tmp_path)).open_table("bang_chunks")
    return table.to_arrow().to_pylist()


def test_card_text_lemmatized_contains_sk_name_token(tmp_path):
    """sk_name tokens must appear in text_lemmatized (3× repeated name block)."""
    chunk = _single_card(sk_name="Léčka", alt_names=[])
    records = _build_and_fetch(tmp_path, [chunk])
    tl = records[0]["text_lemmatized"]
    # "Léčka" after strip-diacritics → "lecka", after lemmatization still present
    assert "lecka" in tl


def test_card_text_lemmatized_contains_alt_name_token(tmp_path):
    """alt_names tokens must appear in text_lemmatized."""
    chunk = _single_card(sk_name="Léčka", alt_names=["Ambush"])
    records = _build_and_fetch(tmp_path, [chunk])
    tl = records[0]["text_lemmatized"]
    assert "ambush" in tl


def test_card_text_lemmatized_multiple_alt_names(tmp_path):
    """All alt_names tokens must appear in text_lemmatized."""
    chunk = _single_card(
        name_sk="Fratelli di Sangue",
        name_orig="fratellidisangue",
        sk_name="Pokrvní bratia",
        alt_names=["Blood Brothers", "Pokrvní Bratři"],
    )
    records = _build_and_fetch(tmp_path, [chunk])
    tl = records[0]["text_lemmatized"]
    assert "blood" in tl
    assert "brothers" in tl


def test_card_name_block_repeated_three_times(tmp_path):
    """The name token must appear at least 3× in text_lemmatized for cards."""
    chunk = _single_card(name_sk="Agguato", name_orig="agguato", sk_name="Agguato", alt_names=[])
    records = _build_and_fetch(tmp_path, [chunk])
    tl = records[0]["text_lemmatized"]
    assert tl.count("agguato") >= 3


def test_rule_section_name_not_tripled(tmp_path):
    """Rule section chunks must NOT triple the title — no boost."""
    chunk = {
        "id": "rule_general_prehled",
        "type": "rule_section",
        "section_title": "Prehľad",
        "name_sk": "",
        "sk_name": "",
        "alt_names": [],
        "name_orig": "",
        "category": "",
        "expansion": "",
        "image_path": "",
        "source": "general",
        "text": "Stručný prehľad pravidiel hry.",
    }
    embeddings = np.zeros((1, 768), dtype=np.float32)
    build_table([chunk], embeddings, str(tmp_path))
    import lancedb
    records = lancedb.connect(str(tmp_path)).open_table("bang_chunks").to_arrow().to_pylist()
    tl = records[0]["text_lemmatized"]
    assert tl.count("prehla") < 3  # token after diacritic-strip; must appear ≤2×


def test_card_without_sk_name_or_alt_names_still_boosts_name_sk(tmp_path):
    """When sk_name == name_sk and alt_names is empty, name_sk still gets 3× boost."""
    chunk = _single_card(name_sk="Bang", name_orig="bang", sk_name="Bang", alt_names=[])
    records = _build_and_fetch(tmp_path, [chunk])
    tl = records[0]["text_lemmatized"]
    assert tl.count("bang") >= 3
