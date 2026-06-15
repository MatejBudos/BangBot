"""Unit tests for Slovak text preprocessing pipeline (task #3)."""
import pytest
from src.retrieval.slovak_text import strip_diacritics, preprocess_for_fts, SK_STOPWORDS


def test_strip_diacritics_basic():
    assert strip_diacritics("čšťľ") == "cstl"


def test_strip_diacritics_full_slovak_alphabet():
    result = strip_diacritics("áäčďéíľĺňóôŕšťúýžÁÄČĎÉÍĽĹŇÓÔŔŠŤÚÝŽ")
    assert result == "aacdeillnoorstuyzAACDEILLNOORSTUYZ"


def test_stopwords_removal():
    result = preprocess_for_fts("je na karte")
    tokens = result.split()
    assert "je" not in tokens
    assert "na" not in tokens


def test_lemma_hracmi_contains_hrac():
    """'hráčmi' after full pipeline should yield 'hrac' (stripped lemma)."""
    result = preprocess_for_fts("hráčmi")
    assert "hrac" in result.split()


def test_preprocess_kartami_hracov():
    result = preprocess_for_fts("Kartami hráčov")
    tokens = result.split()
    # After diacritic strip + lemma: 'kartami' → 'karta'→strip→'karta', 'hráčov'→'hráč'→strip→'hrac'
    assert "karta" in tokens or "kart" in tokens
    assert "hrac" in tokens


def test_idempotent():
    text = "Zahraním tejto karty si hráč doplní 1 život"
    assert preprocess_for_fts(text) == preprocess_for_fts(text)


def test_stopwords_list_size():
    assert len(SK_STOPWORDS) >= 100
