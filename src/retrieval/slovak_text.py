"""Slovak text preprocessing pipeline for FTS/BM25 indexing."""
from __future__ import annotations

import re
import unicodedata
import simplemma

# ---------------------------------------------------------------------------
# Diacritic stripping — explicit map for Slovak alphabet
# ---------------------------------------------------------------------------

_DIACRITIC_MAP = str.maketrans(
    "áäčďéíľĺňóôŕšťúýžÁÄČĎÉÍĽĹŇÓÔŔŠŤÚÝŽ",
    "aacdeillnoorstuyzAACDEILLNOORSTUYZ",
)


def strip_diacritics(text: str) -> str:
    """Map Slovak diacritics to ASCII equivalents."""
    return text.translate(_DIACRITIC_MAP)


# ---------------------------------------------------------------------------
# Slovak stopwords (~160 words)
# ---------------------------------------------------------------------------

SK_STOPWORDS: frozenset[str] = frozenset({
    # pronouns
    "ja", "ty", "on", "ona", "ono", "my", "vy", "oni", "ony",
    "mi", "ti", "mu", "jej", "nam", "vam", "im",
    "ma", "ta", "ho", "ju", "nas", "vas", "ich",
    "sa", "si",
    "moj", "tvoj", "jeho", "ich", "nas", "vas",
    "ten", "ta", "to", "toto", "tieto", "tento",
    "ktory", "ktora", "ktore", "ktori",
    "kto", "co", "aky", "aka", "ake",
    # prepositions
    "a", "aj", "ale", "alebo", "ani",
    "do", "od", "na", "po", "za", "pri", "pred", "nad",
    "pod", "cez", "bez", "pre", "zo", "z", "o", "u",
    "k", "ku", "vo", "v", "s", "so",
    # conjunctions & particles
    "ze", "ze", "ze", "lebo", "preto", "teda", "iba", "len",
    "ak", "aby", "ci", "nie", "ano", "tak", "takze", "pritom",
    "este", "uz", "uz", "vtedy", "potom", "zatial",
    "kym", "pokym", "kedze",
    # verbs (very common)
    "je", "su", "byt", "bol", "bola", "bolo", "boli",
    "bude", "budu", "byva", "ma", "maju", "mat",
    "moze", "mozno", "mozno", "mozem", "mozete", "mozno",
    "musiet", "musi", "musia",
    "mozno", "treba", "mozno",
    # articles / determiners (SK has none, but common filler)
    "avsak", "napriek", "voci",
    "kazdy", "kazda", "kazde",
    "niektory", "niektora", "niektore",
    "vsetok", "vsetka", "vsetko", "vsetci", "vsetky",
    "ziadny", "ziadna", "ziadne",
    "iny", "ina", "ine",
    "nejaky", "nejaka", "nejake",
    "sam", "sama", "samo",
    # numbers (written)
    "jeden", "jedna", "jedno", "dva", "dve", "tri", "styri",
    "pat", "sest", "sedem", "osem", "devat", "desat",
    # misc
    "tu", "tam", "kde", "kedy", "ako", "kolko", "preco",
    "hore", "dole", "von", "dnu",
    "resp", "napr", "atd", "tzv",
    "i", "e",
})


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def preprocess_for_fts(text: str) -> str:
    """Lowercase → tokenize → lemmatize(sk) → strip diacritics → remove stopwords → join.

    Lemmatization runs before diacritic stripping because simplemma needs accented
    forms to correctly identify Slovak morphological variants (e.g. hráčov → hráč).
    Stopwords and final output are diacritic-free for robust FTS matching.
    """
    lowered = text.lower()
    tokens = re.findall(r"\w+", lowered)
    lemmas = [simplemma.lemmatize(t, lang="sk") for t in tokens]
    ascii_tokens = [strip_diacritics(lemma) for lemma in lemmas]
    filtered = [t for t in ascii_tokens if t not in SK_STOPWORDS]
    return " ".join(filtered)
