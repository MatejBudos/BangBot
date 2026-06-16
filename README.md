---
title: BangBot
emoji: 🤠
colorFrom: yellow
colorTo: red
sdk: streamlit
sdk_version: "1.48.1"
app_file: app.py
pinned: false
---

# BangRag

Hybrid RAG system for rules of the Slovak card game **Bang!**. Ask questions in Slovak, get answers grounded in the official rulebook.

**Live demo:** *(coming soon — HF Space)*

---

## Problem

The Bang! rulebook exists as a set of LaTeX documents in Slovak. Players frequently argue about edge cases ("Does the Gatling count toward the Bang! limit?"). This project lets you query the rules in natural language and get cited answers.

---

## Architecture

```
.tex files → LaTeX parser → chunks.jsonl
                                 │
                    ┌────────────┴────────────┐
               dense embed              FTS (lemmatized)
          (multilingual-e5-base)       (simplemma + LanceDB)
                    └────────────┬────────────┘
                              RRF fusion (k=60)
                                 │
                            top-5 chunks
                                 │
                         Gemini 2.0 Flash
                                 │
                          streamed answer [Z1][Z2]
```

---

## Key decisions

- **`intfloat/multilingual-e5-base`** — strong Slovak performance without fine-tuning; fits in HF Spaces free CPU tier
- **LanceDB native hybrid + simplemma** — single dependency for vector + FTS; simplemma handles Slovak morphology (kartami → karta) without a full NLP pipeline
- **RRF fusion (k=60)** — simple, parameter-free combination of dense and sparse scores; outperforms weighted sum on short queries
- **Gemini 2.0 Flash** — 1500 free requests/day; graceful fallback to retrieval-only when quota is exhausted
- **HF Spaces + Streamlit** — zero-cost hosting, no Docker needed

---

## Ablation results

*(populated after eval — see `eval/results.md`)*

| Variant | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---|---|---|---|
| Dense only | — | — | — | — |
| Sparse only | — | — | — | — |
| Hybrid (RRF) | — | — | — | — |

---

## Slovak-specific challenges

- **Morphology**: Slovak is highly inflected — "kartami", "kartách", "karte" all mean "card". Simplemma lemmatizes before FTS indexing so all forms match.
- **Diacritics**: Queries often omit accents ("co robi Pivo" vs "čo robí Pivo"). Diacritic stripping is applied to both index and query.
- **Code-switching**: Card names mix Slovak and Italian/English (Mancato, Birra, Gatling). The parser preserves `name_orig` from image filenames and `name_sk` from captions for cross-lingual matching.

---

## Local development

```bash
# 1. Clone and install
git clone https://github.com/<user>/bang-rag.git
cd bang-rag
pip install -r requirements.txt

# 2. Set secrets
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# fill in GEMINI_API_KEY and APP_PASSWORD

# 3. Build the index
python scripts/build_index.py

# 4. Run the app
streamlit run app.py
```

---

## Deploy

Push to `main` triggers a GitHub Action that:
1. Runs `pytest tests/`
2. On success, force-pushes to the HF Space remote

Secrets required: `HF_TOKEN` in GitHub repo settings, `GEMINI_API_KEY` + `APP_PASSWORD` in HF Space settings.
