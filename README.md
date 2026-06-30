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

**Live demo:** *(HF Space — see deploy section)*

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
                          BangAgent (tool-calling loop)
                          ├── search_rules (sparse/dense/hybrid)
                          └── select_chunks (filter relevant)
                                 │
                         selected chunks
                                 │
                        OpenAI (model z config/agent.toml)
                                 │
                          streamed answer [Z1][Z2]
```

---

## Key decisions

- **`intfloat/multilingual-e5-base`** — strong Slovak performance without fine-tuning; fits in HF Spaces free CPU tier
- **LanceDB native hybrid + simplemma** — single dependency for vector + FTS; simplemma handles Slovak morphology (kartami → karta) without a full NLP pipeline
- **RRF fusion (k=60)** — simple, parameter-free combination of dense and sparse scores; outperforms weighted sum on short queries
- **Agentic retrieval (BangAgent)** — iterative tool-calling loop: agent decides what to search (sparse/dense/hybrid), reads full chunk text, then explicitly selects relevant chunks via `select_chunks`; better than one-shot retrieval for multi-entity queries
- **OpenAI tool-calling** — agent model + gen model konfigurovateľné v `config/agent.toml`; graceful fallback to retrieval-only when quota is exhausted
- **HF Spaces + Streamlit** — zero-cost hosting, no Docker needed

---

## Ablation results

| Variant | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---|---|---|---|
| Dense only | 0.40 | 0.59 | 0.65 | 0.50 |
| Sparse only | **0.88** | **0.93** | **0.95** | **0.91** |
| Hybrid (RRF) | 0.61 | 0.80 | 0.91 | 0.72 |

*Eval set: 360 retrieval queries. Sparse dominates on named card/character lookups; agent defaults to sparse.*

---

## Slovak-specific challenges

- **Morphology**: Slovak is highly inflected — "kartami", "kartách", "karte" all mean "card". Simplemma lemmatizes before FTS indexing so all forms match.
- **Diacritics**: Queries often omit accents ("co robi Pivo" vs "čo robí Pivo"). Diacritic stripping is applied to both index and query.
- **Code-switching**: Card names mix Slovak and Italian/English (Mancato, Birra, Gatling). The parser preserves `name_orig` from image filenames and `sk_name` from captions for cross-lingual matching.
- **Term disambiguation**: "ťahať kartu" (flip for check) ≠ "potiahnuť kartu" (draw to hand). Glossary terms are indexed as individual chunks so the agent can look them up.

---

## Local development

```bash
# 1. Clone and install
git clone https://github.com/<user>/BangRag.git
cd BangRag
pip install -r requirements.txt

# 2. Set secrets
cp .env.example .env
# fill in OPENAI_KEY and APP_PASSWORD

# 3. Build the index (downloads ~280 MB model on first run)
python scripts/build_index.py

# 4. Run the app
streamlit run app.py
```

---

## Deploy

Push to `main` triggers a GitHub Action that:
1. Runs `pytest tests/`
2. On success, force-pushes an orphan commit to the HF Space (no binary history)

Secrets required: `HF_TOKEN` in GitHub repo settings, `OPENAI_KEY` + `APP_PASSWORD` in HF Space settings.
