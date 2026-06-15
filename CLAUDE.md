# BangRag — Claude Code Context

This file is auto-loaded into every Claude Code session in this repo.

## What this project is

Hybrid RAG nad slovenským korpusom pravidiel kartovej hry Bang!. Portfolio projekt + verejné demo na HF Spaces. Single-turn Q&A, žiadna multi-turn história.

**Read first:** `ARCHITECTURE.md` (architektúra + rozhodnutia) a `TASKS.md` (executable tasks s acceptance criteria).

## Stack (fixed — neuvažuj alternatívy bez explicitného súhlasu)

- Python 3.11
- `sentence-transformers` + `intfloat/multilingual-e5-base` (dense, lokálne)
- `lancedb` (native hybrid search + RRF, FTS na `text_lemmatized` stĺpci)
- `simplemma` (lang='sk') pre lemmatizáciu, custom diacritic strip + SK stopwords
- `google-genai` + `gemini-2.0-flash` (streaming)
- `streamlit` (UI, hostované na HF Spaces)
- `pytest`

## Konvencie

- **Slovak text in user-facing strings** (UI labels, error messages, system prompt, README), English in code identifiers
- **No `cd` in commands** — používaj absolute paths, working dir je `D:\Projects\BangRag`
- **Shell:** bash on Windows — Unix syntax (`/dev/null`, forward slashes)
- **Žiadne emoji** v kóde alebo dokumentácii pokiaľ to user explicitne nepýta
- **Žiadne komentáre vysvetľujúce WHAT** — len WHY pri non-obvious logic
- **Žiadne docstring-y** okrem 1-line description pri verejných funkciách
- **Žiadne backwards-compat hacks** — toto je greenfield projekt
- **Žiadne early abstractions** — pridaj abstrakciu až keď máš 3+ reálne use cases

## Anti-patterns (toto nerob)

- Vytvárať `BM25S` + manuálny RRF — používame **LanceDB native hybrid**
- Pridávať cross-encoder reranker vo v1
- Pridávať card images do UI (copyright risk)
- Pridávať `ragas` alebo iné eval frameworks — máme custom `src/eval/`
- Persistovať s SQLite, Postgres, alebo iné DB — len LanceDB + JSONL files
- Vytvárať Docker setup — HF Spaces buildí cez `requirements.txt`
- Refaktorovať `IParser.py` skeleton — implementuj `latex_parser.py` priamo

## Task workflow

1. Pri starte session: `TaskList` na pozretie čo je odblokované
2. Pred prácou na tasku: `TaskUpdate` status na `in_progress`
3. Po dokončení: spusti `pytest tests/` — musí prejsť
4. `TaskUpdate` na `completed` **iba ak všetky acceptance criteria sú splnené**
5. Ak narazíš na nejednoznačnosť → opýtaj sa, nehádaj

## Git workflow

- **Žiadny git push bez explicitnej požiadavky**
- Commit message po slovensky alebo anglicky, krátko, imperatívne ("pridaj parser pre karty")
- Žiadny `--no-verify`, žiadny `git reset --hard` bez súhlasu
- Žiadny `git add .` — pridávaj konkrétne súbory

## Repo layout

Viď `ARCHITECTURE.md` sekcia 8. Krátky reminder:
```
src/{ingestion,embedding,retrieval,generation,eval,app}/
scripts/{build_index,query,generate_eval}.py
tests/test_*.py
eval/qa.jsonl
artifacts/{chunks.jsonl,.lance/}
data/corpus/*.tex  # existing, neupravovať
app.py  # thin wrapper -> src/app/streamlit_app.py
```

## Acceptance criteria mantra

Každý task v `TASKS.md` má check-list. **Neoznačuj task ako completed kým všetky checkboxy nie sú splnené.** Ak niečo nestihneš, hold ho v `in_progress` a vytvor follow-up task.
