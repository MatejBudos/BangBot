"""Streamlit UI pre BangRag — auth, rate limiting, streaming QA, logging."""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import streamlit as st

from src.generation.agent import BangAgent
from src.generation.openai_client import LLMUnavailable, OpenAIClient
from src.retrieval.lancedb_store import HybridRetriever

_DB_PATH = "artifacts/.lance"
_STATE_FILE = "state.json"
_LOG_FILE = "logs/queries.jsonl"
_MAX_SESSION_QUERIES = 30
_MAX_IP_PER_HOUR = 15
_MAX_DAILY_LLM = 1200

_CATEGORY_COLORS: dict[str, str] = {
    "hneda": "orange",
    "modra": "blue",
    "zelena": "green",
    "postava": "gray",
    "general_rules": "violet",
    "fistful": "red",
    "highnoon": "red",
    "wildwest": "red",
}

# Persists across Streamlit reruns within the same process (per-IP sliding window)
_ip_requests: dict[str, list[float]] = defaultdict(list)


def _build_index() -> None:
    """Build LanceDB index from corpus — runs only on first startup when index is absent."""
    from src.embedding.embedder import Embedder
    from src.ingestion.latex_parser import parse_corpus, write_chunks_jsonl
    from src.retrieval.lancedb_store import build_table

    data_dir = Path("data/corpus")
    out_dir = Path("artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)

    chunks = parse_corpus(data_dir)
    write_chunks_jsonl(chunks, out_dir / "chunks.jsonl")

    embedder = Embedder()
    texts = [
        f"{c.get('name_sk') or c.get('section_title') or ''}: {c['text']}".lstrip(": ")
        for c in chunks
    ]
    embeddings = embedder.embed_passages(texts)
    build_table(chunks, embeddings, str(out_dir / ".lance"))


@st.cache_resource(show_spinner="Inicializujem znalostný základ (prvý štart ~3–5 min)…")
def _get_retriever() -> HybridRetriever:
    if not Path(_DB_PATH).exists():
        _build_index()
    return HybridRetriever(_DB_PATH)


@st.cache_resource
def _get_client() -> OpenAIClient | None:
    try:
        return OpenAIClient()
    except ValueError:
        return None


# ── Global daily LLM counter (state.json) ─────────────────────────────────────

def _load_state() -> dict:
    today = datetime.date.today().isoformat()
    p = Path(_STATE_FILE)
    if p.exists():
        try:
            s = json.loads(p.read_text(encoding="utf-8"))
            if s.get("date") == today:
                return s
        except Exception:
            pass
    return {"date": today, "count": 0, "llm_disabled": False}


def _save_state(state: dict) -> None:
    Path(_STATE_FILE).write_text(json.dumps(state), encoding="utf-8")


def _increment_llm(state: dict) -> dict:
    state["count"] += 1
    if state["count"] >= _MAX_DAILY_LLM:
        state["llm_disabled"] = True
    _save_state(state)
    return state


# ── IP extraction + rate limiting ─────────────────────────────────────────────

def _get_client_ip() -> str:
    try:
        forwarded = st.context.headers.get("X-Forwarded-For", "")
        if forwarded:
            return forwarded.split(",")[0].strip()
    except Exception:
        pass
    return "local"


def _hash_ip(ip: str) -> str:
    return hashlib.sha256(ip.encode()).hexdigest()[:12]


def _within_ip_limit(ip: str) -> bool:
    now = time.time()
    cutoff = now - 3600
    _ip_requests[ip] = [t for t in _ip_requests[ip] if t > cutoff]
    return len(_ip_requests[ip]) < _MAX_IP_PER_HOUR


def _record_ip(ip: str) -> None:
    _ip_requests[ip].append(time.time())


# ── Logging (queries.jsonl) ────────────────────────────────────────────────────

def _queries_today() -> int:
    p = Path(_LOG_FILE)
    if not p.exists():
        return 0
    today = datetime.date.today().isoformat()
    try:
        return sum(
            1
            for line in p.read_text(encoding="utf-8").splitlines()
            if line.strip() and json.loads(line).get("ts", "").startswith(today)
        )
    except Exception:
        return 0


def _log_query(
    ip_hash: str,
    query: str,
    chunks: list[dict],
    llm_used: bool,
    latency_ms: float,
    tokens_used: int | None,
    agent_steps: int = 0,
) -> None:
    Path(_LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": datetime.datetime.utcnow().isoformat(),
        "ip_hash": ip_hash,
        "query": query,
        "retrieved_ids": [c.get("id", "") for c in chunks],
        "rrf_scores": [round(c.get("_rrf_score") or 0.0, 6) for c in chunks],
        "llm_used": llm_used,
        "latency_ms": round(latency_ms, 1),
        "tokens_used": tokens_used,
        "agent_steps": agent_steps,
    }
    with open(_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(json.dumps(entry, ensure_ascii=False), flush=True)


# ── UI helpers ─────────────────────────────────────────────────────────────────

def _badge(category: str) -> str:
    color = _CATEGORY_COLORS.get(category, "gray")
    return f":{color}[{category}]"


def _render_chunk(chunk: dict, idx: int, show_scores: bool) -> None:
    name = chunk.get("sk_name") or chunk.get("caption_name") or "Zdroj"
    cat = chunk.get("category")
    st.markdown(f"**[Z{idx}] {name}** {_badge(cat)}")
    if show_scores:
        dense = chunk.get("_dense_score") or 0.0
        sparse = chunk.get("_sparse_score") or 0.0
        rrf = chunk.get("_rrf_score") or 0.0
        st.caption(f"dense={dense:.4f}  sparse={sparse:.3f}  rrf={rrf:.5f}")
    st.text(chunk.get("text", ""))


# ── Auth gate ──────────────────────────────────────────────────────────────────

def _password_gate() -> bool:
    """Returns True when session is authenticated."""
    if st.session_state.get("authed"):
        return True
    expected = os.environ.get("APP_PASSWORD", "")
    with st.sidebar:
        pwd = st.text_input("Heslo pre prístup", type="password", key="_pwd")
    if pwd:
        if pwd == expected:
            st.session_state["authed"] = True
            st.rerun()
        else:
            st.sidebar.error("Nesprávne heslo.")
    st.info("Zadaj heslo v bočnom paneli pre prístup k aplikácii.")
    return False


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(page_title="BangRag", layout="wide")

    if not _password_gate():
        return

    if "query_count" not in st.session_state:
        st.session_state["query_count"] = 0

    global_state = _load_state()

    # Sidebar
    with st.sidebar:
        st.title("BangRag")
        st.divider()
        st.metric("Queries dnes", _queries_today())
        budget_left = max(0, _MAX_DAILY_LLM - global_state["count"])
        st.metric("LLM budget (zostatok)", budget_left)
        st.divider()
        show_scores = st.toggle("Zobraziť skóre", value=True)

    # Main
    st.title("Spýtaj sa na pravidlá Bang!")

    with st.form("query_form"):
        query = st.text_input("Otázka", placeholder="Čo robí karta Pivo?")
        submitted = st.form_submit_button("Hľadať", type="primary")

    if not submitted or not query.strip():
        return

    # Per-session rate limit
    if st.session_state["query_count"] >= _MAX_SESSION_QUERIES:
        st.error(f"Dosiahol si limit {_MAX_SESSION_QUERIES} otázok v tejto relácii.")
        return

    # Per-IP rate limit
    ip = _get_client_ip()
    if not _within_ip_limit(ip):
        st.error("Príliš veľa požiadaviek z tvojej IP adresy. Skús znova neskôr.")
        return

    _record_ip(ip)
    ip_hash = _hash_ip(ip)
    st.session_state["query_count"] += 1

    retriever = _get_retriever()
    client = _get_client()
    t0 = time.monotonic()

    chunks: list[dict] = []
    llm_used = False
    tokens_used: int | None = None
    agent_steps = 0
    llm_disabled = global_state.get("llm_disabled", False)

    if llm_disabled or client is None:
        with st.spinner("Hľadám v pravidlách..."):
            chunks = retriever.search(query.strip(), k=5)
        st.warning("Dnešný limit LLM bol vyčerpaný. Tu sú nájdené pravidlá:")
    else:
        agent = BangAgent(retriever)
        try:
            with st.status("Agent prehľadáva pravidlá...", expanded=True) as status:
                chunks = agent.collect_context(query.strip())
                agent_steps = len(agent.tool_calls_log)
                for step in agent.tool_calls_log:
                    status.write(
                        f"Hľadám: \"{step.query}\" "
                        f"[{step.variant}] → {step.n_new} nových výsledkov"
                    )
                status.update(label="Hotovo.", state="complete")
            st.write_stream(client.stream_answer(query.strip(), chunks))
            llm_used = True
            tokens_used = client.last_token_count
            global_state = _increment_llm(global_state)
        except LLMUnavailable:
            if not chunks:
                chunks = retriever.search(query.strip(), k=5)
            st.warning("Dnešný limit LLM bol vyčerpaný. Tu sú nájdené pravidlá:")

    # Sources expander
    with st.expander(f"Zdroje ({len(chunks)})"):
        for i, chunk in enumerate(chunks, start=1):
            _render_chunk(chunk, i, show_scores)
            if i < len(chunks):
                st.divider()

    total_ms = (time.monotonic() - t0) * 1000
    _log_query(ip_hash, query.strip(), chunks, llm_used, total_ms, tokens_used, agent_steps)
