"""CLI: Manual retrieval testing — zobrazí presný kontext posielaný do LLM."""
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.generation.prompts import format_context, _chunk_label
from src.retrieval.lancedb_store import HybridRetriever

_DB_PATH = str(Path(__file__).parent.parent / "artifacts" / ".lance")


def main() -> int:
    if len(sys.argv) < 2:
        print('Usage: python scripts/query.py "<otázka>"')
        return 1

    query = " ".join(sys.argv[1:])
    retriever = HybridRetriever(_DB_PATH)
    results = retriever.search(query, k=5)

    # Skóre pre debug
    print(f"Query: {query!r}\n")
    print("── Retrieval skóre ──────────────────────────────────")
    for i, r in enumerate(results, start=1):
        rrf = r.get("_rrf_score") or 0.0
        dense = r.get("_dense_score") or 0.0
        sparse = r.get("_sparse_score")
        sparse_str = f"{sparse:.3f}" if sparse is not None else "N/A"
        print(f"[Z{i}] {r['id']}  rrf={rrf:.5f}  dense={dense:.4f}  sparse={sparse_str}")
        print(f"      {_chunk_label(r)}")

    # Presný kontext posielaný do LLM
    print("\n── LLM kontext (format_context) ─────────────────────")
    print(format_context(results))

    return 0


if __name__ == "__main__":
    sys.exit(main())
