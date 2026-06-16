"""CLI: Build LanceDB index from .tex corpus (parse → embed → store)."""
import argparse
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.latex_parser import parse_corpus, write_chunks_jsonl
from src.embedding.embedder import Embedder
from src.retrieval.lancedb_store import build_table


def main() -> int:
    parser = argparse.ArgumentParser(description="Build BangRag LanceDB index")
    parser.add_argument("--data", default="data/corpus", help="Path to .tex corpus dir")
    parser.add_argument("--out", default="artifacts", help="Output directory")
    args = parser.parse_args()

    data_dir = Path(args.data)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()

    print("Parsing corpus...")
    chunks = parse_corpus(data_dir)
    counts = Counter(c["type"] for c in chunks)
    print(f"  {len(chunks)} chunks: {dict(counts)}")

    jsonl_path = out_dir / "chunks.jsonl"
    write_chunks_jsonl(chunks, jsonl_path)
    print(f"  Written {jsonl_path}")

    print("Embedding...")
    embedder = Embedder()
    # Prepend name/title to text so dense embeddings capture card identity
    texts = [
        f"{c.get('name_sk') or c.get('section_title') or ''}: {c['text']}".lstrip(": ")
        for c in chunks
    ]
    embeddings = embedder.embed_passages(texts)
    print(f"  Embedded {len(texts)} passages, shape {embeddings.shape}, dtype {embeddings.dtype}")

    print("Building LanceDB table...")
    db_path = str(out_dir / ".lance")
    build_table(chunks, embeddings, db_path)
    print(f"  Table written to {db_path}")

    elapsed = time.monotonic() - t0
    print(f"Done in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
