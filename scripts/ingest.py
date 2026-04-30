#!/usr/bin/env python3
"""Ingest corpus.json into ChromaDB and verify with a test query.

Usage::

    python scripts/ingest.py
    python scripts/ingest.py --corpus data/generated/corpus.json
    python scripts/ingest.py --test-query "What is the company vacation policy?"
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path so `config` and `src` are importable.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings
from src.rag.ingestion import load_corpus, build_index, load_existing_index
from src.rag.retriever import EnterpriseRetriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest corpus into ChromaDB.")
    parser.add_argument(
        "--corpus",
        type=Path,
        default=None,
        help="Path to corpus.json (default: data/generated/corpus.json)",
    )
    parser.add_argument(
        "--test-query",
        type=str,
        default="What is the company vacation policy?",
        help="Test query to run after ingestion.",
    )
    args = parser.parse_args()

    corpus_path = args.corpus or (settings.data_output_dir / "corpus.json")

    # Ensure output directories exist.
    settings.ensure_dirs()

    # --- Load documents ---
    print(f"Loading corpus from {corpus_path} ...")
    documents = load_corpus(corpus_path)
    print(f"  Loaded {len(documents)} documents.")

    # --- Build index (embed + store in ChromaDB) ---
    print("Building index (chunking, embedding, storing in ChromaDB) ...")
    t0 = time.perf_counter()
    index = build_index(documents=documents)
    elapsed = time.perf_counter() - t0
    print(f"  Index built in {elapsed:.1f}s.")

    # --- Stats ---
    # Retrieve the vector store to report chunk count.
    from src.rag.vector_store import get_chroma_client

    client = get_chroma_client()
    collection = client.get_or_create_collection("enterprise_docs")
    num_chunks = collection.count()
    print(f"  ChromaDB collection 'enterprise_docs' contains {num_chunks} chunks.")

    # --- Test query ---
    print(f"\nRunning test query: \"{args.test_query}\"")
    retriever = EnterpriseRetriever(index)
    results = retriever.retrieve(args.test_query)
    print(f"  Retrieved {len(results)} nodes:\n")
    for i, nws in enumerate(results, start=1):
        meta = nws.node.metadata or {}
        score = f"{nws.score:.4f}" if nws.score is not None else "N/A"
        snippet = nws.node.get_content()[:120].replace("\n", " ")
        print(f"  [{i}] score={score} | source={meta.get('source_type', '?')} "
              f"| title={meta.get('title', '?')}")
        print(f"      {snippet}...")
        print()

    print("Ingestion complete.")


if __name__ == "__main__":
    main()
