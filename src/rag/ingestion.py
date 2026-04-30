"""Document loading, chunking, and indexing for the RAG pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llama_index.core import Document, Settings as LlamaSettings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter

from config.settings import settings
from src.rag.embeddings import get_embedding_model
from src.rag.vector_store import get_vector_store


# Metadata keys that every corpus document should carry.
_METADATA_KEYS = [
    "id",
    "source_type",
    "author",
    "department",
    "access_level",
    "trust_score",
    "created_at",
]


def load_corpus(corpus_path: Path | None = None) -> list[Document]:
    """Load ``corpus.json`` into LlamaIndex :class:`Document` objects.

    Each JSON record is converted into a ``Document`` with its full text as
    the ``text`` field and all standard metadata preserved so downstream
    retrieval and defenses can inspect it.

    Parameters
    ----------
    corpus_path:
        Path to the corpus JSON file.  Defaults to
        ``settings.data_output_dir / "corpus.json"``.
    """
    path = corpus_path or (settings.data_output_dir / "corpus.json")
    with open(path, "r", encoding="utf-8") as f:
        records: list[dict[str, Any]] = json.load(f)

    documents: list[Document] = []
    for record in records:
        # Build metadata dict from standard keys present in the record.
        metadata: dict[str, Any] = {}
        for key in _METADATA_KEYS:
            if key in record:
                metadata[key] = record[key]

        # Include title in metadata for source attribution.
        if "title" in record:
            metadata["title"] = record["title"]

        # Carry over any extra nested metadata the generators may have added.
        # ChromaDB requires flat (str/int/float/None) metadata values.
        if "metadata" in record and isinstance(record["metadata"], dict):
            for mk, mv in record["metadata"].items():
                if isinstance(mv, (str, int, float, type(None))):
                    metadata[f"extra_{mk}"] = mv
                elif isinstance(mv, list):
                    metadata[f"extra_{mk}"] = ",".join(str(x) for x in mv)

        text = record.get("content", "")
        doc = Document(text=text, metadata=metadata)

        # Use the record id as the document id for traceability.
        if "id" in record:
            doc.doc_id = record["id"]

        documents.append(doc)

    return documents


def get_sentence_splitter() -> SentenceSplitter:
    """Return a ``SentenceSplitter`` configured from project settings."""
    return SentenceSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )


def build_index(
    documents: list[Document] | None = None,
    corpus_path: Path | None = None,
) -> VectorStoreIndex:
    """Build (or rebuild) a :class:`VectorStoreIndex` from corpus documents.

    The full ingestion pipeline:
      1. Load JSON corpus into ``Document`` objects.
      2. Split into chunks via ``SentenceSplitter``.
      3. Embed with the project's HuggingFace model.
      4. Store in ChromaDB.

    Parameters
    ----------
    documents:
        Pre-loaded documents.  If ``None``, documents are loaded from
        *corpus_path*.
    corpus_path:
        Path to ``corpus.json``.  Only used when *documents* is ``None``.

    Returns
    -------
    VectorStoreIndex
        Ready-to-query index backed by ChromaDB.
    """
    if documents is None:
        documents = load_corpus(corpus_path)

    # Configure LlamaIndex global settings for this pipeline.
    embed_model = get_embedding_model()
    LlamaSettings.embed_model = embed_model
    LlamaSettings.chunk_size = settings.chunk_size
    LlamaSettings.chunk_overlap = settings.chunk_overlap

    # Prepare vector store backed storage context.
    vector_store = get_vector_store()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    splitter = get_sentence_splitter()

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        transformations=[splitter],
        show_progress=True,
    )

    return index


def load_existing_index() -> VectorStoreIndex:
    """Load an existing index from the persisted ChromaDB store.

    Use this when data has already been ingested and you just want to
    query the existing index without re-embedding.
    """
    embed_model = get_embedding_model()
    LlamaSettings.embed_model = embed_model

    vector_store = get_vector_store()
    return VectorStoreIndex.from_vector_store(vector_store)
