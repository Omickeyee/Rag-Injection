"""ChromaDB vector store setup for the RAG pipeline."""

from pathlib import Path

import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore

from config.settings import settings


_COLLECTION_NAME = "enterprise_docs"


def get_chroma_client(persist_dir: Path | None = None) -> chromadb.PersistentClient:
    """Return a persistent ChromaDB client.

    Parameters
    ----------
    persist_dir:
        Directory for ChromaDB storage.  Defaults to
        ``settings.chroma_persist_dir``.
    """
    path = str(persist_dir or settings.chroma_persist_dir)
    return chromadb.PersistentClient(path=path)


def get_vector_store(
    collection_name: str = _COLLECTION_NAME,
    persist_dir: Path | None = None,
) -> ChromaVectorStore:
    """Get or create a ChromaVectorStore backed by ChromaDB.

    Parameters
    ----------
    collection_name:
        Name of the ChromaDB collection.
    persist_dir:
        Directory for ChromaDB storage.  Defaults to
        ``settings.chroma_persist_dir``.

    Returns
    -------
    ChromaVectorStore
        A llama-index vector store wrapping the ChromaDB collection.
    """
    client = get_chroma_client(persist_dir)
    collection = client.get_or_create_collection(name=collection_name)
    return ChromaVectorStore(chroma_collection=collection)
