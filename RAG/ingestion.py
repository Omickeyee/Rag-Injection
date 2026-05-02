from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from llama_index.core import Document, Settings as LlamaSettings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from settings import settings
from RAG.embeddings import get_embedding_model
from RAG.vector_store import get_vector_store

_METADATA_KEYS = [
    "id",
    "source_type",
    "author",
    "department",
    "access_level",
    "trust_score",
    "created_at",
]

def load_corpus(corpus_path = None):
    path = corpus_path or (settings.data_output_dir / "corpus.json")
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)
    documents = []
    for record in records:
        metadata = {}
        for key in _METADATA_KEYS:
            if key in record:
                metadata[key] = record[key]
        if "title" in record:
            metadata["title"] = record["title"]
        if "metadata" in record and isinstance(record["metadata"], dict):
            for mk, mv in record["metadata"].items():
                if isinstance(mv, (str, int, float, type(None))):
                    metadata[f"extra_{mk}"] = mv
                elif isinstance(mv, list):
                    metadata[f"extra_{mk}"] = ",".join(str(x) for x in mv)
        text = record.get("content", "")
        doc = Document(text=text, metadata=metadata)
        if "id" in record:
            doc.doc_id = record["id"]
        documents.append(doc)
    return documents

def get_sentence_splitter() -> SentenceSplitter:
    return SentenceSplitter(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)

def build_index(documents = None, corpus_path = None):
    if documents is None:
        documents = load_corpus(corpus_path)
    embed_model = get_embedding_model()
    LlamaSettings.embed_model = embed_model
    LlamaSettings.chunk_size = settings.chunk_size
    LlamaSettings.chunk_overlap = settings.chunk_overlap
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

def load_existing_index():
    embed_model = get_embedding_model()
    LlamaSettings.embed_model = embed_model
    vector_store = get_vector_store()
    return VectorStoreIndex.from_vector_store(vector_store)
