from pathlib import Path
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from config.settings import settings

_COLLECTION_NAME = "enterprise_docs"

def get_chroma_client(persist_dir = None):
    path = str(persist_dir or settings.chroma_persist_dir)
    return chromadb.PersistentClient(path=path)

def get_vector_store(collection_name = _COLLECTION_NAME, persist_dir = None):
    client = get_chroma_client(persist_dir)
    collection = client.get_or_create_collection(name=collection_name)
    return ChromaVectorStore(chroma_collection=collection)
