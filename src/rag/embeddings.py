"""HuggingFace embedding wrapper for the RAG pipeline."""

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from config.settings import settings


def get_embedding_model() -> HuggingFaceEmbedding:
    """Create and return a HuggingFace embedding model instance.

    Uses the model specified in ``settings.embedding_model_name``
    (default: ``BAAI/bge-small-en-v1.5``, 384-dimensional).
    """
    return HuggingFaceEmbedding(
        model_name=settings.embedding_model_name,
    )
