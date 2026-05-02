from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from settings import settings

def get_embedding_model():
    return HuggingFaceEmbedding(model_name=settings.embedding_model_name)
