from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    chroma_persist_dir = Path("./chroma_db")
    data_output_dir = Path("./data/generated")
    model_output_dir = Path("./models")
    data_seed_dir = Path("./data/seed")
    data_attacks_dir = Path("./data/attacks")
    ollama_base_url = "http://localhost:11434"
    ollama_model = "llama3.1:8b"
    embedding_model_name = "BAAI/bge-small-en-v1.5"
    embedding_dimension = 384
    reranker_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    chunk_size = 512
    chunk_overlap = 50
    top_k = 5
    trust_score_threshold = 0.6
    safety_score_threshold = 0.5
    detector_threshold = 0.7
    num_clean_docs = 500
    num_poisoned_per_type = 5
    reranker_relevance_weight = 0.5
    reranker_safety_weight = 0.3
    reranker_trust_weight = 0.2

    def ensure_dirs(self):
        self.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
        self.data_output_dir.mkdir(parents=True, exist_ok=True)
        self.model_output_dir.mkdir(parents=True, exist_ok=True)

settings = Settings()
