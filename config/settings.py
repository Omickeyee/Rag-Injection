"""Central configuration for the RAG Prompt Injection project.

All tunable parameters in one place, loaded from environment variables
and .env file via pydantic-settings.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Project-wide settings loaded from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Paths ---
    chroma_persist_dir: Path = Path("./chroma_db")
    data_output_dir: Path = Path("./data/generated")
    model_output_dir: Path = Path("./models")
    data_seed_dir: Path = Path("./data/seed")
    data_attacks_dir: Path = Path("./data/attacks")

    # --- Ollama LLM ---
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"

    # --- Embedding Model ---
    embedding_model_name: str = "BAAI/bge-small-en-v1.5"
    embedding_dimension: int = 384

    # --- Reranker Model ---
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- RAG Parameters ---
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k: int = 5

    # --- Defense Thresholds ---
    trust_score_threshold: float = 0.6
    safety_score_threshold: float = 0.5
    detector_threshold: float = 0.7

    # --- Data Generation Parameters ---
    num_clean_docs: int = 500
    num_poisoned_per_type: int = 5

    # --- Reranker Weights ---
    reranker_relevance_weight: float = 0.5
    reranker_safety_weight: float = 0.3
    reranker_trust_weight: float = 0.2

    def ensure_dirs(self) -> None:
        """Create output directories if they do not exist."""
        self.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
        self.data_output_dir.mkdir(parents=True, exist_ok=True)
        self.model_output_dir.mkdir(parents=True, exist_ok=True)


# Singleton instance for convenient imports
settings = Settings()
