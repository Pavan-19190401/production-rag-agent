import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

# Base Directory of Project
BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # API Keys
    OPENAI_API_KEY: str = Field(default="", description="OpenAI API Key")
    GEMINI_API_KEY: str = Field(default="", description="Google Gemini API Key")

    # LLM Settings
    LLM_PROVIDER: str = Field(default="auto", description="LLM provider: openai, gemini, or mock")
    LLM_MODEL: str = Field(default="gpt-4o-mini", description="Primary generation model")
    FALLBACK_LLM_MODEL: str = Field(default="gpt-3.5-turbo", description="Fallback generation model")
    LLM_TEMPERATURE: float = Field(default=0.1, description="Low temperature for factual grounding")

    # Embeddings & Reranking
    EMBEDDING_MODEL: str = Field(default="BAAI/bge-small-en-v1.5", description="Dense embedding model")
    RERANKER_MODEL: str = Field(default="BAAI/bge-reranker-base", description="Cross-encoder reranker model")

    # Vector DB (Qdrant)
    QDRANT_URL: str = Field(default=str(BASE_DIR / "qdrant_data"), description="Qdrant URL or local storage path")
    QDRANT_API_KEY: str = Field(default="", description="Qdrant Cloud API key")
    QDRANT_COLLECTION: str = Field(default="production_rag", description="Collection name")

    # Redis Cache
    REDIS_URL: str = Field(default="", description="Redis URL, empty for local in-memory cache")

    # Security & Budget
    API_KEY: str = Field(default="rag-prod-secret-key-change-me", description="Internal API key for client authentication")
    RATE_LIMIT_PER_MINUTE: int = Field(default=60, description="Max requests per minute per key")
    DAILY_BUDGET_USD: float = Field(default=5.00, description="Daily spending cap")
    MAX_TOKENS_PER_QUERY: int = Field(default=1024, description="Max output tokens per answer")

    # Observability
    LANGFUSE_PUBLIC_KEY: str = Field(default="", description="Langfuse public key")
    LANGFUSE_SECRET_KEY: str = Field(default="", description="Langfuse secret key")
    LANGFUSE_HOST: str = Field(default="https://cloud.langfuse.com", description="Langfuse host URL")

    # Chunking & Data
    CHUNK_SIZE: int = Field(default=512, description="Target chunk size in tokens/characters")
    CHUNK_OVERLAP: int = Field(default=50, description="Overlap between consecutive chunks")
    CHUNK_STRATEGY: str = Field(default="recursive", description="Strategy: fixed, recursive, or heading")
    DATA_RAW_DIR: str = Field(default=str(BASE_DIR / "data" / "raw"), description="Raw document directory")
    DATA_PROCESSED_DIR: str = Field(default=str(BASE_DIR / "data" / "processed"), description="Processed storage directory")

settings = Settings()
