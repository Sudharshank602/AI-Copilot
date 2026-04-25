"""
AI Personal Intelligence Copilot
Configuration Management Module

FREE API STACK:
  • Groq API       — Free tier, ultra-fast LLaMA-3 / Mixtral inference
                     Sign up: https://console.groq.com (no credit card needed)
  • HuggingFace    — Free local embeddings via sentence-transformers
                     No API key needed — runs 100% on your machine
  • FAISS          — Free, local vector search (Facebook AI)
  • SQLite         — Free, local database

ZERO paid services. ZERO credit cards. Runs on any laptop.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Literal
from functools import lru_cache


class Settings(BaseSettings):
    """
    Application-wide settings loaded from environment variables or .env file.
    """

    # ─── App Meta ────────────────────────────────────────────────────────────
    APP_NAME: str = "AI Personal Intelligence Copilot"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # ─── Groq LLM (FREE) ─────────────────────────────────────────────────────
    # Get your free key at: https://console.groq.com
    # Free tier: 14,400 requests/day, 30 req/min
    GROQ_API_KEY: str = Field(default="", description="Groq API Key (free at console.groq.com)")

    # Available FREE Groq models (2025):
    #   llama-3.3-70b-versatile   <- best quality, recommended
    #   llama-3.1-8b-instant      <- fastest
    #   mixtral-8x7b-32768        <- great for long context (32k tokens)
    #   gemma2-9b-it              <- Google Gemma 2
    GROQ_MODEL: str = "llama-3.3-70b-versatile"

    # ─── Embeddings — HuggingFace LOCAL (FREE, no key needed) ────────────────
    HF_EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    HF_CACHE_DIR: str = "./models/hf_cache"
    EMBEDDING_DIMENSION: int = 384

    # ─── Vector Database (FREE, local) ───────────────────────────────────────
    VECTOR_DB_TYPE: Literal["faiss", "chromadb"] = "faiss"
    FAISS_INDEX_PATH: str = "./data/faiss_index"
    CHROMA_DB_PATH: str = "./data/chroma_db"
    CHROMA_COLLECTION_NAME: str = "copilot_knowledge"

    # ─── SQLite Database (FREE, local) ───────────────────────────────────────
    SQLITE_DB_PATH: str = "./data/copilot.db"
    DATABASE_URL: str = "sqlite:///./data/copilot.db"

    # ─── Backend Server ──────────────────────────────────────────────────────
    BACKEND_HOST: str = "0.0.0.0"
    BACKEND_PORT: int = 8000
    API_BASE_URL: str = "http://localhost:8000"
    SECRET_KEY: str = "dev-secret-key-change-in-production"

    # ─── Memory & RAG ────────────────────────────────────────────────────────
    MAX_MEMORY_TOKENS: int = 4000
    MEMORY_WINDOW_SIZE: int = 10
    RAG_TOP_K: int = 5
    RAG_SIMILARITY_THRESHOLD: float = 0.35
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    # ─── Generation Parameters ───────────────────────────────────────────────
    LLM_TEMPERATURE: float = 0.7
    LLM_MAX_TOKENS: int = 2048

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
