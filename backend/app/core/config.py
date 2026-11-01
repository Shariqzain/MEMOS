"""
MEMOS — Centralized Configuration
Loads and validates all environment variables using Pydantic Settings.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import List, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All configurable parameters for MEMOS."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ──────────────────────────────────────────
    app_env: Literal["development", "production", "test"] = "development"
    log_level: str = "INFO"
    backend_port: int = 8001
    frontend_url: str = "http://localhost:5173"
    cors_origins: str = "http://localhost:5173,http://localhost:3000"

    # ── MongoDB ───────────────────────────────────────────────
    mongo_uri: str = "mongodb://localhost:27017"
    mongo_db_name: str = "memos"

    # ── ChromaDB ──────────────────────────────────────────────
    chroma_mode: Literal["embedded", "http"] = "embedded"
    chroma_persist_path: str = "./data/chroma"
    chroma_host: str = "localhost"
    chroma_port: int = 8001

    # ── Neo4j ─────────────────────────────────────────────────
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "memos_password"
    neo4j_database: str = "neo4j"
    neo4j_enabled: bool = False

    # ── LLM ───────────────────────────────────────────────────
    llm_provider: Literal["ollama", "openai", "gemini", "anthropic"] = "ollama"
    llm_model: str = "llama3.2"
    llm_base_url: str = "http://localhost:11434"
    llm_api_key: str = ""

    # ── Embedding ─────────────────────────────────────────────
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_provider: Literal["sentence_transformers", "ollama"] = "sentence_transformers"

    # ── Retrieval ─────────────────────────────────────────────
    semantic_retrieval_limit: int = 20
    graph_retrieval_limit: int = 15
    final_context_limit: int = 8
    similarity_threshold: float = 0.20
    rrf_k: int = 60

    # ── Memory Retention ─────────────────────────────────────
    min_importance_score: float = 0.35
    min_confidence_score: float = 0.40
    memory_dedup_threshold: float = 0.92

    # ── Demo User ─────────────────────────────────────────────
    demo_user_id: str = "demo_user"
    demo_user_name: str = "Demo User"

    @property
    def cors_origins_list(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache()
def get_settings() -> Settings:
    """Return cached Settings singleton."""
    return Settings()
