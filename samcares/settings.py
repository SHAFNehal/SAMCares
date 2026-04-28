"""Runtime configuration for SAMCares."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    """Application settings loaded from environment variables."""

    model_id: str = os.getenv("SAMCARES_MODEL_ID", "Qwen/Qwen3-8B")
    embedding_model_id: str = os.getenv(
        "SAMCARES_EMBEDDING_MODEL_ID",
        "sentence-transformers/all-MiniLM-L6-v2",
    )
    vector_store_path: Path = Path(
        os.getenv("SAMCARES_VECTOR_STORE_PATH", PROJECT_ROOT / "vector_data")
    )
    hf_token: str | None = os.getenv("HF_TOKEN")
    max_new_tokens: int = int(os.getenv("SAMCARES_MAX_NEW_TOKENS", "1024"))
    temperature: float = float(os.getenv("SAMCARES_TEMPERATURE", "0.7"))
    top_p: float = float(os.getenv("SAMCARES_TOP_P", "0.8"))
    top_k: int = int(os.getenv("SAMCARES_TOP_K", "20"))
    retriever_k: int = int(os.getenv("SAMCARES_RETRIEVER_K", "6"))


settings = Settings()

