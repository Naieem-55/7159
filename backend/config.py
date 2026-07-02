"""Application configuration, loaded from environment variables / .env file.

Keeping all settings in one place avoids scattering `os.environ` calls
throughout the codebase and makes it obvious what the app needs to run.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings:
    """Typed wrapper around environment variables."""

    # --- OpenAI ---
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")

    # --- Paths ---
    book_path: Path = BASE_DIR / "data" / "book.txt"
    rubric_path: Path = BASE_DIR / "data" / "rubric.md"
    assignments_dir: Path = BASE_DIR / "assignments"
    reports_dir: Path = BASE_DIR / "reports"
    vector_store_dir: Path = BASE_DIR / "vector_store"

    # --- Chunking / retrieval ---
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "700"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "120"))
    top_k: int = int(os.getenv("TOP_K", "4"))

    def validate(self) -> None:
        """Raise a clear error early if required secrets are missing."""
        if not self.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
            )


settings = Settings()
