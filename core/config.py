"""Centralized application configuration.

This module is the single point of truth for every runtime parameter of the
Zero-Trust RAG system: ChromaDB connection, Ollama endpoint and model,
embedding-model identifier, filesystem paths, and FastAPI / Streamlit network
bindings.

Values are loaded once from the process environment (with fallback to a
local ``.env`` file) by `pydantic_settings.BaseSettings`. The rest of the
codebase MUST read configuration through :func:`get_settings` and never
inspect environment variables directly. This keeps the boundary between
infrastructure and business logic clean.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strongly-typed application configuration loaded from `.env`.

    Resolution order for every field:

      1. The corresponding process environment variable (case-insensitive).
      2. The matching key in a ``.env`` file at the working directory root.
      3. The default declared on the field itself.

    Unknown environment variables are silently ignored so private extras can
    coexist with the schema without raising validation errors.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ─────────────────────── ChromaDB (vector store) ─────────────────────────
    chroma_host: str = Field(
        default="localhost",
        description="Hostname of the ChromaDB HTTP service.",
    )
    chroma_port: int = Field(
        default=8001,
        ge=1,
        le=65535,
        description="TCP port of the ChromaDB HTTP service.",
    )
    chroma_ssl: bool = Field(
        default=False,
        description="Whether the ChromaDB endpoint uses HTTPS.",
    )
    chroma_tenant: str = Field(
        default="default_tenant",
        description="ChromaDB tenant for multi-tenancy isolation.",
    )
    chroma_database: str = Field(
        default="default_database",
        description="ChromaDB database within the configured tenant.",
    )

    # ─────────────────────── Ollama (local LLM server) ──────────────────────
    ollama_host: str = Field(
        default="localhost",
        description="Hostname of the Ollama server.",
    )
    ollama_port: int = Field(
        default=11434,
        ge=1,
        le=65535,
        description="TCP port of the Ollama server.",
    )
    ollama_model: str = Field(
        default="llama3.2",
        description="Identifier of the model used for LLM calls.",
    )
    ollama_request_timeout_s: int = Field(
        default=180,
        ge=1,
        description="Timeout (in seconds) applied to each Ollama HTTP call.",
    )
    ollama_num_ctx: int = Field(
        default=8192,
        ge=512,
        description="Context-window size sent to Ollama via the `num_ctx` option.",
    )

    # ─────────────────────── Embedding model ────────────────────────────────
    embedding_model_name: str = Field(
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        description=(
            "HuggingFace identifier of the sentence-encoder used to build "
            "ChromaDB vectors. Default is the multilingual MiniLM-L12 variant "
            "after the Phase 3 Step 4 Q4 finding (English query over a "
            "Spanish-language warranty clause): the English-only "
            "`all-MiniLM-L6-v2` failed to align cross-lingual paraphrases. "
            "Changing this value invalidates every existing ChromaDB "
            "collection — the runtime guard in `infrastructure.vector_db` "
            "raises ConfigurationError on mismatch so the corpus must be "
            "re-ingested before service can resume."
        ),
    )
    embedding_device: Literal["cpu", "cuda"] = Field(
        default="cpu",
        description="Torch device used by the embedding model.",
    )

    # ─────────────────────── Application paths ──────────────────────────────
    raw_docs_dir: Path = Field(
        default=Path("./data/raw_docs"),
        description="Directory where uploaded source documents are stored.",
    )
    bm25_index_dir: Path = Field(
        default=Path("./data/bm25_indexes"),
        description="Directory where the per-department BM25 indexes are persisted.",
    )
    artifacts_dir: Path = Field(
        default=Path("./data/artifacts"),
        description=(
            "Directory for cached pipeline artefacts: KSPs, parent-child indexes, "
            "intermediate document metadata, and any other on-disk state owned "
            "by the application."
        ),
    )
    users_fixture_path: Path = Field(
        default=Path("./data/demo_users.json"),
        description=(
            "Path to the JSON fixture that backs the demo `JsonUserStore`. "
            "The demo deliberately ships without an auth stack: the Streamlit "
            "frontend reads `list_all()` to populate a user-picker dropdown "
            "and the api adapter resolves the selected `user_id` to a `User` "
            "via `UserStore.get`. Replace with an OAuth-backed store in "
            "production by swapping the `UserStore` binding — domain code "
            "does not depend on the JSON shape."
        ),
    )

    # ─────────────────────── FastAPI backend ────────────────────────────────
    api_host: str = Field(
        default="127.0.0.1",
        description="Network interface FastAPI binds to.",
    )
    api_port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="TCP port for the FastAPI server.",
    )
    api_log_level: Literal["debug", "info", "warning", "error", "critical"] = Field(
        default="info",
        description="Uvicorn log level for the FastAPI process.",
    )

    # ─────────────────────── Streamlit frontend ─────────────────────────────
    frontend_host: str = Field(
        default="127.0.0.1",
        description="Network interface Streamlit binds to.",
    )
    frontend_port: int = Field(
        default=8501,
        ge=1,
        le=65535,
        description="TCP port for the Streamlit UI.",
    )

    # ─────────────────────── Application-wide ───────────────────────────────
    app_log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Default log level for application loggers.",
    )

    # ─────────────────────── Validators ─────────────────────────────────────
    @field_validator("raw_docs_dir", "bm25_index_dir", "artifacts_dir", mode="after")
    @classmethod
    def _ensure_directory_exists(cls, value: Path) -> Path:
        """Create the directory on first access and return its absolute path.

        Auto-creating the working directories at config-load time means any
        downstream module can assume they exist; it also surfaces filesystem
        permission errors immediately at startup rather than on first write.
        """
        value.mkdir(parents=True, exist_ok=True)
        return value.resolve()

    # ─────────────────────── Computed URLs ──────────────────────────────────
    @computed_field  # type: ignore[prop-decorator]
    @property
    def chroma_base_url(self) -> str:
        """The HTTP(S) base URL of the ChromaDB service."""
        scheme = "https" if self.chroma_ssl else "http"
        return f"{scheme}://{self.chroma_host}:{self.chroma_port}"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def ollama_base_url(self) -> str:
        """The HTTP base URL of the Ollama server."""
        return f"http://{self.ollama_host}:{self.ollama_port}"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def ollama_generate_url(self) -> str:
        """Full URL for the Ollama ``/api/generate`` endpoint."""
        return f"{self.ollama_base_url}/api/generate"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def ollama_tags_url(self) -> str:
        """Full URL for the Ollama ``/api/tags`` endpoint (model listing)."""
        return f"{self.ollama_base_url}/api/tags"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton :class:`Settings` instance for this process.

    The result is cached so configuration is loaded and validated exactly
    once. I'll use this example function as a FastAPI dependency to inject configuration
    into request handlers:

    .. code-block:: python

        from typing import Annotated
        from fastapi import Depends
        from core.config import Settings, get_settings

        async def handler(
            settings: Annotated[Settings, Depends(get_settings)],
        ) -> dict[str, str]:
            return {"model": settings.ollama_model}

    Returns:
        The validated :class:`Settings` object.
    """
    return Settings()  # type: ignore[call-arg]
