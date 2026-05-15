"""Infrastructure adapters: thin wrappers around external services.

These classes are *technical* adapters — they translate native client errors
into the project's :mod:`core.exceptions` hierarchy and read all connection
parameters from :class:`~core.config.Settings`. They MUST NOT contain any
business logic (no chunking, no RBAC validation, no prompt engineering).
That logic lives under :mod:`domain` and :mod:`application`.
"""

from infrastructure.embedder import Embedder, get_embedder
from infrastructure.lexical_db import BM25Client, IndexedChunk
from infrastructure.llm_client import BaseLLM, LLMResponse, OllamaClient
from infrastructure.user_store import JsonUserStore, UserStore
from infrastructure.vector_db import ChromaDBClient

__all__: list[str] = [
    "BaseLLM",
    "LLMResponse",
    "OllamaClient",
    "ChromaDBClient",
    "BM25Client",
    "IndexedChunk",
    "Embedder",
    "get_embedder",
    "UserStore",
    "JsonUserStore",
]
