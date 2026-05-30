"""FastAPI dependency-injection providers.

Every long-lived component (ChromaDB client, Ollama client, embedder,
chunker, router, retriever, orchestrators) is built once per process
via :func:`functools.lru_cache` and re-used across requests. The
providers translate this layer cake into FastAPI's ``Depends(...)``
mechanism so handlers can declare the orchestrator they need without
caring how it was wired.

``resolve_user`` is the api-layer half of the demo identity scheme:
the Streamlit frontend sends an ``X-User-Id`` header, the provider
resolves it against the :class:`JsonUserStore`, and the handler
receives a fully-typed :class:`~domain.users.User` value object. The
day OAuth lands, only this function changes — handlers and orchestrators
stay untouched.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from fastapi import Depends, Header

from application.document_classifier import LLMDocumentClassifier
from application.ingestion_app import IngestionApp
from application.query_app import QueryApp
from core.config import Settings, get_settings
from core.exceptions import UserNotFoundError
from domain.chunking.core_chunker import CustomRBACChunker
from domain.chunking.parsers import DocumentParser
from domain.retrieval.ensemble_retriever import AsymmetricEnsembleRetriever
from domain.routing.ksp_index import KSPRouterIndex
from domain.routing.router import HierarchicalRouter
from domain.users import User
from infrastructure.embedder import Embedder, get_embedder
from infrastructure.lexical_db import BM25Client
from infrastructure.llm_client import OllamaClient
from infrastructure.user_store import JsonUserStore, UserStore
from infrastructure.vector_db import ChromaDBClient


# ─────────────────────────────────────────────────────────────────────────────
# Infrastructure singletons
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_app_settings() -> Settings:
    """Return the application :class:`Settings` singleton."""
    return get_settings()


@lru_cache(maxsize=1)
def get_vector_db() -> ChromaDBClient:
    return ChromaDBClient()


@lru_cache(maxsize=1)
def get_lexical_db() -> BM25Client:
    return BM25Client()


@lru_cache(maxsize=1)
def get_llm() -> OllamaClient:
    return OllamaClient()


@lru_cache(maxsize=1)
def get_shared_embedder() -> Embedder:
    """Re-export the shared embedder under a FastAPI-friendly name."""
    return get_embedder()


@lru_cache(maxsize=1)
def get_user_store() -> UserStore:
    return JsonUserStore()


# ─────────────────────────────────────────────────────────────────────────────
# Domain singletons
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_ksp_index() -> KSPRouterIndex:
    return KSPRouterIndex(vector_db=get_vector_db(), embedder=get_shared_embedder())


@lru_cache(maxsize=1)
def get_router() -> HierarchicalRouter:
    return HierarchicalRouter(ksp_index=get_ksp_index())


@lru_cache(maxsize=1)
def get_retriever() -> AsymmetricEnsembleRetriever:
    return AsymmetricEnsembleRetriever(
        vector_db=get_vector_db(),
        lexical_db=get_lexical_db(),
        embedder=get_shared_embedder(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Application orchestrators
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_ingestion_app() -> IngestionApp:
    return IngestionApp(
        parser=DocumentParser(),
        classifier=LLMDocumentClassifier(get_llm()),
        chunker=CustomRBACChunker(),
        vector_db=get_vector_db(),
        lexical_db=get_lexical_db(),
        ksp_index=get_ksp_index(),
        embedder=get_shared_embedder(),
    )


@lru_cache(maxsize=1)
def get_query_app() -> QueryApp:
    return QueryApp(
        router=get_router(),
        retriever=get_retriever(),
        llm=get_llm(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Identity resolution
# ─────────────────────────────────────────────────────────────────────────────

def resolve_user(
    x_user_id: Annotated[str, Header(alias="X-User-Id")],
    store: Annotated[UserStore, Depends(get_user_store)],
) -> User:
    """Translate the ``X-User-Id`` header into a typed :class:`User`.

    The Streamlit demo sets this header from the user-picker dropdown.
    A missing header raises FastAPI's default 422; an unknown
    ``user_id`` propagates :class:`UserNotFoundError` which the
    api-layer exception handler converts to 404.

    Args:
        x_user_id: The ``X-User-Id`` request header (validated as a
            non-empty string by FastAPI).
        store:     The injected :class:`UserStore` singleton.

    Returns:
        A populated :class:`User`.

    Raises:
        UserNotFoundError: ``x_user_id`` is not in the fixture.
    """
    cleaned = x_user_id.strip()
    if not cleaned:
        # FastAPI's Header validator already rejects missing header, but
        # an explicit empty value (X-User-Id: "") slips through.
        raise UserNotFoundError(user_id="")
    return store.get(cleaned)
