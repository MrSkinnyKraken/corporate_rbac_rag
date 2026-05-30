"""FastAPI surface for the Zero-Trust RAG demo.

Exposes the application orchestrators behind a small HTTP API tailored
to the Streamlit frontend:

  * ``GET  /health``                        - liveness + dependency probe
  * ``GET  /users``                         - list demo identities
  * ``GET  /users/{user_id}``               - resolve one identity
  * ``POST /ingestion/propose``             - upload + LLM-classify
  * ``POST /ingestion/commit``              - persist after HITL edit
  * ``POST /query``                         - end-to-end answer pipeline

Identity is carried via the ``X-User-Id`` header (see
:func:`~api.dependencies.resolve_user`). CORS is open to all origins
because the deployment target is a local-only demo; tighten before
exposing the api beyond ``localhost``.

The handlers stay *thin*: parse + validate → delegate to an orchestrator
→ map the result through :mod:`api.schemas`. Every exception that
descends from :class:`~core.exceptions.ZeroTrustBaseError` is
intercepted by :func:`~api.errors.register_exception_handlers` and
emitted as a uniform JSON error body.
"""

from __future__ import annotations

import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, AsyncIterator

from fastapi import (
    Depends,
    FastAPI,
    File,
    HTTPException,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware

from api.dependencies import (
    get_app_settings,
    get_ingestion_app,
    get_llm,
    get_query_app,
    get_shared_embedder,
    get_user_store,
    get_vector_db,
    resolve_user,
)
from api.errors import register_exception_handlers
from api.schemas import (
    CommitRequest,
    HealthResponse,
    IngestionProposalSchema,
    IngestionReportSchema,
    QueryRequest,
    QueryResponseSchema,
    UserSchema,
)
from application.ingestion_app import IngestionApp
from application.query_app import QueryApp
from core.config import Settings
from core.exceptions import DocumentProcessingError, IngestionError
from domain.chunking.parsers import DocumentParser
from domain.users import User
from infrastructure.embedder import Embedder
from infrastructure.llm_client import OllamaClient
from infrastructure.user_store import UserStore
from infrastructure.vector_db import ChromaDBClient


_SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({
    ".txt", ".md", ".csv", ".pdf", ".docx", ".xlsx", ".pptx", ".html", ".htm",
})


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Touch the user store at boot so a broken fixture fails fast.

    Other heavy resources (embedder, LLM client) are loaded lazily on
    first use — the ~2s embedder cold start is amortised by the
    process-wide singleton and only paid once.
    """
    _ = get_user_store()           # raises ConfigurationError if the JSON is broken
    yield


# ─────────────────────────────────────────────────────────────────────────────
# App factory
# ─────────────────────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    """Build and configure the FastAPI app."""
    app = FastAPI(
        title="Zero-Trust RAG demo API",
        version="0.1.0",
        description=(
            "HTTP surface for the Zero-Trust RAG demo: identity, "
            "ingestion (two-phase HITL), and query answering."
        ),
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],        # tighten before any non-local deployment
        allow_methods=["*"],
        allow_headers=["*"],
    )
    register_exception_handlers(app)
    _register_routes(app)
    return app


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

def _register_routes(app: FastAPI) -> None:

    # ── Health ─────────────────────────────────────────────────────────────
    @app.get("/health", response_model=HealthResponse, tags=["meta"])
    def health(
        settings: Annotated[Settings, Depends(get_app_settings)],
        vector_db: Annotated[ChromaDBClient, Depends(get_vector_db)],
        llm: Annotated[OllamaClient, Depends(get_llm)],
        embedder: Annotated[Embedder, Depends(get_shared_embedder)],
    ) -> HealthResponse:
        """Liveness + dependency probe.

        Never raises; always returns 200 with the status of each backend
        so the Streamlit frontend can degrade gracefully ("Ollama is
        down, queries disabled").
        """
        chroma_ok = vector_db.health_check()
        ollama_ok = llm.health_check()
        overall = "ok" if (chroma_ok and ollama_ok) else "degraded"
        return HealthResponse(
            status=overall,
            chroma="up" if chroma_ok else "down",
            ollama="up" if ollama_ok else "down",
            embedder_loaded=embedder.is_loaded(),
            embedding_model=settings.embedding_model_name,
            llm_model=settings.ollama_model,
        )

    # ── Users ──────────────────────────────────────────────────────────────
    @app.get("/users", response_model=list[UserSchema], tags=["users"])
    def list_users(
        store: Annotated[UserStore, Depends(get_user_store)],
    ) -> list[UserSchema]:
        """Return every demo identity for the Streamlit user-picker."""
        return [UserSchema.from_domain(u) for u in store.list_all()]

    @app.get("/users/{user_id}", response_model=UserSchema, tags=["users"])
    def get_user(
        user_id: str,
        store: Annotated[UserStore, Depends(get_user_store)],
    ) -> UserSchema:
        """Resolve a single user by id.

        Raises:
            UserNotFoundError → 404 via the exception handler.
        """
        return UserSchema.from_domain(store.get(user_id))

    # ── Ingestion ──────────────────────────────────────────────────────────
    @app.post(
        "/ingestion/propose",
        response_model=IngestionProposalSchema,
        tags=["ingestion"],
    )
    def ingestion_propose(
        file: Annotated[UploadFile, File(description="Document to ingest.")],
        user: Annotated[User, Depends(resolve_user)],
        app_ingestion: Annotated[IngestionApp, Depends(get_ingestion_app)],
    ) -> IngestionProposalSchema:
        """Phase A: parse + LLM-classify the uploaded file.

        The frontend persists the returned :class:`IngestionProposalSchema`
        in its session and resends it (possibly with edited metadata)
        to ``POST /ingestion/commit``.
        """
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Upload must include a filename.",
            )
        ext = Path(file.filename).suffix.lower()
        if ext not in _SUPPORTED_EXTENSIONS:
            raise DocumentProcessingError(
                f"Unsupported file extension {ext!r}. Supported: "
                f"{sorted(_SUPPORTED_EXTENSIONS)}.",
                source=file.filename,
            )

        # Stream the upload to a temp DIRECTORY using the ORIGINAL
        # filename. The parser dispatches on suffix, and the orchestrator
        # propagates `path.name` to the IngestionProposal.source_file —
        # so preserving the original name keeps audit logs and the UI
        # citations human-readable. We do NOT persist to
        # `settings.raw_docs_dir` yet: the LLM proposal might be
        # rejected at the HITL step, in which case there is no reason
        # to keep the upload on disk.
        tmp_dir = Path(tempfile.mkdtemp(prefix="ingest_"))
        tmp_path = tmp_dir / Path(file.filename).name
        with tmp_path.open("wb") as fp:
            fp.write(file.file.read())

        try:
            proposal = app_ingestion.propose(tmp_path, user)
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
                tmp_dir.rmdir()
            except OSError:
                pass  # best-effort cleanup; not worth raising

        return IngestionProposalSchema.from_domain(proposal)

    @app.post(
        "/ingestion/commit",
        response_model=IngestionReportSchema,
        tags=["ingestion"],
    )
    def ingestion_commit(
        body: CommitRequest,
        user: Annotated[User, Depends(resolve_user)],
        app_ingestion: Annotated[IngestionApp, Depends(get_ingestion_app)],
    ) -> IngestionReportSchema:
        """Phase B: persist the proposal under the user-confirmed metadata.

        Performs the uploader cross-check internally (the
        :meth:`IngestionApp.commit` raises :class:`IngestionError` if
        the ``X-User-Id`` differs from
        ``proposal.uploader_user_id``).
        """
        proposal_domain = body.proposal.to_domain()
        final_metadata_domain = body.final_metadata.to_domain()
        report = app_ingestion.commit(proposal_domain, final_metadata_domain, user)
        return IngestionReportSchema.from_domain(report)

    # ── Query ──────────────────────────────────────────────────────────────
    @app.post("/query", response_model=QueryResponseSchema, tags=["query"])
    def query_answer(
        body: QueryRequest,
        user: Annotated[User, Depends(resolve_user)],
        app_query: Annotated[QueryApp, Depends(get_query_app)],
    ) -> QueryResponseSchema:
        """Route → retrieve → LLM → answer pipeline.

        Refusals (``refused=True``) are returned with HTTP 200 and a
        populated :class:`QueryResponseSchema.refusal_reason`. The
        frontend chooses how to render them — they are not error
        responses.
        """
        response = app_query.answer(body.query, user)
        return QueryResponseSchema.from_domain(response)


# ─────────────────────────────────────────────────────────────────────────────
# Module-level app instance for uvicorn
# ─────────────────────────────────────────────────────────────────────────────

app = create_app()
