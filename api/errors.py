"""Exception → HTTP-status mappers for the FastAPI surface.

Every error our code raises descends from :class:`ZeroTrustBaseError`.
Each branch of that hierarchy maps to a single, deterministic HTTP
status code so the api layer never improvises: identity → 4xx,
infrastructure → 5xx, business pipeline → 4xx for input issues / 5xx
for orchestrator failures.

Handlers also produce a uniform JSON body so clients (the Streamlit
demo, future SDKs) can pattern-match without parsing free-form prose::

    {
      "error":   "<exception class name>",
      "detail":  "<human-readable message>",
      "context": { ... optional structured fields ... }
    }
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from core.exceptions import (
    ConfigurationError,
    DocumentProcessingError,
    InfrastructureError,
    IngestionError,
    LLMConnectionError,
    LLMGenerationError,
    LLMResponseFormatError,
    LexicalIndexError,
    MetadataValidationError,
    PipelineError,
    RoutingError,
    UnauthorizedAccessError,
    UserNotFoundError,
    VectorDBConnectionError,
    ZeroTrustBaseError,
)


def _error_body(
    exc: BaseException,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Uniform JSON shape for every error response."""
    return {
        "error": type(exc).__name__,
        "detail": str(exc),
        "context": context or {},
    }


def register_exception_handlers(app: FastAPI) -> None:
    """Wire every domain exception to its HTTP status."""

    # ── Identity / authorisation ────────────────────────────────────────────
    @app.exception_handler(UserNotFoundError)
    async def _user_not_found(_req: Request, exc: UserNotFoundError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content=_error_body(exc, context={"user_id": exc.user_id}),
        )

    @app.exception_handler(UnauthorizedAccessError)
    async def _unauthorised(_req: Request, exc: UnauthorizedAccessError) -> JSONResponse:
        ctx: dict[str, Any] = {"resource_id": exc.resource_id}
        if exc.user_clearance is not None:
            ctx["user_clearance"] = int(exc.user_clearance)
        if exc.required_clearance is not None:
            ctx["required_clearance"] = int(exc.required_clearance)
        if exc.user_department is not None:
            ctx["user_department"] = exc.user_department.value
        if exc.required_departments:
            ctx["required_departments"] = list(exc.required_departments)
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content=_error_body(exc, context=ctx),
        )

    # ── Input / metadata validation ─────────────────────────────────────────
    @app.exception_handler(MetadataValidationError)
    async def _metadata_invalid(_req: Request, exc: MetadataValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=_error_body(exc),
        )

    @app.exception_handler(DocumentProcessingError)
    async def _doc_invalid(_req: Request, exc: DocumentProcessingError) -> JSONResponse:
        ctx: dict[str, Any] = {}
        if exc.source is not None:
            ctx["source"] = exc.source
        if exc.details:
            ctx["details"] = exc.details
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=_error_body(exc, context=ctx),
        )

    @app.exception_handler(IngestionError)
    async def _ingest_failure(_req: Request, exc: IngestionError) -> JSONResponse:
        # Ingestion failures are usually input-related (bad LLM output that
        # the user can re-edit; uploader mismatch; user-edited metadata
        # that violates an invariant). 400 lets the UI surface them.
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=_error_body(exc),
        )

    @app.exception_handler(RoutingError)
    async def _routing_failure(_req: Request, exc: RoutingError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=_error_body(exc),
        )

    @app.exception_handler(PipelineError)
    async def _pipeline_failure(_req: Request, exc: PipelineError) -> JSONResponse:
        # Generic catch for the rest of the PipelineError tree (empty
        # query, invalid alpha, etc.).
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=_error_body(exc),
        )

    # ── External services unavailable / failing ─────────────────────────────
    @app.exception_handler(VectorDBConnectionError)
    async def _vector_unreachable(_req: Request, exc: VectorDBConnectionError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=_error_body(exc, context={"service": "chromadb"}),
        )

    @app.exception_handler(LexicalIndexError)
    async def _lexical_failure(_req: Request, exc: LexicalIndexError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=_error_body(exc, context={"service": "bm25"}),
        )

    @app.exception_handler(LLMConnectionError)
    async def _llm_unreachable(_req: Request, exc: LLMConnectionError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=_error_body(exc, context={"service": "ollama"}),
        )

    @app.exception_handler(LLMGenerationError)
    async def _llm_generation(_req: Request, exc: LLMGenerationError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content=_error_body(exc, context={"service": "ollama"}),
        )

    @app.exception_handler(LLMResponseFormatError)
    async def _llm_bad_format(_req: Request, exc: LLMResponseFormatError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content=_error_body(exc, context={"service": "ollama"}),
        )

    @app.exception_handler(InfrastructureError)
    async def _infra_fallback(_req: Request, exc: InfrastructureError) -> JSONResponse:
        # Fallback for any future InfrastructureError subclass.
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=_error_body(exc),
        )

    # ── Configuration ───────────────────────────────────────────────────────
    @app.exception_handler(ConfigurationError)
    async def _config_failure(_req: Request, exc: ConfigurationError) -> JSONResponse:
        # Misconfiguration is operator-facing; surface as 500 so monitoring
        # treats it as an incident rather than user error.
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=_error_body(exc, context={"hint": "Check Settings / .env"}),
        )

    # ── Root catch-all ──────────────────────────────────────────────────────
    @app.exception_handler(ZeroTrustBaseError)
    async def _zerotrust_fallback(_req: Request, exc: ZeroTrustBaseError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=_error_body(exc),
        )
