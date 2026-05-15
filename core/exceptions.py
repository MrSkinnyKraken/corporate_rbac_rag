"""Custom exception hierarchy for the Zero-Trust RAG system.

Every error raised by application code should derive from
:class:`ZeroTrustBaseError`, never from the broad built-ins. Catch sites can
then choose the right granularity:

* ``except ZeroTrustBaseError``       — any error this system raises.
* ``except SecurityError``            — RBAC / auth violations only.
* ``except InfrastructureError``      — external-service failures (DB, LLM…).
* ``except LLMError``                 — anything related to the LLM call chain.
* ``except VectorDBConnectionError``  — ChromaDB unreachable specifically.
There's probably more I'll be adding as the system construction proceeds.

The hierarchy is::

    ZeroTrustBaseError                  (root)
    ├── ConfigurationError
    ├── SecurityError
    │   ├── UnauthorizedAccessError
    │   ├── MetadataValidationError
    │   └── UserNotFoundError
    ├── InfrastructureError
    │   ├── VectorDBConnectionError
    │   ├── LexicalIndexError
    │   └── LLMError
    │       ├── LLMConnectionError
    │       ├── LLMGenerationError
    │       └── LLMResponseFormatError
    └── PipelineError
        ├── DocumentProcessingError
        ├── IngestionError
        └── RoutingError

All exceptions support standard ``raise NewError(...) from original`` chaining,
so the original cause is preserved in tracebacks without manual ``cause`` fields.
"""

from __future__ import annotations

from typing import Any

from core.security import ClearanceLevel, Department


# =============================================================================
# Root
# =============================================================================

class ZeroTrustBaseError(Exception):
    """Base class for every exception raised by the Zero-Trust RAG system.

    Sub-classes add semantic categorisation but never strip information; the
    underlying ``args`` and ``__cause__`` (set by ``raise ... from e``) are
    preserved for diagnostics and structured logging.
    """


# =============================================================================
# Configuration
# =============================================================================

class ConfigurationError(ZeroTrustBaseError):
    """Raised when application configuration is missing or inconsistent.

    Examples:
        - A required environment variable cannot be parsed.
        - A path declared in :class:`~core.config.Settings` cannot be created.
        - Two services are configured to listen on the same port.
    """


# =============================================================================
# Security branch
# =============================================================================

class SecurityError(ZeroTrustBaseError):
    """Parent of every RBAC / authorisation / metadata-validation error."""


class UnauthorizedAccessError(SecurityError):
    """Raised when a subject attempts to access a resource it cannot read.

    Carries the structured context required by the audit logger so a
    Zero-Trust violation can be traced to the specific user, resource, and
    rule that rejected it.

    Args:
        message: Human-readable summary of the violation.
        user_clearance: The clearance held by the requesting user.
        required_clearance: The minimum clearance the resource requires.
        user_department: The requesting user's department.
        required_departments: The set of departments allowed to read the
            resource (the value of the chunk's ``allowed_departments`` list,
            including possibly the ``"all"`` wildcard).
        resource_id: An opaque identifier for the rejected resource (e.g.
            ``"chunk:fin_q1_2026.txt#003"``).
    """

    def __init__(
        self,
        message: str = "Access denied by Zero-Trust policy.",
        *,
        user_clearance: ClearanceLevel | None = None,
        required_clearance: ClearanceLevel | None = None,
        user_department: Department | None = None,
        required_departments: list[str] | None = None,
        resource_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.user_clearance: ClearanceLevel | None = user_clearance
        self.required_clearance: ClearanceLevel | None = required_clearance
        self.user_department: Department | None = user_department
        self.required_departments: list[str] = required_departments or []
        self.resource_id: str | None = resource_id

    def __str__(self) -> str:  # pragma: no cover  (cosmetic)
        parts: list[str] = [super().__str__()]
        if self.user_clearance is not None and self.required_clearance is not None:
            parts.append(
                f"user_cl={self.user_clearance.value} "
                f"required_cl={self.required_clearance.value}"
            )
        if self.user_department is not None:
            parts.append(f"user_dept={self.user_department.value}")
        if self.required_departments:
            parts.append(f"required_depts={self.required_departments}")
        if self.resource_id is not None:
            parts.append(f"resource={self.resource_id}")
        return " | ".join(parts)


class MetadataValidationError(SecurityError):
    """Raised when a chunk or document carries invalid security metadata.

    Examples:
        - ``clearance_level`` outside ``0..3``.
        - ``department`` value not in :class:`~core.security.Department`.
        - ``allowed_departments`` containing values not in
          :class:`~core.security.AllowedDepartment`.
    """


class UserNotFoundError(SecurityError):
    """Raised when a requested user identity cannot be resolved.

    The application requests a user by ``user_id`` from the
    :class:`~infrastructure.user_store.UserStore`; if the store cannot
    find that id (typo, deleted account, fixture out of sync) this is
    raised. Under SecurityError because failing to resolve identity is
    an auth-layer event — the api adapter should translate it into a
    401/403 response, never into a 500.
    """

    def __init__(self, user_id: str, message: str | None = None) -> None:
        super().__init__(message or f"User not found: {user_id!r}.")
        self.user_id: str = user_id


# =============================================================================
# Infrastructure branch
# =============================================================================

class InfrastructureError(ZeroTrustBaseError):
    """Parent of every error raised by the infrastructure adapters.

    These represent failures of external services (vector DB, lexical index,
    LLM server) — i.e., the boundary between the application and the world.
    """


class VectorDBConnectionError(InfrastructureError):
    """Raised when the ChromaDB service cannot be reached or refuses the request.

    The cause is preserved via ``raise ... from e`` chaining; common
    underlying types are :class:`ConnectionError`, :class:`TimeoutError`,
    and ``requests.exceptions.RequestException`` (raised inside
    ``chromadb.HttpClient``).
    """


class LexicalIndexError(InfrastructureError):
    """Raised on failures of the BM25 lexical index (load, save, query)."""


class LLMError(InfrastructureError):
    """Parent of every LLM-related error (transport, generation, parsing)."""


class LLMConnectionError(LLMError):
    """Raised when the LLM server (Ollama) is unreachable or times out."""


class LLMGenerationError(LLMError):
    """Raised when the LLM server returns an HTTP error or refuses generation."""


class LLMResponseFormatError(LLMError):
    """Raised when the LLM response is not parseable as expected.

    Typical case: the model was prompted with ``format=json`` but returned
    malformed JSON, or a required field is missing from a structured-output
    schema. Carrying the raw response in ``args[0]`` lets diagnostics show
    exactly what came back.
    """


# =============================================================================
# Pipeline branch
# =============================================================================

class PipelineError(ZeroTrustBaseError):
    """Parent of every business-pipeline orchestration error.

    Distinct from :class:`InfrastructureError` because the cause lies in the
    application's own logic (chunking, routing, ingestion control flow) rather
    than in an external service.
    """


class DocumentProcessingError(PipelineError):
    """Raised when a document cannot be loaded, decoded, or chunked.

    Examples:
        - Unsupported file extension.
        - PDF file is encrypted or corrupted.
        - DOCX schema is malformed.
    """

    def __init__(
        self,
        message: str,
        *,
        source: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.source: str | None = source
        self.details: dict[str, Any] = details or {}


class IngestionError(PipelineError):
    """Raised by the ingestion orchestrator when a step in the upload-to-index
    pipeline fails irrecoverably (e.g., metadata classification rejected,
    embedding step failed mid-batch)."""


class RoutingError(PipelineError):
    """Raised when the hierarchical router fails to produce a viable target.

    Examples:
        - The Router-Index is empty.
        - The KSP for the top-1 document is missing the ``home_department`` key.
        - The router returned a department that does not match any
          physical collection.
    """


# =============================================================================
# Public re-exports
# =============================================================================

__all__: list[str] = [
    # Root
    "ZeroTrustBaseError",
    # Configuration
    "ConfigurationError",
    # Security
    "SecurityError",
    "UnauthorizedAccessError",
    "MetadataValidationError",
    "UserNotFoundError",
    # Infrastructure
    "InfrastructureError",
    "VectorDBConnectionError",
    "LexicalIndexError",
    "LLMError",
    "LLMConnectionError",
    "LLMGenerationError",
    "LLMResponseFormatError",
    # Pipeline
    "PipelineError",
    "DocumentProcessingError",
    "IngestionError",
    "RoutingError",
]
