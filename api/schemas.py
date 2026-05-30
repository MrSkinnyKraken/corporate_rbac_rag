"""Pydantic v2 schemas for the FastAPI surface.

These models live at the api/domain boundary: they serialise domain
dataclasses to JSON for HTTP responses and parse incoming JSON bodies
back into the typed domain objects the orchestrators expect.

Keeping them separate from the internal dataclasses (instead of making
the dataclasses Pydantic-native) preserves two properties:

  * The domain stays free of HTTP-stack concerns — `User`,
    `IngestionProposal`, `QueryResponse` … remain pure Python and can
    be unit-tested without FastAPI installed.
  * Wire formats can evolve independently. A schema field rename or
    versioning bump does not require changes inside `domain/` or
    `application/`.

Conventions:
  * Schemas suffixed with ``Schema`` are wire formats.
  * ``from_domain(...)`` is the conversion *out* (response building);
    ``to_domain(...)`` is the conversion *in* (request unpacking).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from application.document_classifier import ProposedMetadata
from application.ingestion_app import IngestionProposal, IngestionReport
from application.query_app import Citation, QueryResponse
from core.security import AllowedDepartment, ClearanceLevel, Department
from domain.users import User


# ─────────────────────────────────────────────────────────────────────────────
# Users
# ─────────────────────────────────────────────────────────────────────────────

class UserSchema(BaseModel):
    """Wire format for a :class:`~domain.users.User`."""

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "user_id": "u-005",
            "username": "Laia Roca",
            "clearance_level": 3,
            "department": "legal",
            "email": "laia.roca@demo.local",
        }
    })

    user_id: str
    username: str
    clearance_level: int = Field(ge=0, le=3)
    department: str
    email: str | None = None

    @classmethod
    def from_domain(cls, u: User) -> "UserSchema":
        return cls(
            user_id=u.user_id,
            username=u.username,
            clearance_level=int(u.clearance_level),
            department=u.department.value,
            email=u.email,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Ingestion
# ─────────────────────────────────────────────────────────────────────────────

class ProposedMetadataSchema(BaseModel):
    """Wire format for the LLM-proposed (or user-edited) document metadata."""

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "home_department": "legal",
            "allowed_departments": ["legal", "operations"],
            "clearance_level": 2,
            "ksp_text": "Master service agreement warranty annex covering "
                        "defect remediation, claims procedure and refund terms.",
            "reasoning": "Document is a legal contract addendum.",
        }
    })

    home_department: str
    allowed_departments: list[str] = Field(min_length=1)
    clearance_level: int = Field(ge=0, le=3)
    ksp_text: str = Field(min_length=1)
    reasoning: str = ""

    @classmethod
    def from_domain(cls, pm: ProposedMetadata) -> "ProposedMetadataSchema":
        return cls(
            home_department=pm.home_department.value,
            allowed_departments=[d.value for d in pm.allowed_departments],
            clearance_level=int(pm.clearance_level),
            ksp_text=pm.ksp_text,
            reasoning=pm.reasoning,
        )

    def to_domain(self) -> ProposedMetadata:
        return ProposedMetadata(
            home_department=Department(self.home_department.strip().lower()),
            allowed_departments=[
                AllowedDepartment(d.strip().lower()) for d in self.allowed_departments
            ],
            clearance_level=ClearanceLevel.from_int(int(self.clearance_level)),
            ksp_text=self.ksp_text,
            reasoning=self.reasoning,
        )


class IngestionProposalSchema(BaseModel):
    """Wire format for :class:`~application.ingestion_app.IngestionProposal`.

    Carries ``parsed_text`` end-to-end so the api adapter can be stateless
    between propose and commit: the frontend keeps the proposal in
    its session and resends it on commit. A future iteration may swap
    this for a server-side cache keyed by ``parent_doc_id`` if the
    payloads grow too large.
    """

    parent_doc_id: str
    source_file: str
    parsed_text: str
    text_excerpt: str
    proposed_metadata: ProposedMetadataSchema
    uploader_user_id: str
    llm_latency_s: float

    @classmethod
    def from_domain(cls, p: IngestionProposal) -> "IngestionProposalSchema":
        return cls(
            parent_doc_id=p.parent_doc_id,
            source_file=p.source_file,
            parsed_text=p.parsed_text,
            text_excerpt=p.text_excerpt,
            proposed_metadata=ProposedMetadataSchema.from_domain(p.proposed_metadata),
            uploader_user_id=p.uploader_user_id,
            llm_latency_s=p.llm_latency_s,
        )

    def to_domain(self) -> IngestionProposal:
        return IngestionProposal(
            parent_doc_id=self.parent_doc_id,
            source_file=self.source_file,
            parsed_text=self.parsed_text,
            text_excerpt=self.text_excerpt,
            proposed_metadata=self.proposed_metadata.to_domain(),
            uploader_user_id=self.uploader_user_id,
            llm_latency_s=self.llm_latency_s,
        )


class CommitRequest(BaseModel):
    """Request body for ``POST /ingestion/commit``."""

    proposal: IngestionProposalSchema
    final_metadata: ProposedMetadataSchema


class IngestionReportSchema(BaseModel):
    """Wire format for :class:`~application.ingestion_app.IngestionReport`."""

    parent_doc_id: str
    source_file: str
    uploader_user_id: str
    home_collection: str
    bm25_index_name: str
    ksp_id: str
    final_home_department: str
    final_clearance_level: int
    final_allowed_departments: list[str]
    n_chunks_total: int
    n_chunks_pii: int
    n_chunks_extended: int
    cross_dept_rules_summary: dict[str, int]
    total_latency_s: float

    @classmethod
    def from_domain(cls, r: IngestionReport) -> "IngestionReportSchema":
        return cls(
            parent_doc_id=r.parent_doc_id,
            source_file=r.source_file,
            uploader_user_id=r.uploader_user_id,
            home_collection=r.home_collection,
            bm25_index_name=r.bm25_index_name,
            ksp_id=r.ksp_id,
            final_home_department=r.final_home_department,
            final_clearance_level=r.final_clearance_level,
            final_allowed_departments=list(r.final_allowed_departments),
            n_chunks_total=r.n_chunks_total,
            n_chunks_pii=r.n_chunks_pii,
            n_chunks_extended=r.n_chunks_extended,
            cross_dept_rules_summary=dict(r.cross_dept_rules_summary),
            total_latency_s=r.total_latency_s,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Query
# ─────────────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """Request body for ``POST /query``."""

    model_config = ConfigDict(json_schema_extra={
        "example": {"query": "What is the warranty duration in the MSA?"}
    })

    query: str = Field(min_length=1)


class CitationSchema(BaseModel):
    parent_doc_id: str
    source_file: str
    n_accessible_chunks: int
    n_total_chunks: int

    @classmethod
    def from_domain(cls, c: Citation) -> "CitationSchema":
        return cls(
            parent_doc_id=c.parent_doc_id,
            source_file=c.source_file,
            n_accessible_chunks=c.n_accessible_chunks,
            n_total_chunks=c.n_total_chunks,
        )


class QueryResponseSchema(BaseModel):
    """Wire format for :class:`~application.query_app.QueryResponse`."""

    user_id: str
    query: str
    answer: str
    citations: list[CitationSchema]
    refused: bool
    refusal_reason: str | None
    intent: str
    alpha: float
    target_departments: list[str]
    candidates_pre_rbac: int
    candidates_post_rbac: int
    n_chunks_used: int
    n_parents_used: int
    routing_latency_s: float
    retrieval_latency_s: float
    llm_latency_s: float
    total_latency_s: float
    llm_model: str | None

    @classmethod
    def from_domain(cls, r: QueryResponse) -> "QueryResponseSchema":
        return cls(
            user_id=r.user_id,
            query=r.query,
            answer=r.answer,
            citations=[CitationSchema.from_domain(c) for c in r.citations],
            refused=r.refused,
            refusal_reason=r.refusal_reason,
            intent=r.intent,
            alpha=r.alpha,
            target_departments=list(r.target_departments),
            candidates_pre_rbac=r.candidates_pre_rbac,
            candidates_post_rbac=r.candidates_post_rbac,
            n_chunks_used=r.n_chunks_used,
            n_parents_used=r.n_parents_used,
            routing_latency_s=r.routing_latency_s,
            retrieval_latency_s=r.retrieval_latency_s,
            llm_latency_s=r.llm_latency_s,
            total_latency_s=r.total_latency_s,
            llm_model=r.llm_model,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    """``GET /health`` response — surfaces the state of every backend."""

    status: str                # "ok" | "degraded"
    chroma: str                # "up" | "down"
    ollama: str                # "up" | "down"
    embedder_loaded: bool
    embedding_model: str
    llm_model: str
