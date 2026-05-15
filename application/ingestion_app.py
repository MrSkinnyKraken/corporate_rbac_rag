"""Document ingestion orchestrator (Phase 4 Step 7).

The :class:`IngestionApp` composes every Phase 1-3 / Phase 4 component
into one cohesive two-phase use case:

    Phase A  —  propose
        upload bytes (or path)
            └─▶ DocumentParser            (extract text)
                └─▶ LLMDocumentClassifier (LLM proposes metadata)
                    └─▶ IngestionProposal (returned to UI)

    Phase B  —  commit
        IngestionProposal + (possibly edited) ProposedMetadata
            └─▶ CustomRBACChunker         (structural split + PII isolation)
                └─▶ allowed_departments expansion  (from final_metadata)
                    └─▶ Embedder         (per-chunk dense vector)
                        └─▶ ChromaDBClient (chunks_<dept>)
                            └─▶ BM25Client.fit refresh
                                └─▶ KSPRouterIndex.add
                                    └─▶ IngestionReport

The split between propose and commit materialises the
human-in-the-loop confirmation the chunker docstring explicitly defers
("Two human-in-the-loop / LLM steps belong to the orchestration
layer"). Both methods accept the :class:`User` value object the demo
api adapter resolves from the Streamlit user-picker.

Idempotency: ``commit`` deletes any pre-existing chunks / KSP record
with the same ``parent_doc_id`` before writing, so retries — including
the UI's "Commit" button being clicked twice — converge to one copy
rather than accumulating duplicates.

Failure ordering: operations run from least-durable to most-durable so
a mid-pipeline crash leaves the corpus in a safer state:

    1. parse                (in-memory only)
    2. chunk + embed        (in-memory only)
    3. Chroma chunks write  (delete-then-add; durable but recoverable)
    4. BM25 refit + save    (derived from Chroma)
    5. KSP write            (least critical — missing KSP just means
                              the router can't surface this doc; the
                              chunks are still retrievable by direct
                              dept routing)
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.config import Settings, get_settings
from core.exceptions import (
    IngestionError,
    PipelineError,
)
from core.security import AllowedDepartment, ClearanceLevel, Department
from domain.chunking.core_chunker import ChunkInput, CustomRBACChunker
from domain.chunking.cross_dept_rules import expand_allowed_departments
from domain.chunking.parsers import DocumentParser
from domain.routing.ksp_index import KSPRecord, KSPRouterIndex
from domain.users import User
from infrastructure.embedder import Embedder, get_embedder
from infrastructure.lexical_db import BM25Client, IndexedChunk
from infrastructure.vector_db import ChromaDBClient

from application.document_classifier import (
    LLMDocumentClassifier,
    ProposedMetadata,
)


_CHUNK_COLLECTION_PREFIX: str = "chunks_"
_TEXT_EXCERPT_FOR_UI: int = 2000


# ─────────────────────────────────────────────────────────────────────────────
# Proposal + report dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class IngestionProposal:
    """Phase A output handed to the user-confirmation step.

    Carries everything the api adapter needs to render a review form
    AND everything :meth:`IngestionApp.commit` needs to do the writes
    without re-parsing or re-classifying. The ``parsed_text`` is
    repeated here precisely to avoid the re-parse: the api adapter
    serialises the whole proposal between the two requests.

    Attributes:
        parent_doc_id: Document-scoped UUID generated at propose-time.
            Stable across propose → commit so the second phase can
            address the same chunk family on retries.
        source_file:   Original filename for audit / UI display.
        parsed_text:   Full text from :class:`DocumentParser`. Re-used
            by commit so we never parse twice.
        text_excerpt:  Truncated preview (~2 kB) for the UI to show
            without dumping the whole document.
        proposed_metadata: The LLM's classification — the user may
            edit this before commit. Treated by commit as an
            authoritative input ONLY when re-passed through the
            :class:`ProposedMetadata` constructor (so editing rules
            stay enforced).
        uploader_user_id: ``User.user_id`` of the upload originator.
            Carried through to the audit log on commit.
        llm_latency_s: Wall-clock cost of the classifier call.
    """

    parent_doc_id: str
    source_file: str
    parsed_text: str
    text_excerpt: str
    proposed_metadata: ProposedMetadata
    uploader_user_id: str
    llm_latency_s: float


@dataclass(frozen=True, slots=True)
class IngestionReport:
    """Phase B output: what was written, where, and how it was tagged.

    Returned to the api adapter for display and persisted (in a future
    step) to an audit log table. Every field is a primitive scalar or a
    short list so the report is JSON-serialisable end-to-end.

    The ``final_allowed_departments`` field carries the LLM/user
    baseline; per-chunk extensions added by the cross-department rules
    are NOT collapsed back into it (a baseline broadening would no
    longer be auditable). They are summarised in
    ``cross_dept_rules_summary`` and ``n_chunks_extended`` instead, so
    operators can see how much the rules amplified visibility.
    """

    parent_doc_id: str
    source_file: str
    uploader_user_id: str
    home_collection: str           # e.g. "chunks_finance"
    bm25_index_name: str           # mirrors home_collection
    ksp_id: str                    # e.g. "ksp:<parent_doc_id>"
    final_home_department: str     # final_metadata.home_department.value
    final_clearance_level: int     # final_metadata.clearance_level.value
    final_allowed_departments: list[str] = field(default_factory=list)
    n_chunks_total: int = 0
    n_chunks_pii: int = 0
    n_chunks_extended: int = 0     # chunks whose allowed_departments grew beyond the baseline
    cross_dept_rules_summary: dict[str, int] = field(default_factory=dict)
    total_latency_s: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class IngestionApp:
    """Two-phase ingestion orchestrator (propose / commit).

    All dependencies are injected so the use case is testable with
    fakes: pass a fake :class:`BaseLLM`, an in-memory
    :class:`ChromaDBClient`, etc. The class is stateless across
    requests — every method takes everything it needs as arguments
    and returns a fresh dataclass.
    """

    def __init__(
        self,
        *,
        parser: DocumentParser,
        classifier: LLMDocumentClassifier,
        chunker: CustomRBACChunker,
        vector_db: ChromaDBClient,
        lexical_db: BM25Client,
        ksp_index: KSPRouterIndex,
        embedder: Embedder | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._parser: DocumentParser = parser
        self._classifier: LLMDocumentClassifier = classifier
        self._chunker: CustomRBACChunker = chunker
        self._vector_db: ChromaDBClient = vector_db
        self._lexical_db: BM25Client = lexical_db
        self._ksp_index: KSPRouterIndex = ksp_index
        self._embedder: Embedder = embedder or get_embedder()
        self._settings: Settings = settings or get_settings()

    # ────────────────────────────── Phase A ─────────────────────────────────
    def propose(self, file_path: Path, uploader: User) -> IngestionProposal:
        """Parse and classify ``file_path`` for human review.

        Args:
            file_path: An absolute path to a file the parser supports.
                Bytes uploads are handled at the api layer by writing
                to ``settings.raw_docs_dir`` first.
            uploader:  The resolved :class:`User` who initiated the
                upload (from the Streamlit user-picker via the api
                adapter).

        Returns:
            A populated :class:`IngestionProposal`. The api adapter is
            expected to surface it to the user, collect edits, then
            re-construct the final :class:`ProposedMetadata` and call
            :meth:`commit`.

        Raises:
            DocumentProcessingError: parser failure (unsupported
                format, file missing, decoder error).
            IngestionError:          classifier returned an unusable
                proposal.
            LLMConnectionError/LLMGenerationError: forwarded from the
                LLM adapter.
        """
        if not isinstance(uploader, User):
            raise IngestionError(
                f"propose() requires a resolved User; got "
                f"{type(uploader).__name__}={uploader!r}."
            )

        parsed_text: str = self._parser.parse(file_path)
        if not parsed_text or not parsed_text.strip():
            raise IngestionError(
                f"Document at {file_path} parsed to empty text; nothing to classify."
            )

        proposed, latency = self._classifier.classify(
            parsed_text, source_file=file_path.name,
        )
        return IngestionProposal(
            parent_doc_id=uuid.uuid4().hex,
            source_file=file_path.name,
            parsed_text=parsed_text,
            text_excerpt=parsed_text[:_TEXT_EXCERPT_FOR_UI],
            proposed_metadata=proposed,
            uploader_user_id=uploader.user_id,
            llm_latency_s=latency,
        )

    # ────────────────────────────── Phase B ─────────────────────────────────
    def commit(
        self,
        proposal: IngestionProposal,
        final_metadata: ProposedMetadata,
        uploader: User,
    ) -> IngestionReport:
        """Write the document under ``final_metadata`` to all three stores.

        Args:
            proposal:       The :class:`IngestionProposal` produced by
                an earlier :meth:`propose` call. Provides the parsed
                text, ``parent_doc_id`` and ``source_file``.
            final_metadata: The authoritative metadata after user
                review. May differ from ``proposal.proposed_metadata``
                in any field — every difference is the user's
                deliberate edit.
            uploader:       The resolved :class:`User`. Cross-checked
                against ``proposal.uploader_user_id``; mismatch is a
                hard error (the api adapter must not let user A commit
                user B's proposal).

        Returns:
            A populated :class:`IngestionReport`.

        Raises:
            IngestionError:          uploader mismatch, empty text,
                user-edited metadata that violates the
                :class:`ProposedMetadata` invariants (re-validated on
                construction).
            VectorDBConnectionError: any ChromaDB failure during the
                writes.
            LexicalIndexError:       BM25 refit failure.
        """
        if not isinstance(uploader, User):
            raise IngestionError(
                f"commit() requires a resolved User; got "
                f"{type(uploader).__name__}={uploader!r}."
            )
        if uploader.user_id != proposal.uploader_user_id:
            raise IngestionError(
                f"Uploader mismatch: proposal was created by "
                f"{proposal.uploader_user_id!r}, commit attempted by "
                f"{uploader.user_id!r}."
            )

        t0 = time.perf_counter()

        # 1) Build the chunker input from the FINAL metadata
        chunk_input = ChunkInput(
            text=proposal.parsed_text,
            parent_doc_id=proposal.parent_doc_id,
            parent_department=final_metadata.home_department,
            parent_clearance=final_metadata.clearance_level,
            source_file=proposal.source_file,
        )

        # 2) Run the chunker — pure deterministic, no I/O
        chunks = self._chunker.chunk(chunk_input)
        if not chunks:
            raise IngestionError(
                f"Chunker produced zero chunks for {proposal.source_file!r}; "
                f"refusing to commit an empty document."
            )

        # 3) Expand allowed_departments per chunk — the two-layer model the
        # chunker explicitly defers. The LLM/user-confirmed baseline sets
        # the floor; per-chunk regex rules (`cross_dept_rules`) may ADD
        # departments to specific chunks based on their content (never
        # remove inherited ones). Empty rule-fire list when nothing
        # matches; an `[ALL]` baseline short-circuits the rules entirely.
        baseline = list(final_metadata.allowed_departments)
        baseline_strs: list[str] = [d.value for d in baseline]
        baseline_set: set[str] = set(baseline_strs)

        n_chunks_extended: int = 0
        cross_dept_rules_summary: dict[str, int] = {}
        for chunk in chunks:
            effective, fired = expand_allowed_departments(
                chunk.page_content, baseline=baseline,
            )
            effective_strs: list[str] = [d.value for d in effective]
            chunk.metadata["allowed_departments"] = json.dumps(effective_strs)
            chunk.metadata["cross_dept_rules_fired"] = json.dumps(fired)
            if set(effective_strs) != baseline_set:
                n_chunks_extended += 1
            for rule_name in fired:
                cross_dept_rules_summary[rule_name] = (
                    cross_dept_rules_summary.get(rule_name, 0) + 1
                )

        # 4) Embed every chunk's text (shared cached Embedder)
        embeddings: list[list[float]] = [
            self._embedder.embed(chunk.page_content) for chunk in chunks
        ]

        # 5) Resolve target physical store
        home_dept: str = final_metadata.home_department.value
        coll_name: str = f"{_CHUNK_COLLECTION_PREFIX}{home_dept}"

        # 6) Write chunks to ChromaDB (delete-then-add for idempotency)
        coll = self._vector_db.get_or_create_collection(
            coll_name,
            embedding_model=self._settings.embedding_model_name,
        )
        self._vector_db.delete_by_metadata(
            coll, where={"parent_doc_id": proposal.parent_doc_id},
        )

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []
        for chunk in chunks:
            chunk_idx = int(chunk.metadata.get("chunk_index", 0))
            ids.append(f"{proposal.parent_doc_id}:{chunk_idx}")
            documents.append(chunk.page_content)
            metadatas.append(_scalarise_metadata(chunk.metadata))

        self._vector_db.add_to_collection(
            coll,
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        # 7) Refit BM25 from ChromaDB (the authoritative source of truth)
        self._refit_bm25(coll, coll_name)

        # 8) KSP record — delete any prior, then add the new one. The KSP
        # carries the *baseline* allowed_departments (document-wide LLM
        # decision), not any per-chunk extension. The router operates at
        # document granularity; per-chunk RBAC is the retriever's job.
        self._ksp_index.delete_doc(proposal.parent_doc_id)
        self._ksp_index.add([KSPRecord(
            ksp_id=f"ksp:{proposal.parent_doc_id}",
            text=final_metadata.ksp_text,
            parent_doc_id=proposal.parent_doc_id,
            source_file=proposal.source_file,
            home_department=final_metadata.home_department,
            clearance_level=int(final_metadata.clearance_level),
            allowed_departments=baseline_strs,
        )])

        # 9) Build report
        n_pii = sum(1 for c in chunks if c.metadata.get("contains_PII"))
        return IngestionReport(
            parent_doc_id=proposal.parent_doc_id,
            source_file=proposal.source_file,
            uploader_user_id=uploader.user_id,
            home_collection=coll_name,
            bm25_index_name=coll_name,
            ksp_id=f"ksp:{proposal.parent_doc_id}",
            final_home_department=home_dept,
            final_clearance_level=int(final_metadata.clearance_level),
            final_allowed_departments=baseline_strs,
            n_chunks_total=len(chunks),
            n_chunks_pii=n_pii,
            n_chunks_extended=n_chunks_extended,
            cross_dept_rules_summary=dict(cross_dept_rules_summary),
            total_latency_s=round(time.perf_counter() - t0, 4),
        )

    # ───────────────────────────── Internals ────────────────────────────────
    def _refit_bm25(self, coll: Any, coll_name: str) -> None:
        """Rebuild the BM25 index for ``coll_name`` from the current Chroma state.

        Pulled into a helper because the read-and-rebuild dance is
        verbose. Always rebuilds from the *full* current state, never
        from a delta, so ChromaDB is the only source of truth and a
        partially-failed prior commit cannot leave BM25 ahead of
        Chroma (which would surface chunks the retriever cannot find).
        """
        raw = self._vector_db.get_all_documents(coll)
        ids = raw.get("ids") or []
        docs = raw.get("documents") or []
        metas = raw.get("metadatas") or []

        indexed: list[IndexedChunk] = []
        for i, cid in enumerate(ids):
            indexed.append(IndexedChunk(
                chunk_id=str(cid),
                text=docs[i] if i < len(docs) else "",
                metadata=dict(metas[i]) if i < len(metas) and metas[i] else {},
            ))

        if not indexed:
            # Defensive: the collection is empty after our delete/add.
            # That should never happen with a non-empty document, but
            # if it does, evicting the BM25 index is safer than
            # carrying a stale one.
            self._lexical_db.evict(coll_name)
            return

        self._lexical_db.fit(coll_name, indexed)
        self._lexical_db.save_index(coll_name)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _scalarise_metadata(raw: dict[str, Any]) -> dict[str, Any]:
    """Ensure every value is a Chroma-compatible scalar (str/int/float/bool).

    ChromaDB metadata accepts only scalars; lists must be JSON-encoded
    strings. The chunker emits ``sensitivity_types`` as a list — flatten
    it here. ``allowed_departments`` has already been JSON-encoded by
    the orchestrator before this helper runs.
    """
    out: dict[str, Any] = {}
    for k, v in raw.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        elif isinstance(v, (list, tuple)):
            out[k] = json.dumps(list(v))
        else:
            out[k] = str(v)
    return out
