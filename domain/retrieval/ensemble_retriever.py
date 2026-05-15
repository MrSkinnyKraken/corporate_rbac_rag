"""Asymmetric Ensemble Retriever — the orchestrator of Phase 2's RAG flow.

Composes every retrieval component validated in Phases 1-3 into a single
end-to-end pipeline:

    query  ─▶  embed                                 (Phase 2 Step 0)
           ─▶  per-dept ChromaDB query  ─┐
                                          ├─▶ pool ─▶ RBAC filter   (Step 4)
           ─▶  per-dept BM25 query      ─┘            │
                                                       ▼
                                            weighted RRF fusion     (Step 2)
                                                       │
                                                       ▼
                                              top-K winners
                                                       │
                                                       ▼
                                  parent-child reconstruction       (Step 3)
                                                       │
                                                       ▼
                                       :class:`RetrievalResult`

Architectural decisions baked in:

* **Per-department physical collections** — the routing strategy adopted at
  the end of Phase 3 Step 4. The retriever accepts a ``target_departments``
  list (typically the top-2 from the KSP router), and for each dept it
  opens both the matching ``chunks_<dept>`` ChromaDB collection AND the
  matching ``<dept>.pkl`` BM25 index, pooling candidates across them.
* **RBAC BEFORE fusion** — the Zero-Trust filter runs on the raw branch
  outputs so RRF only fuses chunks the user is authorised to see. This
  eliminates the *data starvation* failure mode where a top-K RRF result
  ends up empty after a late RBAC cull.
* **Parent-child via metadata** — the retriever uses each winning chunk's
  ``parent_doc_id`` (UUID emitted by the chunker) to fetch its siblings
  via ``ChromaDBClient.get_documents_where``, RBAC-filters those siblings
  too, and stitches them in ``chunk_index`` order. No external JSON
  fixture is required (Phase 1's notebook trick).
* **Empty-result safety** — when no chunks survive RBAC, the retriever
  returns a structured :class:`RetrievalResult` with ``is_empty=True`` and
  a context string carrying the explicit ``[NO ACCESSIBLE CONTEXT]``
  marker so the LLM can produce an honest refusal instead of hallucinating.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.config import Settings, get_settings
from core.exceptions import (
    LexicalIndexError,
    PipelineError,
    VectorDBConnectionError,
)
from core.security import ClearanceLevel, Department
from domain.retrieval.rbac_filter import apply_zero_trust_filter
from domain.retrieval.rrf_fusion import calculate_dynamic_rrf
from infrastructure.embedder import Embedder, get_embedder
from infrastructure.lexical_db import BM25Client
from infrastructure.vector_db import ChromaDBClient


_NO_CONTEXT_MARKER: str = (
    "[NO ACCESSIBLE CONTEXT]\n"
    "The retrieval layer found no chunks the user may read for this query. "
    "Decline to answer and state that the information is not available within "
    "the user's permissions."
)

_COLLECTION_PREFIX: str = "chunks_"


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """The full output of :meth:`AsymmetricEnsembleRetriever.retrieve_secure_context`.

    Attributes:
        chunks: The top-K winning chunks, in RRF order. Each chunk dict
            carries the original metadata plus ``rrf_score`` and
            ``rrf_components`` keys added by the fusion stage.
        parent_documents: One reconstructed parent per unique
            ``parent_doc_id`` among the winners. Each entry has
            ``parent_doc_id``, ``source_file``, ``text``,
            ``n_accessible_chunks`` and ``n_total_chunks``.
        context: The final string handed to the LLM. Empty results are
            marked with :data:`_NO_CONTEXT_MARKER` so the prompt template
            can detect them without inspecting other fields.
        target_departments: The departments the router asked the retriever
            to query.
        candidates_pre_rbac: Distinct chunk count from both branches
            BEFORE the RBAC cut.
        candidates_post_rbac: Distinct chunk count AFTER the RBAC cut.
        alpha_vector: The α weight effectively used by RRF in this run
            (echoed for telemetry / audit logs).
    """

    chunks: list[dict[str, Any]] = field(default_factory=list)
    parent_documents: list[dict[str, Any]] = field(default_factory=list)
    context: str = ""
    target_departments: list[str] = field(default_factory=list)
    candidates_pre_rbac: int = 0
    candidates_post_rbac: int = 0
    alpha_vector: float = 0.5

    @property
    def is_empty(self) -> bool:
        """``True`` iff no chunks survived the pipeline (either no
        candidates from the indexes, or RBAC dropped them all)."""
        return not self.chunks


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class AsymmetricEnsembleRetriever:
    """Orchestrates the full Phase-2 retrieval flow against the production stack.

    The class receives both infrastructure adapters (vector DB and lexical
    DB) as constructor dependencies (Dependency Injection) so it can be
    unit-tested with fakes. Domain primitives (:func:`apply_zero_trust_filter`
    and :func:`calculate_dynamic_rrf`) are imported directly because they
    are pure functions with no side effects.
    """

    def __init__(
        self,
        vector_db: ChromaDBClient,
        lexical_db: BM25Client,
        *,
        settings: Settings | None = None,
        embedder: Embedder | None = None,
        oversample_per_branch: int = 30,
        rrf_k: int = 60,
        final_top_k: int = 5,
    ) -> None:
        """Initialise the retriever.

        Args:
            vector_db: Adapter providing a ChromaDB ``HttpClient`` per the
                Phase 4 production rule.
            lexical_db: Adapter for per-department BM25 indexes.
            settings: Application settings; if ``None``, loaded via
                :func:`~core.config.get_settings`.
            embedder: Shared sentence-encoder service. If ``None``,
                fetched via :func:`~infrastructure.embedder.get_embedder`
                so the model and per-query cache are shared with the
                routing layer — the router's pre-flight embedding of the
                query is reused here at near-zero cost.
            oversample_per_branch: Per-branch, per-department over-fetch
                budget. With 2 routed departments and the default 30, each
                branch yields up to 60 candidates pre-RBAC, totalling up
                to 120 across both branches before deduplication.
            rrf_k: Cormack constant for :func:`calculate_dynamic_rrf`.
            final_top_k: How many fused chunks to return as the winners.
        """
        if oversample_per_branch < 1:
            raise PipelineError(
                f"oversample_per_branch must be >= 1; got {oversample_per_branch}."
            )
        if final_top_k < 1:
            raise PipelineError(f"final_top_k must be >= 1; got {final_top_k}.")
        self._vector_db: ChromaDBClient = vector_db
        self._lexical_db: BM25Client = lexical_db
        self._settings: Settings = settings or get_settings()
        self._embedder: Embedder = embedder or get_embedder()
        self._oversample: int = oversample_per_branch
        self._rrf_k: int = rrf_k
        self._final_top_k: int = final_top_k

    # ─────────────────────────── Public API ─────────────────────────────────
    def retrieve_secure_context(
        self,
        query: str,
        user_clearance: ClearanceLevel,
        user_department: Department,
        target_departments: list[str],
        *,
        alpha: float = 0.5,
    ) -> RetrievalResult:
        """Run the full asymmetric ensemble retrieval pipeline.

        Args:
            query: The user's natural-language question.
            user_clearance: The requesting user's clearance level.
            user_department: The requesting user's home department.
            target_departments: The departments the routing layer chose
                to query (typically top-2 of the KSP router). Each value
                must match an existing physical collection
                ``chunks_<dept>``; unknown departments are silently
                skipped.
            alpha: Vector-branch weight in [0, 1] passed to RRF. Default
                0.5 (balanced); the routing/intent classifier will
                override this dynamically per query.

        Returns:
            A populated :class:`RetrievalResult`. When no chunks survive,
            :attr:`RetrievalResult.is_empty` is ``True`` and ``context``
            holds the explicit no-access marker for the LLM.

        Raises:
            PipelineError: invalid arguments (empty query, alpha out of
                range, empty target list).
            VectorDBConnectionError: ChromaDB unreachable during the
                vector branch.
        """
        # ── Argument validation ────────────────────────────────────────────
        if not query or not query.strip():
            raise PipelineError("Empty query passed to retrieve_secure_context.")
        if not (0.0 <= alpha <= 1.0):
            raise PipelineError(f"alpha must be in [0.0, 1.0], got {alpha!r}.")
        if not target_departments:
            raise PipelineError(
                "target_departments is empty; the router must return at least one."
            )

        # ── 1. Embed the query once ────────────────────────────────────────
        query_embedding: list[float] = self._embed_query(query)

        # ── 2. Branch out: per-dept vector + lexical retrieval ─────────────
        vector_candidates, vector_origins = self._collect_vector_candidates(
            query_embedding, target_departments,
        )
        lexical_candidates = self._collect_lexical_candidates(
            query, target_departments,
        )

        n_pre_rbac: int = self._distinct_count(vector_candidates + lexical_candidates)

        # ── 3. RBAC BEFORE fusion (data-starvation guard) ──────────────────
        cl_int: int = int(user_clearance)
        dept_str: str = user_department.value
        filtered_vector = apply_zero_trust_filter(vector_candidates, cl_int, dept_str)
        filtered_lexical = apply_zero_trust_filter(lexical_candidates, cl_int, dept_str)

        n_post_rbac: int = self._distinct_count(filtered_vector + filtered_lexical)

        if n_post_rbac == 0:
            return RetrievalResult(
                chunks=[],
                parent_documents=[],
                context=_NO_CONTEXT_MARKER,
                target_departments=list(target_departments),
                candidates_pre_rbac=n_pre_rbac,
                candidates_post_rbac=0,
                alpha_vector=alpha,
            )

        # ── 4. Weighted RRF fusion ─────────────────────────────────────────
        fused = calculate_dynamic_rrf(
            filtered_vector, filtered_lexical, alpha, rrf_k=self._rrf_k,
        )

        # ── 5. Slice top-K winners ─────────────────────────────────────────
        top_chunks = fused[: self._final_top_k]

        # ── 6. Parent-child reconstruction (RBAC re-applied to siblings) ──
        parents, context = self._reconstruct_parents(
            top_chunks, vector_origins, cl_int, dept_str,
        )

        # If the parent reconstruction yielded no readable text (every
        # parent's siblings were forbidden), fall back to concatenating the
        # winning chunks themselves so the LLM still gets some context.
        if not context:
            context = self._winners_only_context(top_chunks)

        return RetrievalResult(
            chunks=top_chunks,
            parent_documents=parents,
            context=context,
            target_departments=list(target_departments),
            candidates_pre_rbac=n_pre_rbac,
            candidates_post_rbac=n_post_rbac,
            alpha_vector=alpha,
        )

    # ──────────────────────────── Internals ─────────────────────────────────
    def _embed_query(self, query: str) -> list[float]:
        """Delegate to the shared :class:`Embedder` (cached for repeats)."""
        return self._embedder.embed(query)

    def _collect_vector_candidates(
        self,
        query_embedding: list[float],
        target_departments: list[str],
    ) -> tuple[list[dict[str, Any]], dict[str, str]]:
        """Query each routed department's ChromaDB collection.

        Returns:
            ``(candidates, origin_map)`` where ``origin_map[chunk_id]``
            records the collection a chunk came from. The origin is
            essential for parent-child reconstruction: a winner's
            siblings live in the same physical collection as the winner.
        """
        candidates: list[dict[str, Any]] = []
        origins: dict[str, str] = {}
        for dept in target_departments:
            coll_name = self._collection_name(dept)
            try:
                coll = self._vector_db.get_or_create_collection(
                    coll_name,
                    embedding_model=self._settings.embedding_model_name,
                )
                response = self._vector_db.query_collection(
                    coll,
                    query_embeddings=[query_embedding],
                    n_results=self._oversample,
                )
            except VectorDBConnectionError:
                # Bubble up — the caller (FastAPI handler) translates
                # this into a 503 / honest error to the user.
                raise
            for chunk in self._unpack_query_response(response, coll_name):
                origins[chunk["chunk_id"]] = coll_name
                candidates.append(chunk)
        return candidates, origins

    def _collect_lexical_candidates(
        self,
        query: str,
        target_departments: list[str],
    ) -> list[dict[str, Any]]:
        """Query each routed department's BM25 index.

        Departments without a BM25 index yet (first ingest, or the
        ingestion pipeline never built one) are skipped silently — the
        vector branch still has a chance to surface relevant chunks for
        that dept. A failing BM25 query for a single dept does NOT abort
        the whole retrieval.
        """
        candidates: list[dict[str, Any]] = []
        for dept in target_departments:
            coll_name = self._collection_name(dept)
            if not self._lexical_db.has_index(coll_name):
                continue
            try:
                hits = self._lexical_db.search(
                    coll_name, query, top_k=self._oversample,
                )
            except LexicalIndexError:
                # One bad index should not poison the whole query;
                # the application logger should pick this up via the
                # exception's __cause__ chain on a higher level.
                continue
            for indexed_chunk, score in hits:
                candidates.append({
                    "chunk_id": indexed_chunk.chunk_id,
                    "text": indexed_chunk.text,
                    "metadata": dict(indexed_chunk.metadata),
                    "score": score,
                    "_origin_collection": coll_name,
                })
        return candidates

    def _reconstruct_parents(
        self,
        winners: list[dict[str, Any]],
        vector_origins: dict[str, str],
        user_clearance: int,
        user_department: str,
    ) -> tuple[list[dict[str, Any]], str]:
        """Fetch every accessible sibling of every winner and stitch context.

        Steps for each unique ``parent_doc_id`` among the winners:

          1. Determine the home collection (the winner's
             ``_origin_collection`` or, as fallback, ``vector_origins``).
          2. Fetch every chunk in that collection with the same
             ``parent_doc_id`` via the ChromaDB ``where`` filter.
          3. Re-apply the Zero-Trust filter to the siblings — a doc that
             a user can read at chunk level may still contain
             higher-clearance fragments (e.g., an STRICT IBAN row inside
             a CONFIDENTIAL contract). The RBAC layer must still
             remove them.
          4. Sort the survivors by ``chunk_index``, stitch them with
             newlines, and emit one parent record + one context segment.

        Returns:
            ``(parent_docs, context_string)``. ``context_string`` is
            empty if every parent had zero accessible siblings; the
            caller falls back to the winners-only context in that case.
        """
        parent_groups: dict[str, list[dict[str, Any]]] = {}
        for winner in winners:
            pid = self._parent_id(winner)
            if pid:
                parent_groups.setdefault(pid, []).append(winner)

        parents_out: list[dict[str, Any]] = []
        context_segments: list[str] = []

        for pid, group in parent_groups.items():
            origin = (
                group[0].get("_origin_collection")
                or vector_origins.get(group[0].get("chunk_id", ""))
            )
            if not origin:
                continue
            try:
                siblings = self._fetch_siblings(origin, pid)
            except VectorDBConnectionError:
                # Skip this parent; the LLM will work with what's left.
                continue

            accessible = apply_zero_trust_filter(
                siblings, user_clearance, user_department,
            )
            if not accessible:
                continue

            accessible.sort(key=self._chunk_index_key)
            full_text = "\n".join(
                c.get("text", "") for c in accessible if c.get("text")
            )
            if not full_text.strip():
                continue
            source_meta = accessible[0].get("metadata") or {}
            source = str(source_meta.get("source_file", pid))

            parents_out.append({
                "parent_doc_id":         pid,
                "source_file":           source,
                "text":                  full_text,
                "n_accessible_chunks":   len(accessible),
                "n_total_chunks":        len(siblings),
            })
            context_segments.append(
                f"=== Source: {source} (parent_doc_id={pid}) ===\n{full_text}"
            )

        return parents_out, "\n\n".join(context_segments)

    def _fetch_siblings(
        self,
        collection_name: str,
        parent_doc_id: str,
    ) -> list[dict[str, Any]]:
        """Fetch every chunk of ``parent_doc_id`` from ``collection_name``."""
        coll = self._vector_db.get_or_create_collection(
            collection_name,
            embedding_model=self._settings.embedding_model_name,
        )
        raw = self._vector_db.get_documents_where(
            coll, where={"parent_doc_id": parent_doc_id},
        )
        ids = raw.get("ids") or []
        docs = raw.get("documents") or []
        metas = raw.get("metadatas") or []
        result: list[dict[str, Any]] = []
        for i, cid in enumerate(ids):
            result.append({
                "chunk_id": cid,
                "text":     docs[i] if i < len(docs) else "",
                "metadata": metas[i] if i < len(metas) else {},
            })
        return result

    # ─────────────────────────── Tiny helpers ───────────────────────────────
    @staticmethod
    def _collection_name(department: str) -> str:
        """Map a department string to its physical ChromaDB / BM25 name."""
        return f"{_COLLECTION_PREFIX}{department.strip().lower()}"

    @staticmethod
    def _unpack_query_response(
        response: dict[str, Any],
        collection_name: str,
    ) -> list[dict[str, Any]]:
        """Convert a ChromaDB query response into a list of canonical chunks."""
        if not response.get("ids") or not response["ids"][0]:
            return []
        ids = response["ids"][0]
        docs = (response.get("documents") or [[]])[0]
        metas = (response.get("metadatas") or [[]])[0]
        dists = (response.get("distances") or [[]])[0]
        chunks: list[dict[str, Any]] = []
        for i, cid in enumerate(ids):
            chunks.append({
                "chunk_id":            cid,
                "text":                docs[i] if i < len(docs) else "",
                "metadata":            metas[i] if i < len(metas) else {},
                "distance":            dists[i] if i < len(dists) else None,
                "_origin_collection":  collection_name,
            })
        return chunks

    @staticmethod
    def _parent_id(chunk: dict[str, Any]) -> str | None:
        """Extract a parent_doc_id from a chunk dict (None if missing)."""
        meta = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
        candidate = meta.get("parent_doc_id") or chunk.get("parent_doc_id")
        return str(candidate) if candidate else None

    @staticmethod
    def _chunk_index_key(chunk: dict[str, Any]) -> int:
        """Return ``chunk_index`` as int for sort, defaulting to 0 when missing."""
        meta = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
        try:
            return int(meta.get("chunk_index", 0))
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _distinct_count(chunks: list[dict[str, Any]]) -> int:
        """Count distinct chunk identifiers across a candidate list."""
        seen: set[str] = set()
        for c in chunks:
            cid = c.get("chunk_id") or c.get("id")
            if cid:
                seen.add(str(cid))
            else:
                meta = c.get("metadata") if isinstance(c.get("metadata"), dict) else {}
                seen.add(
                    f"{meta.get('parent_doc_id', '?')}#{meta.get('chunk_index', '?')}"
                )
        return len(seen)

    @staticmethod
    def _winners_only_context(winners: list[dict[str, Any]]) -> str:
        """Fallback context when every parent reconstruction was forbidden.

        Concatenates just the winning chunks' own text. Better than an
        empty context — the LLM at least sees the snippets that ranked
        highest, even if their parent paragraphs are out of reach.
        """
        segments: list[str] = []
        for w in winners:
            text = w.get("text") or ""
            if text.strip():
                segments.append(text)
        return "\n\n".join(segments)
