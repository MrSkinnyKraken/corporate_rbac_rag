"""Key Semantic Points (KSP) Router-Index.

The KSP index is a *specialised ChromaDB collection* — one record per
ingested document — that the router queries to decide which physical
``chunks_<dept>`` collections the retrieval engine should open.

A KSP record holds:

  * **text**: a 1-3 sentence dense summary of the source document. The
    ingestion pipeline (later phase) is responsible for producing it
    via the LLM-based document classifier (Phase 3 Step 1).
  * **home_department**: the single :class:`~core.security.Department`
    the document belongs to — the value the router promotes to
    ``target_departments`` when this KSP is matched.
  * **parent_doc_id**, **source_file**: the link back to the actual
    document so audit logs can trace a routing decision.
  * **clearance_level**, **allowed_departments**: RBAC fields, inherited
    from the document's coarsest security envelope. The router applies
    the same Zero-Trust filter as the retriever does on chunks — there
    is no point routing a user to a department they cannot read,
    AND surfacing the *existence* of a forbidden document via its KSP
    summary would itself be a leak.

The KSP collection is embedded with the same model as the per-department
chunk collections — :data:`~core.config.Settings.embedding_model_name` —
so the query vector produced by the shared :class:`~infrastructure.embedder.Embedder`
is reusable on both indexes. The embedding-model runtime guard (Phase 4
follow-up) is enforced on the KSP collection too.

Importantly, the index exposes a strict separation between *writing*
(used by the ingestion pipeline) and *reading* (used by the router).
The router never invokes :meth:`add` or :meth:`delete_doc`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Sequence

from core.config import Settings, get_settings
from core.exceptions import PipelineError, RoutingError
from core.security import Department
from domain.retrieval.rbac_filter import apply_zero_trust_filter
from infrastructure.embedder import Embedder, get_embedder
from infrastructure.vector_db import ChromaDBClient


_KSP_COLLECTION_NAME: str = "ksp_router_index"


# ─────────────────────────────────────────────────────────────────────────────
# Public dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class KSPRecord:
    """One KSP entry as produced by the ingestion pipeline.

    Constructed BEFORE storage. The :meth:`KSPRouterIndex.add` method
    takes a sequence of these and writes them to ChromaDB after
    converting ``allowed_departments`` to its JSON-encoded form (the
    ChromaDB metadata limitation also applied to chunks).

    Attributes:
        ksp_id: A globally-unique id for this KSP record. Convention:
            ``"ksp:<parent_doc_id>"`` (one KSP per document).
        text: The 1-3 sentence summary the embedder operates on.
        parent_doc_id: UUID of the source document; the same value lives
            in every chunk produced from this document.
        source_file: Human-readable filename for audit logs and UI.
        home_department: The routing answer this KSP delivers — must be
            a :class:`~core.security.Department` enum (i.e., never the
            wildcard ``"all"`` of :class:`AllowedDepartment`).
        clearance_level: Minimum clearance to learn that this document
            exists; matches the *coarsest* chunk-level clearance.
        allowed_departments: Departments authorised to read this
            document's content (chunk-level RBAC). May include the
            wildcard ``"all"`` from
            :data:`~core.security.ALLOWED_DEPARTMENT_WILDCARD`.
    """

    ksp_id: str
    text: str
    parent_doc_id: str
    source_file: str
    home_department: Department
    clearance_level: int
    allowed_departments: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class KSPMatch:
    """One scored hit returned by :meth:`KSPRouterIndex.query`."""

    ksp_id: str
    home_department: str
    parent_doc_id: str
    source_file: str
    distance: float  # cosine distance from the query (lower = closer)

    @property
    def similarity(self) -> float:
        """Convenience: 1 - cosine_distance, monotonically a "score"."""
        return 1.0 - self.distance


# ─────────────────────────────────────────────────────────────────────────────
# Router-Index
# ─────────────────────────────────────────────────────────────────────────────

class KSPRouterIndex:
    """Read/write adapter for the ``ksp_router_index`` ChromaDB collection.

    Two distinct usage modes:

      * **Read** — the routing layer calls :meth:`query` with the user's
        query plus their RBAC context. The method returns the top-N
        accessible :class:`KSPMatch` records; everything the user
        cannot read is filtered out before returning.
      * **Write** — the ingestion pipeline calls :meth:`add` once per
        ingested document, after the document classifier has produced
        the KSP text and routing metadata.

    Both modes share lazy collection creation: the first call after
    process start creates the collection with the embedding-model stamp
    so the runtime guard in :class:`~infrastructure.vector_db.ChromaDBClient`
    catches future model swaps.
    """

    def __init__(
        self,
        vector_db: ChromaDBClient,
        *,
        settings: Settings | None = None,
        embedder: Embedder | None = None,
        collection_name: str = _KSP_COLLECTION_NAME,
    ) -> None:
        """Initialise the KSP Router-Index.

        Args:
            vector_db: ChromaDB adapter shared with the retrieval engine.
            settings: Application settings; if ``None``, fetched via
                :func:`~core.config.get_settings`.
            embedder: Shared sentence-encoder service. If ``None``, the
                process-wide singleton is used so the per-query cache is
                shared with the retrieval layer.
            collection_name: Override the default
                ``"ksp_router_index"`` (used only by tests).
        """
        if not collection_name or not collection_name.strip():
            raise PipelineError("KSPRouterIndex requires a non-empty collection_name.")
        self._vector_db: ChromaDBClient = vector_db
        self._settings: Settings = settings or get_settings()
        self._embedder: Embedder = embedder or get_embedder()
        self._collection_name: str = collection_name
        # Lazily-resolved Collection handle (so construction is cheap)
        self._collection: Any | None = None

    # ─────────────────────────── Public API ─────────────────────────────────
    @property
    def collection_name(self) -> str:
        """The physical ChromaDB collection backing this index."""
        return self._collection_name

    def has_entries(self) -> bool:
        """``True`` iff the collection has at least one KSP stored.

        Used by the application layer to short-circuit routing on a
        fresh deployment: an empty index can only return zero target
        departments, so we can skip the embed-and-query roundtrip and
        return an empty :class:`~domain.routing.router.RoutingDecision`.
        """
        coll = self._get_collection()
        return self._vector_db.collection_count(coll) > 0

    def add(self, records: Sequence[KSPRecord]) -> None:
        """Bulk-insert KSP records into the index.

        Idempotency is the *caller's* responsibility (the ingestion
        pipeline should pass ``ksp_id`` values that are stable across
        re-ingestions of the same document).

        Args:
            records: KSPs to insert. An empty sequence is a no-op.

        Raises:
            PipelineError: any record carries an invalid
                ``home_department`` (must be a :class:`Department` enum,
                never the wildcard ``"all"``).
            VectorDBConnectionError: ChromaDB is unreachable.
        """
        if not records:
            return

        ids: list[str] = []
        embeddings: list[list[float]] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for rec in records:
            if not isinstance(rec.home_department, Department):
                raise PipelineError(
                    f"KSPRecord home_department must be a Department enum, "
                    f"got {type(rec.home_department).__name__}={rec.home_department!r}."
                )
            ids.append(rec.ksp_id)
            embeddings.append(self._embedder.embed(rec.text))
            documents.append(rec.text)
            metadatas.append({
                "parent_doc_id":       rec.parent_doc_id,
                "source_file":         rec.source_file,
                "home_department":     rec.home_department.value,
                "clearance_level":     int(rec.clearance_level),
                # ChromaDB metadata only accepts scalars; JSON-encode the list.
                "allowed_departments": json.dumps(
                    [d.lower() for d in rec.allowed_departments]
                ),
            })

        coll = self._get_collection()
        self._vector_db.add_to_collection(
            coll,
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def query(
        self,
        query: str,
        *,
        user_clearance: int,
        user_department: str,
        top_k: int = 5,
    ) -> list[KSPMatch]:
        """Find the top-``top_k`` accessible KSPs nearest to ``query``.

        Pipeline:

          1. Embed ``query`` via the shared :class:`Embedder` (cached).
          2. Run a cosine kNN over the KSP collection with an oversample
             of ``max(top_k, 10)`` to leave headroom for RBAC drop-outs.
          3. Apply :func:`~domain.retrieval.rbac_filter.apply_zero_trust_filter`
             to the raw hits. KSPs the user cannot read are dropped —
             both the routing decision AND the *existence signal* are
             gated by RBAC.
          4. Slice the survivors to ``top_k`` and convert to
             :class:`KSPMatch` records.

        Args:
            query: The raw user query.
            user_clearance: Integer clearance level (0-3); typically
                read from ``ClearanceLevel.<level>.value``.
            user_department: The user's home department lowercased; the
                wildcard ``"all"`` is only valid on the *resource* side,
                never the requesting subject.
            top_k: How many accessible KSPs to return.

        Returns:
            Up to ``top_k`` matches, sorted by ascending cosine distance.

        Raises:
            PipelineError: invalid ``top_k`` or empty ``query``.
            VectorDBConnectionError: ChromaDB unreachable.
        """
        if not query or not query.strip():
            raise PipelineError("Empty query passed to KSPRouterIndex.query.")
        if top_k < 1:
            raise PipelineError(f"top_k must be >= 1; got {top_k}.")

        coll = self._get_collection()
        oversample: int = max(top_k * 2, 10)
        query_embedding: list[float] = self._embedder.embed(query)
        response = self._vector_db.query_collection(
            coll,
            query_embeddings=[query_embedding],
            n_results=oversample,
        )

        raw_chunks: list[dict[str, Any]] = self._unpack_response(response)
        accessible = apply_zero_trust_filter(
            raw_chunks, user_clearance, user_department,
        )
        return [self._to_match(c) for c in accessible[:top_k]]

    def delete_doc(self, parent_doc_id: str) -> None:
        """Remove every KSP whose ``parent_doc_id`` matches.

        Used by the ingestion pipeline when re-ingesting or deleting a
        document. Conventionally there is exactly one KSP per document,
        but the API tolerates multiple in case the design later evolves
        to multi-KSP-per-doc (e.g., one per major section).

        Raises:
            VectorDBConnectionError: ChromaDB unreachable.
        """
        coll = self._get_collection()
        existing = self._vector_db.get_documents_where(
            coll, where={"parent_doc_id": parent_doc_id}, include=["metadatas"],
        )
        ids = existing.get("ids") or []
        if not ids:
            return
        try:
            coll.delete(ids=ids)
        except Exception as exc:  # noqa: BLE001
            raise RoutingError(
                f"Failed to delete KSP records for parent_doc_id={parent_doc_id!r}: {exc}"
            ) from exc

    # ─────────────────────────── Internals ──────────────────────────────────
    def _get_collection(self) -> Any:
        """Lazy-create (with embedding-model stamp) and cache the collection."""
        if self._collection is None:
            self._collection = self._vector_db.get_or_create_collection(
                self._collection_name,
                embedding_model=self._settings.embedding_model_name,
            )
        return self._collection

    @staticmethod
    def _unpack_response(response: dict[str, Any]) -> list[dict[str, Any]]:
        """Convert a ChromaDB query response into RBAC-filter-ready chunks."""
        if not response.get("ids") or not response["ids"][0]:
            return []
        ids = response["ids"][0]
        docs = (response.get("documents") or [[]])[0]
        metas = (response.get("metadatas") or [[]])[0]
        dists = (response.get("distances") or [[]])[0]
        out: list[dict[str, Any]] = []
        for i, cid in enumerate(ids):
            out.append({
                "chunk_id": cid,
                "text":     docs[i] if i < len(docs) else "",
                "metadata": metas[i] if i < len(metas) else {},
                "distance": dists[i] if i < len(dists) else None,
            })
        return out

    @staticmethod
    def _to_match(chunk: dict[str, Any]) -> KSPMatch:
        """Project an unpacked KSP chunk into the public :class:`KSPMatch`."""
        meta: dict[str, Any] = chunk.get("metadata") or {}
        distance_val: Any = chunk.get("distance")
        distance: float = float(distance_val) if distance_val is not None else 1.0
        return KSPMatch(
            ksp_id=str(chunk.get("chunk_id", "")),
            home_department=str(meta.get("home_department", "")),
            parent_doc_id=str(meta.get("parent_doc_id", "")),
            source_file=str(meta.get("source_file", "")),
            distance=distance,
        )
