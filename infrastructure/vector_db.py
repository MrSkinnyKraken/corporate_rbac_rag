"""ChromaDB adapter — thin wrapper around :class:`chromadb.HttpClient`.

This module provides :class:`ChromaDBClient`, a technical adapter that:

* Builds the ``HttpClient`` from :class:`~core.config.Settings` (host, port,
  ssl, tenant, database) — no hard-coded values anywhere.
* Exposes the small set of ChromaDB operations the application actually
  needs (collections create / get / list / delete, vector add, vector query).
* Translates *transport-level* failures (server unreachable, connection
  refused, timeout) into :class:`~core.exceptions.VectorDBConnectionError`,
  preserving the original cause via ``raise ... from e`` chaining.
* Lets *domain-level* ChromaDB errors propagate unchanged (e.g., a
  ``NotFoundError`` when querying a non-existent collection): wrapping those
  as connection errors would lose semantic information.

No business logic lives here — no RBAC checks, no embedding computation, no
chunking. Those belong to :mod:`domain` and :mod:`application`.
"""

from __future__ import annotations

from typing import Any, Sequence

import chromadb
from chromadb.api.models.Collection import Collection

from core.config import Settings, get_settings
from core.exceptions import ConfigurationError, VectorDBConnectionError


_EMBEDDING_MODEL_META_KEY: str = "embedding_model"


# Network-level errors that all map to "the database is unreachable".
# `requests.exceptions.ConnectionError`, used internally by chromadb's
# HttpClient, is a subclass of `OSError`, so this tuple covers it.
_NETWORK_ERRORS: tuple[type[BaseException], ...] = (
    ConnectionError,
    OSError,
    TimeoutError,
)


class ChromaDBClient:
    """Adapter around :class:`chromadb.HttpClient`.

    The adapter is lightweight and stateless apart from the underlying HTTP
    client. ChromaDB 0.4.x's ``HttpClient`` is itself lazy — the actual
    network connection is only opened on the first method call (heartbeat,
    list, query …) — so any constructor failure typically points at a
    misconfigured tenant / database, not at the server being down.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """Build the ChromaDB client from the application settings.

        Args:
            settings: Application settings to read host/port/tenant from. If
                ``None``, fetched via :func:`~core.config.get_settings`.

        Raises:
            VectorDBConnectionError: if the underlying client cannot be
                instantiated due to a network-level error (rare, since the
                client is lazy).
        """
        self._settings: Settings = settings or get_settings()
        # ChromaDB 0.4.x's HttpClient is *eager*: its constructor issues an
        # HTTP request to validate the tenant/database. When the server is
        # down the underlying `requests.exceptions.ConnectionError` is
        # re-wrapped as a plain `ValueError("Could not connect to a Chroma
        # server…")` — outside the standard network-error hierarchy. A
        # broad catch in the constructor is therefore justified: any failure
        # here is, by construction, a "cannot reach Chroma" condition.
        try:
            self._client: chromadb.api.ClientAPI = chromadb.HttpClient(
                host=self._settings.chroma_host,
                port=self._settings.chroma_port,
                ssl=self._settings.chroma_ssl,
                tenant=self._settings.chroma_tenant,
                database=self._settings.chroma_database,
            )
        except Exception as exc:  # noqa: BLE001  (intentional, see comment above)
            raise VectorDBConnectionError(
                f"Cannot initialise ChromaDB client at "
                f"{self._settings.chroma_base_url}: {exc}"
            ) from exc

    # ------------------------------------------------------------ Health
    def health_check(self) -> bool:
        """Return ``True`` iff the ChromaDB server responds to a heartbeat.

        Never raises — used by the application's startup self-check.
        """
        try:
            self._client.heartbeat()
            return True
        except Exception:  # noqa: BLE001  (intentional broad catch)
            return False

    # -------------------------------------------------------- Collections
    def get_or_create_collection(
        self,
        name: str,
        *,
        metadata: dict[str, Any] | None = None,
        embedding_model: str | None = None,
    ) -> Collection:
        """Return the collection ``name``, creating it if it does not exist.

        Args:
            name: ChromaDB collection name (3-63 chars, alphanumerics +
                ``_-``). The application is expected to enforce its own
                naming convention (e.g., one collection per real
                department: ``chunks_finance``, ``chunks_hr`` …).
            metadata: Optional collection-level metadata. Defaults to
                ``{"hnsw:space": "cosine"}`` to keep the distance metric
                consistent with every notebook experiment in Phases 1-3.
            embedding_model: Optional embedding-model identifier to stamp
                onto the collection (HuggingFace name, e.g.
                ``sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2``).
                When provided, the adapter enforces a *fail-fast* model
                guard: on creation the model name is recorded in
                collection metadata; on subsequent opens the stored value
                is compared against the requested one and a
                :class:`~core.exceptions.ConfigurationError` is raised on
                mismatch. Empty/legacy collections without a stamp are
                lazily stamped on first access (no-op if already stored).

        Returns:
            The :class:`Collection` handle.

        Raises:
            VectorDBConnectionError: if the server is unreachable.
            ConfigurationError: if ``embedding_model`` is supplied and the
                stored value on the collection differs — indicates the
                application's ``EMBEDDING_MODEL_NAME`` no longer matches
                the vectors persisted in this collection. Fix by either
                re-ingesting the corpus with the current model or
                reverting the setting.
        """
        # ChromaDB 0.4.x quirk: calling
        # ``client.get_or_create_collection(name, metadata=...)`` on an
        # *existing* collection silently REPLACES its server-side metadata
        # with the new dict — wiping any previously-stored
        # ``embedding_model`` stamp before the guard has a chance to verify
        # it. We therefore split the path: list collections to decide,
        # then either ``get_collection`` (touches no metadata) or
        # ``create_collection`` (writes the stamp atomically on creation).
        base_metadata: dict[str, Any] = dict(metadata) if metadata else {"hnsw:space": "cosine"}
        if embedding_model is not None:
            base_metadata[_EMBEDDING_MODEL_META_KEY] = embedding_model

        try:
            existing_names = {c.name for c in self._client.list_collections()}
        except _NETWORK_ERRORS as exc:
            raise VectorDBConnectionError(
                f"ChromaDB unreachable during list_collections (for "
                f"get_or_create_collection({name!r})): {exc}"
            ) from exc

        if name in existing_names:
            try:
                coll = self._client.get_collection(name=name)
            except _NETWORK_ERRORS as exc:
                raise VectorDBConnectionError(
                    f"ChromaDB unreachable during get_collection({name!r}): {exc}"
                ) from exc
        else:
            try:
                coll = self._client.create_collection(
                    name=name,
                    metadata=base_metadata,
                )
            except _NETWORK_ERRORS as exc:
                raise VectorDBConnectionError(
                    f"ChromaDB unreachable during create_collection({name!r}): {exc}"
                ) from exc

        if embedding_model is not None:
            self._enforce_embedding_model_stamp(coll, embedding_model)
        return coll

    def _enforce_embedding_model_stamp(
        self,
        collection: Collection,
        expected_model: str,
    ) -> None:
        """Verify ``collection`` was built with ``expected_model`` (lazy-stamp if absent).

        The runtime guard for the multilingual-embedding follow-up: a
        collection persists not only vectors but also the *name* of the
        encoder that produced them, recorded in
        ``collection.metadata["embedding_model"]``. The first time a
        legacy or freshly-created collection is opened with an explicit
        ``embedding_model`` argument, the stamp is written. Subsequent
        opens verify that the stamp matches and raise
        :class:`ConfigurationError` if it does not.

        Critically, the function re-reads the metadata directly from the
        server via ``client.get_collection`` rather than trusting the
        ``Collection.metadata`` attribute on the handle returned by
        ``get_or_create_collection``: in ChromaDB 0.4.x that attribute
        echoes the *requested* metadata for an already-existing
        collection, which would defeat the mismatch check.

        Args:
            collection: The handle returned by ``get_or_create_collection``.
            expected_model: The model identifier that the application is
                currently configured to use.

        Raises:
            ConfigurationError: stored ``embedding_model`` differs from
                ``expected_model`` — silently mixing would produce
                geometrically incompatible vectors and silently destroy
                retrieval quality.
            VectorDBConnectionError: ``Collection.modify`` is an HTTP call
                under the hood; transport failures during the lazy stamp
                are translated.
        """
        try:
            authoritative = self._client.get_collection(name=collection.name)
        except _NETWORK_ERRORS as exc:
            raise VectorDBConnectionError(
                f"ChromaDB unreachable while reading metadata of "
                f"{collection.name!r}: {exc}"
            ) from exc

        current_meta: dict[str, Any] = dict(authoritative.metadata or {})
        stored = current_meta.get(_EMBEDDING_MODEL_META_KEY)
        if stored is None:
            # ChromaDB 0.4.x rejects modify() calls whose metadata dict
            # contains any "hnsw:*" key — even with the original value —
            # because the distance function is immutable once the
            # collection is created, and modify() replaces the metadata
            # dict wholesale (no merge). Stripping the hnsw:* keys here
            # is empirically safe: the distance function is persisted
            # inside the HNSW index itself, so cosine ranking continues
            # to work after the stamp even though "hnsw:space" no longer
            # appears in user-visible metadata.
            mutable_meta: dict[str, Any] = {
                k: v for k, v in current_meta.items() if not k.startswith("hnsw:")
            }
            mutable_meta[_EMBEDDING_MODEL_META_KEY] = expected_model
            try:
                collection.modify(metadata=mutable_meta)
            except _NETWORK_ERRORS as exc:
                raise VectorDBConnectionError(
                    f"ChromaDB unreachable while stamping embedding model on "
                    f"{collection.name!r}: {exc}"
                ) from exc
            return
        if stored != expected_model:
            raise ConfigurationError(
                f"Embedding-model mismatch on collection {collection.name!r}: "
                f"stored={stored!r}, current={expected_model!r}. "
                f"The vectors in this collection were produced by a different "
                f"encoder and are geometrically incompatible with queries from "
                f"the current model. Either re-ingest the corpus with "
                f"EMBEDDING_MODEL_NAME={expected_model!r}, or revert the setting "
                f"to {stored!r} to keep using the existing vectors."
            )

    def list_collections(self) -> list[str]:
        """Return the names of every collection currently in the database.

        Raises:
            VectorDBConnectionError: if the server is unreachable.
        """
        try:
            return [c.name for c in self._client.list_collections()]
        except _NETWORK_ERRORS as exc:
            raise VectorDBConnectionError(
                f"ChromaDB unreachable during list_collections: {exc}"
            ) from exc

    def delete_collection(self, name: str) -> None:
        """Delete the collection ``name``.

        Domain-level ``NotFoundError`` (from ``chromadb.errors``) propagates
        unchanged when the collection does not exist; only transport
        failures are translated.

        Raises:
            VectorDBConnectionError: if the server is unreachable.
        """
        try:
            self._client.delete_collection(name=name)
        except _NETWORK_ERRORS as exc:
            raise VectorDBConnectionError(
                f"ChromaDB unreachable during delete_collection({name!r}): {exc}"
            ) from exc

    # ------------------------------------------------------ Vector ops
    def add_to_collection(
        self,
        collection: Collection,
        *,
        ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        documents: Sequence[str],
        metadatas: Sequence[dict[str, Any]],
    ) -> None:
        """Add vectors and their associated payloads to ``collection``.

        Args:
            collection: A :class:`Collection` handle obtained from
                :meth:`get_or_create_collection`.
            ids: One unique id per vector.
            embeddings: One vector (sequence of floats) per id.
            documents: One source-text string per id.
            metadatas: One scalar-only metadata dict per id (ChromaDB does
                not accept lists / nested objects as metadata values).

        Raises:
            VectorDBConnectionError: if the server is unreachable.
        """
        try:
            collection.add(
                ids=list(ids),
                embeddings=[list(e) for e in embeddings],
                documents=list(documents),
                metadatas=list(metadatas),
            )
        except _NETWORK_ERRORS as exc:
            raise VectorDBConnectionError(
                f"ChromaDB unreachable during add_to_collection({collection.name!r}): {exc}"
            ) from exc

    def query_collection(
        self,
        collection: Collection,
        *,
        query_embeddings: Sequence[Sequence[float]],
        n_results: int = 5,
        where: dict[str, Any] | None = None,
        include: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Run a nearest-neighbour query against ``collection``.

        Args:
            collection: A :class:`Collection` handle.
            query_embeddings: One or more query vectors. Each result tier in
                the response will have one entry per query vector.
            n_results: How many nearest neighbours to return per query.
            where: Optional ChromaDB ``where`` filter on metadata. The
                adapter does not validate the filter — invalid filters
                surface as native ChromaDB errors, not connection errors.
            include: Which response components to fetch. Defaults to
                ``["metadatas", "distances", "documents"]``, which is what
                every Phase 1-3 retrieval call needed.

        Returns:
            The raw response dict from ChromaDB. Keys typically include
            ``ids``, ``distances``, ``metadatas``, ``documents`` (each a
            list-of-lists indexed by query then by rank).

        Raises:
            VectorDBConnectionError: if the server is unreachable.
        """
        try:
            return collection.query(
                query_embeddings=[list(qe) for qe in query_embeddings],
                n_results=n_results,
                where=where,
                include=list(include) if include else ["metadatas", "distances", "documents"],
            )
        except _NETWORK_ERRORS as exc:
            raise VectorDBConnectionError(
                f"ChromaDB unreachable during query_collection({collection.name!r}): {exc}"
            ) from exc

    def get_documents_where(
        self,
        collection: Collection,
        *,
        where: dict[str, Any],
        include: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Fetch every document in ``collection`` whose metadata matches ``where``.

        Used by the retrieval layer to reconstruct parent context: given a
        winning chunk's ``parent_doc_id``, this method fetches every
        sibling chunk of the same source document so the orchestrator can
        stitch them back into the LLM context (Phase 2 Step 3 logic, now
        executed against a live ChromaDB collection rather than a JSON
        snapshot).

        Args:
            collection: A :class:`Collection` handle.
            where:      ChromaDB ``where`` filter on metadata
                (e.g., ``{"parent_doc_id": "uuid-…"}``).
            include:    Which response components to fetch. Defaults to
                ``["metadatas", "documents"]``.

        Returns:
            The raw response dict from ChromaDB (keys typically
            ``ids``, ``metadatas``, ``documents``).

        Raises:
            VectorDBConnectionError: if the server is unreachable.
        """
        try:
            return collection.get(
                where=where,
                include=list(include) if include else ["metadatas", "documents"],
            )
        except _NETWORK_ERRORS as exc:
            raise VectorDBConnectionError(
                f"ChromaDB unreachable during get_documents_where on "
                f"{collection.name!r}: {exc}"
            ) from exc

    def get_all_documents(
        self,
        collection: Collection,
        *,
        include: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Fetch every document currently stored in ``collection``.

        Used by the ingestion orchestrator to refit the per-department
        BM25 index from scratch after every commit (Chroma is the
        source of truth; BM25 is a derived index). For a few-thousand-
        chunk demo corpus this is fine; for large-scale deployments,
        consider an incremental refit strategy.

        Args:
            collection: A :class:`Collection` handle.
            include:    Which response components to fetch. Defaults to
                        ``["metadatas", "documents"]``.

        Returns:
            The raw response dict from ChromaDB (keys ``ids``,
            ``metadatas``, ``documents``).

        Raises:
            VectorDBConnectionError: if the server is unreachable.
        """
        try:
            return collection.get(
                include=list(include) if include else ["metadatas", "documents"],
            )
        except _NETWORK_ERRORS as exc:
            raise VectorDBConnectionError(
                f"ChromaDB unreachable during get_all_documents on "
                f"{collection.name!r}: {exc}"
            ) from exc

    def delete_by_metadata(
        self,
        collection: Collection,
        *,
        where: dict[str, Any],
    ) -> int:
        """Delete every chunk in ``collection`` matching ``where``.

        Used to make re-ingestion idempotent: before writing the new
        chunks for a document, the orchestrator deletes any existing
        chunks for the same ``parent_doc_id`` so the corpus does not
        accumulate duplicates across retries.

        Args:
            collection: A :class:`Collection` handle.
            where:      ChromaDB ``where`` filter on metadata
                        (e.g. ``{"parent_doc_id": "uuid-…"}``).

        Returns:
            The number of chunks that were deleted.

        Raises:
            VectorDBConnectionError: if the server is unreachable.
        """
        try:
            matching = collection.get(where=where, include=["metadatas"])
        except _NETWORK_ERRORS as exc:
            raise VectorDBConnectionError(
                f"ChromaDB unreachable during delete_by_metadata lookup on "
                f"{collection.name!r}: {exc}"
            ) from exc
        ids = matching.get("ids") or []
        if not ids:
            return 0
        try:
            collection.delete(ids=list(ids))
        except _NETWORK_ERRORS as exc:
            raise VectorDBConnectionError(
                f"ChromaDB unreachable during delete on "
                f"{collection.name!r}: {exc}"
            ) from exc
        return len(ids)

    # ------------------------------------------------------ Introspection
    def collection_count(self, collection: Collection) -> int:
        """Return the number of vectors currently stored in ``collection``.

        Raises:
            VectorDBConnectionError: if the server is unreachable.
        """
        try:
            return int(collection.count())
        except _NETWORK_ERRORS as exc:
            raise VectorDBConnectionError(
                f"ChromaDB unreachable during collection_count({collection.name!r}): {exc}"
            ) from exc
