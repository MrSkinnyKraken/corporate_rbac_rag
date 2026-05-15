"""BM25 lexical-retrieval adapter with on-disk persistence.

Each *index name* maps 1:1 to a department (``chunks_finance``,
``chunks_hr`` …) so the retrieval layer queries only the indexes of the
routed departments. This per-department partitioning is the primary
mitigation for the linear post-filter cost identified during Phase 2: at
query time we score against a small per-department corpus rather than the
full ingest, then RBAC-filter inline. The result is constant-ish latency
even as the global corpus scales to thousands of documents.

Indexes are pickled to ``settings.bm25_index_dir`` so they survive
process restarts. Loading is lazy — :meth:`BM25Client.search` deserialises
on first use and caches the result in memory for the lifetime of the
process.

Pure infrastructure: no business logic, no RBAC, no chunking. The
adapter only knows how to fit, persist, load, and query a BM25 index.
"""

from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from core.config import Settings, get_settings
from core.exceptions import LexicalIndexError


# A simple Unicode-aware word tokenizer; matches the Phase 1-3 notebook
# behaviour so an index built in production scores identically to its
# notebook equivalent.
_TOKEN_RE: re.Pattern[str] = re.compile(r"\w+", re.UNICODE)

# Names must be filesystem-safe and short; this matches the constraint
# ChromaDB itself imposes on collection names (3-63 chars, alnum + _-).
_NAME_RE: re.Pattern[str] = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{1,62}$")


def _tokenize(text: str) -> list[str]:
    """Lowercased word tokens — consistent with the notebook implementation."""
    return _TOKEN_RE.findall(text.lower())


@dataclass(frozen=True, slots=True)
class IndexedChunk:
    """A chunk as stored inside a BM25 index, with its lookup metadata.

    Attributes:
        chunk_id: A globally-unique stable identifier (the same one used
            in the matching ChromaDB collection).
        text:     Raw chunk text. The BM25 index is built from a tokenised
            view of this string; storing the original lets the retriever
            return the chunk verbatim without re-fetching from ChromaDB.
        metadata: Scalar-only metadata dict (clearance_level, department,
            allowed_departments, parent_doc_id, chunk_index, …). Lists
            stored as JSON strings — same convention as in ChromaDB.
    """

    chunk_id: str
    text: str
    metadata: dict[str, Any]


@dataclass
class _Index:
    """In-memory pair (BM25Okapi, parallel chunks list).

    Internal — never exposed outside the module.
    """

    bm25: BM25Okapi
    chunks: list[IndexedChunk]


class BM25Client:
    """Adapter around :class:`rank_bm25.BM25Okapi` with per-collection persistence.

    Every public method that touches the filesystem or the underlying
    ``BM25Okapi`` translates failures into
    :class:`~core.exceptions.LexicalIndexError`, preserving the original
    cause via ``raise ... from e`` chaining.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialise the adapter.

        Args:
            settings: Application settings. If ``None``, fetched via
                :func:`~core.config.get_settings`. The directory in
                ``settings.bm25_index_dir`` is created on first access.
        """
        self._settings: Settings = settings or get_settings()
        self._dir: Path = self._settings.bm25_index_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, _Index] = {}

    # ─────────────────────────── Public API ─────────────────────────────────
    def fit(self, name: str, chunks: list[IndexedChunk]) -> None:
        """Build a fresh BM25 index for ``name`` from ``chunks``.

        Replaces any in-memory index with the same name. To persist the
        index across restarts, call :meth:`save_index` afterwards.

        Args:
            name:   Index identifier (e.g., ``"chunks_finance"``). Must be
                3-63 chars, alphanumerics with underscores or hyphens.
            chunks: At least one :class:`IndexedChunk` to index.

        Raises:
            LexicalIndexError: invalid name, empty corpus, or BM25
                construction failure.
        """
        self._validate_name(name)
        if not chunks:
            raise LexicalIndexError(
                f"Cannot fit empty BM25 index for {name!r}: no chunks provided."
            )
        try:
            tokens: list[list[str]] = [_tokenize(c.text) for c in chunks]
            bm25 = BM25Okapi(tokens)
        except Exception as exc:  # noqa: BLE001
            raise LexicalIndexError(
                f"Failed to fit BM25 index for {name!r}: {exc}"
            ) from exc
        self._cache[name] = _Index(bm25=bm25, chunks=list(chunks))

    def save_index(self, name: str) -> None:
        """Pickle the in-memory index ``name`` to disk."""
        if name not in self._cache:
            raise LexicalIndexError(
                f"No in-memory index for {name!r}; call fit() before save_index()."
            )
        path = self._index_path(name)
        try:
            with path.open("wb") as fp:
                pickle.dump(self._cache[name], fp, protocol=pickle.HIGHEST_PROTOCOL)
        except OSError as exc:
            raise LexicalIndexError(
                f"Cannot write BM25 index file {path}: {exc}"
            ) from exc

    def load_index(self, name: str) -> None:
        """Deserialise the on-disk index ``name`` into memory."""
        path = self._index_path(name)
        if not path.exists():
            raise LexicalIndexError(
                f"BM25 index file not found: {path}. "
                f"Run the ingestion pipeline to build it."
            )
        try:
            with path.open("rb") as fp:
                obj = pickle.load(fp)
        except Exception as exc:  # noqa: BLE001  (pickle can raise many types)
            raise LexicalIndexError(
                f"Failed to unpickle BM25 index from {path}: {exc}. "
                f"The file may be corrupted or built by an incompatible version."
            ) from exc
        if not isinstance(obj, _Index):
            raise LexicalIndexError(
                f"Pickled object at {path} is not a BM25 _Index "
                f"(got {type(obj).__name__})."
            )
        self._cache[name] = obj

    def has_index(self, name: str) -> bool:
        """Return ``True`` iff an index for ``name`` exists in memory or on disk."""
        if name in self._cache:
            return True
        try:
            return self._index_path(name).exists()
        except LexicalIndexError:
            return False

    def list_indexes(self) -> list[str]:
        """List the names of every index file present on disk, sorted."""
        return sorted(p.stem for p in self._dir.glob("*.pkl"))

    def search(
        self,
        name: str,
        query: str,
        top_k: int = 30,
    ) -> list[tuple[IndexedChunk, float]]:
        """Score the index for ``name`` against ``query`` and return top-k.

        Lazy-loads the index on first call so processes that never query a
        given collection do not pay the deserialisation cost.

        Args:
            name:  Index identifier — must already exist via :meth:`fit` /
                :meth:`save_index`.
            query: Free-text query.
            top_k: Maximum number of (chunk, score) pairs to return.
                BM25 scores ≤ 0 (non-matching documents) are dropped, so
                the actual list may be shorter than ``top_k``.

        Returns:
            A list of ``(IndexedChunk, score)`` tuples in descending score
            order.

        Raises:
            LexicalIndexError: index missing, ``top_k`` invalid, or BM25
                scoring failure.
        """
        if top_k < 1:
            raise LexicalIndexError(f"top_k must be >= 1, got {top_k}.")
        if name not in self._cache:
            self.load_index(name)
        idx = self._cache[name]
        try:
            scores = idx.bm25.get_scores(_tokenize(query))
        except Exception as exc:  # noqa: BLE001
            raise LexicalIndexError(
                f"BM25 scoring failed for index {name!r}: {exc}"
            ) from exc
        order: np.ndarray = np.argsort(scores)[::-1][:top_k]
        result: list[tuple[IndexedChunk, float]] = []
        for i in order:
            score = float(scores[i])
            if score <= 0.0:  # rank_bm25 returns 0 for non-matching docs
                break
            result.append((idx.chunks[int(i)], score))
        return result

    def evict(self, name: str) -> None:
        """Drop the in-memory index for ``name`` (frees RAM)."""
        self._cache.pop(name, None)

    # ──────────────────────────── Internals ─────────────────────────────────
    def _validate_name(self, name: str) -> None:
        if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
            raise LexicalIndexError(
                f"Invalid index name {name!r}: must match {_NAME_RE.pattern}."
            )

    def _index_path(self, name: str) -> Path:
        self._validate_name(name)
        return self._dir / f"{name}.pkl"
