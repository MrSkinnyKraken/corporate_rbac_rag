"""Shared sentence-encoder client.

A *single* lazily-loaded HuggingFace model instance shared across the
application. The routing module and the retrieval module both need to
embed the user query — once for the KSP lookup, once for the per-
department chunk lookup — but loading the model twice would double the
~100MB resident memory cost and double the per-process startup time.

The :class:`Embedder` solves this with two layers of caching:

  * **Model loading** is deferred until the first call to
    :meth:`embed` and then memoised on the instance.
  * **Query embeddings** are memoised in a bounded FIFO cache so a
    single query string embedded by the router does not need to be
    re-embedded by the retriever a few milliseconds later.

A process-wide singleton :func:`get_embedder` is exposed and decorated
with ``functools.lru_cache`` so any module that calls it gets the same
instance and therefore shares both layers of the cache.

The class translates *any* failure from the underlying HuggingFace stack
into :class:`~core.exceptions.PipelineError` so callers don't need to
import ``torch``- or ``transformers``-specific exception types.
"""

from __future__ import annotations

from collections import OrderedDict
from functools import lru_cache
from typing import Any

from core.config import Settings, get_settings
from core.exceptions import PipelineError


_DEFAULT_QUERY_CACHE_SIZE: int = 128


class Embedder:
    """Lazy, cached wrapper around a HuggingFace sentence-encoder.

    Thread-safety: the underlying ``HuggingFaceEmbeddings`` is not
    documented as thread-safe; the application's FastAPI handlers run on
    a single asyncio loop so concurrent calls are serialised at the
    Python level. If concurrent embedding becomes a goal, switch the
    HF model to ``SentenceTransformer`` with a multi-process pool — this
    class would then own the pool and round-robin requests over it.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        cache_size: int = _DEFAULT_QUERY_CACHE_SIZE,
    ) -> None:
        """Build the embedder; do NOT load the model yet.

        Args:
            settings: Application settings (read ``embedding_model_name``
                and ``embedding_device``). If ``None``, fetched via
                :func:`~core.config.get_settings`.
            cache_size: Maximum number of (query → vector) entries held
                in the FIFO cache. Set to ``0`` to disable caching.
        """
        if cache_size < 0:
            raise PipelineError(
                f"cache_size must be >= 0; got {cache_size!r}."
            )
        self._settings: Settings = settings or get_settings()
        self._cache_size: int = cache_size
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._model: Any | None = None

    # ─────────────────────────── Public API ─────────────────────────────────
    @property
    def model_name(self) -> str:
        """The HuggingFace identifier this embedder is bound to."""
        return self._settings.embedding_model_name

    def embed(self, text: str) -> list[float]:
        """Return the dense embedding of ``text`` (cached for repeats).

        Args:
            text: Non-empty query text. Whitespace-only is rejected.

        Returns:
            A list of floats representing the query in the model's
            output space (384 dims for both ``all-MiniLM-L6-v2`` and
            ``paraphrase-multilingual-MiniLM-L12-v2``).

        Raises:
            PipelineError: empty input, model load failure, or any
                underlying HuggingFace / Torch exception during
                inference.
        """
        if not text or not text.strip():
            raise PipelineError("Empty text passed to Embedder.embed.")

        if self._cache_size > 0 and text in self._cache:
            # Move-to-end for FIFO semantics (oldest is at the head)
            self._cache.move_to_end(text, last=True)
            return self._cache[text]

        if self._model is None:
            self._load()
        try:
            vec: list[float] = self._model.embed_query(text)  # type: ignore[union-attr]
        except Exception as exc:  # noqa: BLE001  (HF surface is broad)
            raise PipelineError(
                f"Embedding query failed under model {self.model_name!r}: {exc}"
            ) from exc

        if self._cache_size > 0:
            self._cache[text] = vec
            if len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)  # evict oldest
        return vec

    def is_loaded(self) -> bool:
        """``True`` iff the underlying model has been materialised in memory.

        Used by the healthcheck and warm-up scripts to decide whether to
        force a load before serving traffic.
        """
        return self._model is not None

    def clear_cache(self) -> None:
        """Drop every cached embedding (model itself is kept)."""
        self._cache.clear()

    # ────────────────────────────── Internals ───────────────────────────────
    def _load(self) -> None:
        """Instantiate the underlying ``HuggingFaceEmbeddings`` model.

        Import is deferred to here so the rest of the codebase does not
        pull in ``torch`` and ``transformers`` at import time — the
        ~2s cold start happens at first query, not at process boot.
        """
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            self._model = HuggingFaceEmbeddings(
                model_name=self._settings.embedding_model_name,
                model_kwargs={"device": self._settings.embedding_device},
            )
        except Exception as exc:  # noqa: BLE001
            raise PipelineError(
                f"Failed to load embedding model "
                f"{self._settings.embedding_model_name!r}: {exc}"
            ) from exc


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    """Process-wide singleton :class:`Embedder`.

    Use this in any module that needs to embed queries — the model is
    loaded at most once per process, and the query cache is shared
    between callers so that the router's pre-flight embed of a query
    is reused by the retriever a few milliseconds later.

    The singleton is built from the current :func:`get_settings`
    result; both functions are ``lru_cache``-d, so a single change to
    the environment requires restarting the process to take effect.
    """
    return Embedder()
