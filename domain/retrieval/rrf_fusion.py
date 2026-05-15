"""Weighted Reciprocal Rank Fusion (RRF) for hybrid vector + lexical retrieval.

The classical formula by Cormack, Clarke and Buettcher (2009):

    RRF(d) = Σ_{r ∈ rankers}  α_r · 1 / (k + rank_r(d))

In our two-ranker setup:

    RRF(d) = α · 1 / (k + rank_vector(d))
           + (1 - α) · 1 / (k + rank_lexical(d))

The α parameter is the *vector weight*; ``1 - α`` is the *lexical weight*.
Both are non-negative and sum to 1, so the function is a convex combination
of the two rankers' contributions. The constant ``k`` (default 60) damps
the early-rank advantage so a chunk seen by both rankers — even at lower
ranks each — beats a chunk seen by only one ranker at the very top.

The α value is chosen *dynamically* by the intent classifier upstream
(:mod:`domain.routing`): exact-PII queries lean lexical (low α);
conceptual queries lean semantic (high α). This module accepts any
α ∈ [0, 1] and validates it.

Pure deterministic function — no I/O, no infrastructure dependencies.
"""

from __future__ import annotations

from typing import Any

from core.exceptions import PipelineError


_RRF_K_DEFAULT: int = 60


def calculate_dynamic_rrf(
    vector_rankings: list[dict[str, Any]],
    lexical_rankings: list[dict[str, Any]],
    alpha_vector: float = 0.5,
    *,
    rrf_k: int = _RRF_K_DEFAULT,
) -> list[dict[str, Any]]:
    """Fuse two ranked lists into a single ranking via weighted RRF.

    Args:
        vector_rankings:  Result list from the dense (Chroma) retriever, in
            rank order (best first). Each entry is a dict; a stable
            identifier is read from ``chunk_id``, ``id`` or, as a fallback,
            ``"<parent_doc_id>#<chunk_index>"``.
        lexical_rankings: Result list from the BM25 retriever, with the
            same shape and ordering convention.
        alpha_vector:     Weight assigned to the vector ranker. The lexical
            ranker receives ``1 - alpha_vector``. Must satisfy
            ``0.0 <= alpha_vector <= 1.0``.
        rrf_k:            The Cormack constant. Default 60 matches every
            Phase 2-3 experiment; raising it down-weights early ranks
            relative to late ones.

    Returns:
        The fused list of chunks, sorted by descending RRF score. Two
        keys are added to every returned dict (without mutating the
        input):
          * ``rrf_score``      — the final fused score
          * ``rrf_components`` — sub-scores ``{"vector": …, "lexical": …}``
                                 for diagnostics / debugging
        Returns an empty list iff both input rankings are empty.

    Raises:
        PipelineError: invalid ``alpha_vector`` or non-positive ``rrf_k``.
    """
    if not (0.0 <= alpha_vector <= 1.0):
        raise PipelineError(
            f"alpha_vector must be in [0.0, 1.0]; got {alpha_vector!r}."
        )
    if rrf_k <= 0:
        raise PipelineError(
            f"rrf_k must be a positive integer; got {rrf_k!r}."
        )

    fused: dict[str, dict[str, Any]] = {}

    # ── Vector contributions ────────────────────────────────────────────────
    for rank, chunk in enumerate(vector_rankings, start=1):
        cid = _chunk_identifier(chunk)
        score = alpha_vector * (1.0 / (rrf_k + rank))
        if cid in fused:
            fused[cid]["rrf_score"] += score
            fused[cid]["rrf_components"]["vector"] += score
        else:
            entry = dict(chunk)
            entry["rrf_score"] = score
            entry["rrf_components"] = {"vector": score, "lexical": 0.0}
            fused[cid] = entry

    # ── Lexical contributions ───────────────────────────────────────────────
    alpha_lexical = 1.0 - alpha_vector
    for rank, chunk in enumerate(lexical_rankings, start=1):
        cid = _chunk_identifier(chunk)
        score = alpha_lexical * (1.0 / (rrf_k + rank))
        if cid in fused:
            fused[cid]["rrf_score"] += score
            fused[cid]["rrf_components"]["lexical"] += score
        else:
            entry = dict(chunk)
            entry["rrf_score"] = score
            entry["rrf_components"] = {"vector": 0.0, "lexical": score}
            fused[cid] = entry

    return sorted(fused.values(), key=lambda c: -c["rrf_score"])


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _chunk_identifier(chunk: dict[str, Any]) -> str:
    """Extract a stable identifier from a chunk dict.

    Order of preference: top-level ``chunk_id``, top-level ``id``, and
    finally a composite of ``parent_doc_id`` and ``chunk_index`` taken
    from the chunk's metadata. The composite key fallback ensures
    fusion works even on chunks that the retriever forgot to label
    explicitly.
    """
    cid = chunk.get("chunk_id") or chunk.get("id")
    if cid:
        return str(cid)
    meta = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
    parent = meta.get("parent_doc_id") or chunk.get("parent_doc_id") or "unknown"
    idx = meta.get("chunk_index") if "chunk_index" in meta else chunk.get("chunk_index", "?")
    return f"{parent}#{idx}"
