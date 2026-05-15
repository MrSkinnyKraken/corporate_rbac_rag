"""Zero-Trust RBAC chunk filter.

Pure deterministic function with no I/O, no side effects, no infrastructure
dependencies. Applies the canonical 2D security predicate to a list of
candidate chunks and returns only those the user is authorised to read.

The predicate, evaluated chunk-by-chunk:

    chunk_is_allowed(chunk, user) :=
        chunk.clearance_level <= user_clearance
        AND
        ( user_department    in chunk.allowed_departments
          OR
          ALL_WILDCARD       in chunk.allowed_departments )

Malformed chunk metadata (missing required keys, wrong types, JSON-encoded
list that fails to parse) is treated as a *deny* — fail-secure. Callers
relying on the count of surviving chunks should treat ``len(out)`` as
authoritative; chunks that did not pass the filter are not returned in
any form.
"""

from __future__ import annotations

import json
from typing import Any

from core.security import ALLOWED_DEPARTMENT_WILDCARD


def apply_zero_trust_filter(
    candidate_chunks: list[dict[str, Any]],
    user_clearance: int,
    user_department: str,
) -> list[dict[str, Any]]:
    """Filter ``candidate_chunks`` by the Zero-Trust predicate.

    Args:
        candidate_chunks: A list of chunk dicts. Each chunk's metadata may
            live either at the top level or under a ``"metadata"`` key —
            both shapes are accepted.
        user_clearance:   The requesting user's clearance level as an int
            in ``[0, 3]``. Negative values are treated as
            "no access at all" and the function returns ``[]``.
        user_department:  The requesting user's department as a string
            (case-insensitive). Compared against the lowercased entries
            of each chunk's ``allowed_departments`` field.

    Returns:
        The subset of ``candidate_chunks`` the user may read, in the
        same relative order.

    Notes:
        ``allowed_departments`` is normalised to a lowercased ``list[str]``
        regardless of how it was stored upstream. ChromaDB does not accept
        list values in metadata, so the ingestion layer JSON-encodes the
        list as a string; this filter parses it back.
    """
    if user_clearance < 0:
        return []
    user_dept_lower: str = user_department.strip().lower()

    surviving: list[dict[str, Any]] = []
    for chunk in candidate_chunks:
        metadata = _extract_metadata(chunk)

        # ── Severity gate ────────────────────────────────────────────────
        cl = _safe_int(metadata.get("clearance_level"))
        if cl is None or cl > user_clearance:
            continue  # missing / malformed / over-clearance → deny

        # ── Scope gate ───────────────────────────────────────────────────
        allowed = _normalise_allowed_departments(metadata.get("allowed_departments"))
        if allowed is None:
            continue  # malformed → deny

        if user_dept_lower in allowed or ALLOWED_DEPARTMENT_WILDCARD in allowed:
            surviving.append(chunk)
    return surviving


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_metadata(chunk: dict[str, Any]) -> dict[str, Any]:
    """Return the chunk's metadata dict, supporting nested or flat shapes."""
    nested = chunk.get("metadata")
    return nested if isinstance(nested, dict) else chunk


def _safe_int(value: Any) -> int | None:
    """Coerce to int; return ``None`` on any failure."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalise_allowed_departments(value: Any) -> list[str] | None:
    """Normalise the ``allowed_departments`` field to a lowercased ``list[str]``.

    Accepts the field as:
      * ``list[str]``             — used as-is (lowercased).
      * ``"['hr','finance']"``    — JSON-encoded list (typical when read
                                    back from ChromaDB metadata).
      * ``"hr"``                  — a single department string (legacy
                                    Phase-1 metadata).
      * Anything else             — treated as malformed; returns ``None``
                                    so the caller deny-by-default.
    """
    if value is None:
        return None

    # 1) Already a list/tuple → use directly
    if isinstance(value, (list, tuple)):
        try:
            return [str(d).strip().lower() for d in value]
        except Exception:  # noqa: BLE001
            return None

    # 2) String → either a JSON-encoded list (preferred) or a single department
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                return None
            if not isinstance(parsed, list):
                return None
            try:
                return [str(d).strip().lower() for d in parsed]
            except Exception:  # noqa: BLE001
                return None
        # Plain single-dept string (legacy)
        return [stripped.lower()]

    # 3) Unknown type → deny
    return None
