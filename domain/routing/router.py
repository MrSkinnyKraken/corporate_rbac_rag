"""Hierarchical Router — composes intent classification + KSP routing.

The router is the *single entry point* of the routing layer. It accepts
a raw user query plus the user's RBAC context and returns a
:class:`RoutingDecision` that the application orchestrator hands to
:meth:`~domain.retrieval.ensemble_retriever.AsymmetricEnsembleRetriever.retrieve_secure_context`.

Decision pipeline::

    query ─┬─▶  classify_intent          ──▶ alpha               ┐
           │                                                      ├─▶ RoutingDecision
           └─▶  KSPRouterIndex.query     ──▶ target_departments  ┘
                (RBAC-filtered)

Both branches are independent and stateless; the router does not call
the retriever or the LLM. This keeps routing *cheap* (one cached embed +
one kNN over a tiny KSP collection + a regex pass), so the layer can run
in front of every request without becoming a hot-path bottleneck.

The router does **not raise** when the KSP collection is empty or no
accessible KSPs match — it returns ``RoutingDecision.is_empty=True`` and
the application layer decides whether to short-circuit ("no information
available in your scope") or fall through to the retriever, which will
itself return an empty :class:`~domain.retrieval.ensemble_retriever.RetrievalResult`
with the explicit ``[NO ACCESSIBLE CONTEXT]`` marker.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from core.exceptions import PipelineError
from core.security import ClearanceLevel, Department
from domain.routing.intent_classifier import (
    QueryIntent,
    classify_intent,
    alpha_for,
)
from domain.routing.ksp_index import KSPMatch, KSPRouterIndex


# ─────────────────────────────────────────────────────────────────────────────
# Decision dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class RoutingDecision:
    """The complete output of :meth:`HierarchicalRouter.route`.

    Attributes:
        target_departments: Department string values to feed
            ``AsymmetricEnsembleRetriever.retrieve_secure_context``.
            Empty list iff no KSPs matched or all were filtered by RBAC.
        alpha: The vector-branch weight to feed RRF.
        intent: The :class:`QueryIntent` the classifier assigned.
        ksp_matches: Every KSP hit considered (already RBAC-filtered),
            in distance order. Includes the matches that did not make
            it into ``target_departments`` because of the
            ``max_departments`` cap — useful for audit logs and
            "did you mean department X" UI hints.
    """

    target_departments: list[str] = field(default_factory=list)
    alpha: float = 0.5
    intent: QueryIntent = QueryIntent.MIXED
    ksp_matches: list[KSPMatch] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        """``True`` iff no target departments could be selected."""
        return not self.target_departments


# ─────────────────────────────────────────────────────────────────────────────
# Router
# ─────────────────────────────────────────────────────────────────────────────

class HierarchicalRouter:
    """Compose :func:`classify_intent` and :class:`KSPRouterIndex` into one call."""

    def __init__(
        self,
        ksp_index: KSPRouterIndex,
        *,
        top_k_ksps: int = 5,
        max_departments: int = 2,
    ) -> None:
        """Initialise the router.

        Args:
            ksp_index: The KSP Router-Index to query for department
                selection. Injected (not constructed inside) so unit
                tests can pass a fake index.
            top_k_ksps: How many accessible KSPs to fetch before
                department extraction. Default 5 — small because the
                KSP collection is small (one record per ingested
                document) and we only need the *top* few to identify
                the dominant home departments.
            max_departments: Hard cap on how many distinct departments
                the router promotes to ``target_departments``. Default
                2 — matches the oversample budget the retriever
                allocates per branch. Set to 1 for strict single-dept
                routing; > 3 starts diluting the per-dept candidate
                pool to noise.

        Raises:
            PipelineError: invalid ``top_k_ksps`` or ``max_departments``.
        """
        if top_k_ksps < 1:
            raise PipelineError(f"top_k_ksps must be >= 1; got {top_k_ksps}.")
        if max_departments < 1:
            raise PipelineError(
                f"max_departments must be >= 1; got {max_departments}."
            )
        self._ksp_index: KSPRouterIndex = ksp_index
        self._top_k_ksps: int = top_k_ksps
        self._max_departments: int = max_departments

    def route(
        self,
        query: str,
        user_clearance: ClearanceLevel,
        user_department: Department,
    ) -> RoutingDecision:
        """Produce the routing decision for one query.

        Args:
            query: The raw user query (must be non-empty).
            user_clearance: The requesting user's clearance level.
            user_department: The requesting user's home department.
                Must be a :class:`Department` enum value (the "all"
                wildcard is invalid on the subject side).

        Returns:
            A populated :class:`RoutingDecision`. ``is_empty=True`` when
            no department survived KSP+RBAC; the orchestrator can then
            choose to short-circuit or hand the empty target list to
            the retriever (which itself returns ``is_empty=True`` with
            the ``[NO ACCESSIBLE CONTEXT]`` marker — defence in depth).

        Raises:
            PipelineError: empty query or invalid argument types.
        """
        if not query or not query.strip():
            raise PipelineError("Empty query passed to HierarchicalRouter.route.")
        if not isinstance(user_clearance, ClearanceLevel):
            raise PipelineError(
                f"user_clearance must be a ClearanceLevel; got "
                f"{type(user_clearance).__name__}={user_clearance!r}."
            )
        if not isinstance(user_department, Department):
            raise PipelineError(
                f"user_department must be a Department enum (not "
                f"AllowedDepartment / str); got "
                f"{type(user_department).__name__}={user_department!r}."
            )

        # 1) Intent → alpha (pure regex, ~microseconds)
        intent: QueryIntent = classify_intent(query)
        alpha: float = alpha_for(intent)

        # 2) KSP lookup → target_departments
        # Short-circuit when the KSP collection is empty (fresh deployment
        # before any ingestion has run).
        if not self._ksp_index.has_entries():
            return RoutingDecision(
                target_departments=[],
                alpha=alpha,
                intent=intent,
                ksp_matches=[],
            )

        matches: list[KSPMatch] = self._ksp_index.query(
            query,
            user_clearance=int(user_clearance),
            user_department=user_department.value,
            top_k=self._top_k_ksps,
        )

        # 3) Promote distinct home_department values, preserving rank order.
        target_departments: list[str] = self._extract_departments(matches)

        return RoutingDecision(
            target_departments=target_departments,
            alpha=alpha,
            intent=intent,
            ksp_matches=matches,
        )

    # ─────────────────────────── Internals ──────────────────────────────────
    def _extract_departments(self, matches: list[KSPMatch]) -> list[str]:
        """Pull distinct ``home_department`` values from ``matches``.

        The order of first appearance is preserved (so the closest KSP's
        department wins ties) and the list is truncated to
        ``self._max_departments``. Empty or whitespace department
        strings are filtered out defensively.
        """
        seen: set[str] = set()
        out: list[str] = []
        for m in matches:
            dept = (m.home_department or "").strip().lower()
            if not dept or dept in seen:
                continue
            seen.add(dept)
            out.append(dept)
            if len(out) >= self._max_departments:
                break
        return out
