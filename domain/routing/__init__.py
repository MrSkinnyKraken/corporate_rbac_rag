"""Routing domain: intent classifier, KSP Router-Index, hierarchical router.

The routing layer is the *upstream* of the retrieval layer. It accepts a
raw user query and produces the two values the retriever needs to make
its asymmetric ensemble decision:

  * ``target_departments`` — which physical ``chunks_<dept>`` collections
    should the retriever query.
  * ``alpha``               — the vector-branch weight to use in RRF.

The composition is hierarchical:

    query ──▶ intent classifier  ──▶  alpha
          ──▶ KSP Router-Index   ──▶  target_departments

Both decisions are independent of one another and run cheaply on the
shared :class:`~infrastructure.embedder.Embedder`. The router returns a
single :class:`RoutingDecision` consumed by the application orchestrator.
"""

from domain.routing.intent_classifier import (
    QueryIntent,
    classify_intent,
    dynamic_alpha,
)
from domain.routing.ksp_index import KSPMatch, KSPRecord, KSPRouterIndex
from domain.routing.router import HierarchicalRouter, RoutingDecision

__all__: list[str] = [
    "QueryIntent",
    "classify_intent",
    "dynamic_alpha",
    "KSPRouterIndex",
    "KSPRecord",
    "KSPMatch",
    "HierarchicalRouter",
    "RoutingDecision",
]
