"""Application layer: orchestrators that compose domain primitives with infra.

The packages under :mod:`application` are the only layer authorised to
wire domain components together with infrastructure adapters and LLM
calls. They are the "use cases" of the system:

* :mod:`application.ingestion_app` — accept a document upload, propose
  metadata via the LLM, await user confirmation, then write to the
  three storage backends (chunks_<dept>, BM25, KSP).
* :mod:`application.query_app`     — accept a query, route it, retrieve
  context, and orchestrate the LLM generation. *(Phase 4 Step 8.)*

The application layer is where the demo user fixture (see
[[demo-user-fixture]] project memory) materialises as a :class:`User`
object handed in by the api adapter and propagated to audit logs.
"""

from application.document_classifier import LLMDocumentClassifier
from application.ingestion_app import (
    IngestionApp,
    IngestionProposal,
    IngestionReport,
    ProposedMetadata,
)
from application.query_app import Citation, QueryApp, QueryResponse

__all__: list[str] = [
    "LLMDocumentClassifier",
    "IngestionApp",
    "IngestionProposal",
    "IngestionReport",
    "ProposedMetadata",
    "QueryApp",
    "QueryResponse",
    "Citation",
]
