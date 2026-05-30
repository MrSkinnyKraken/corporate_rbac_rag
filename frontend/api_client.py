"""Thin HTTP client around the FastAPI backend.

Every method maps 1:1 to an endpoint and returns the raw
:class:`httpx.Response` so the Streamlit UI can branch on
``status_code`` and render either the success body or the structured
error envelope (``{error, detail, context}``) produced by
:mod:`api.errors`.

A single :class:`httpx.Client` is owned per :class:`ApiClient` instance
for connection pooling. The Streamlit app constructs one client per
session via :func:`streamlit.cache_resource`, so within a session every
request reuses the same TCP connection to the backend.
"""

from __future__ import annotations

import httpx


class ApiClient:
    """HTTP client for the Zero-Trust RAG demo backend."""

    def __init__(self, base_url: str, *, timeout_s: float = 300.0) -> None:
        """Initialise the client.

        Args:
            base_url: e.g. ``"http://localhost:8000"`` (native dev) or
                ``"http://api:8000"`` (inside docker compose).
            timeout_s: per-request timeout. Generous default (300s)
                because the LLM-bound endpoints (``/ingestion/propose``,
                ``/query``) can take 20-60s on CPU-only inference.
        """
        self._base = base_url.rstrip("/")
        self._client = httpx.Client(timeout=httpx.Timeout(timeout_s))

    @property
    def base_url(self) -> str:
        return self._base

    # ─────────────────────────── Meta / Identity ───────────────────────────
    def health(self) -> httpx.Response:
        return self._client.get(f"{self._base}/health")

    def list_users(self) -> httpx.Response:
        return self._client.get(f"{self._base}/users")

    def get_user(self, user_id: str) -> httpx.Response:
        return self._client.get(f"{self._base}/users/{user_id}")

    # ─────────────────────────── Ingestion ─────────────────────────────────
    def propose_ingestion(
        self,
        *,
        file_bytes: bytes,
        filename: str,
        user_id: str,
    ) -> httpx.Response:
        """POST /ingestion/propose with the uploaded file."""
        return self._client.post(
            f"{self._base}/ingestion/propose",
            files={"file": (filename, file_bytes)},
            headers={"X-User-Id": user_id},
        )

    def commit_ingestion(
        self,
        *,
        proposal: dict,
        final_metadata: dict,
        user_id: str,
    ) -> httpx.Response:
        """POST /ingestion/commit with the (possibly edited) metadata."""
        return self._client.post(
            f"{self._base}/ingestion/commit",
            json={"proposal": proposal, "final_metadata": final_metadata},
            headers={"X-User-Id": user_id},
        )

    # ─────────────────────────── Query ─────────────────────────────────────
    def query(self, *, query_text: str, user_id: str) -> httpx.Response:
        """POST /query with the user question."""
        return self._client.post(
            f"{self._base}/query",
            json={"query": query_text},
            headers={"X-User-Id": user_id},
        )
