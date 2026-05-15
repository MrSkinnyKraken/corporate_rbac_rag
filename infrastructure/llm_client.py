"""LLM adapter implementing the Strategy Pattern.

Provides:
    * :class:`BaseLLM`     — abstract interface every LLM adapter must implement.
    * :class:`LLMResponse` — frozen dataclass returned by every generation call.
    * :class:`OllamaClient` — concrete adapter for a local Ollama HTTP server.

This module is pure infrastructure: no prompt engineering, no business
logic, no RBAC. It only knows how to send a prompt over HTTP and translate
transport / format errors into the project's exception hierarchy.

Adding a new LLM backend (e.g., a remote vLLM server) is a matter of writing
another :class:`BaseLLM` subclass — no caller code changes.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any, Self

import httpx

from core.config import Settings, get_settings
from core.exceptions import (
    LLMConnectionError,
    LLMGenerationError,
    LLMResponseFormatError,
)


# =============================================================================
# Result type
# =============================================================================

@dataclass(frozen=True, slots=True)
class LLMResponse:
    """The result of a single LLM generation call.

    Attributes:
        text: The generated text, with leading / trailing whitespace stripped.
        model: Name of the model that produced the response (taken from the
            settings of the adapter; useful in audit logs when multiple
            backends coexist).
        latency_s: End-to-end wall-clock time of the generation call, in
            seconds, with 4-digit precision.
        finish_reason: Optional reason supplied by the backend (Ollama:
            ``"stop"``, ``"length"`` …). ``None`` if the backend did not report it.
        raw: The full backend response (parsed JSON). Kept for diagnostics
            and downstream extraction (e.g., token counts in newer Ollama
            versions). May be ``None`` for responses that did not parse to
            JSON but were still salvageable.
    """

    text: str
    model: str
    latency_s: float
    finish_reason: str | None = None
    raw: dict[str, Any] | None = field(default=None)


# =============================================================================
# Strategy interface
# =============================================================================

class BaseLLM(ABC):
    """Abstract base class for every LLM adapter (Strategy Pattern).

    Concrete subclasses must implement :meth:`generate_response` and
    :meth:`health_check`. They should also call ``super().__init__()`` and
    accept an injected :class:`~core.config.Settings` instance, so the
    adapter remains testable without mocking the global config.
    """

    @abstractmethod
    def generate_response(
        self,
        prompt: str,
        *,
        temperature: float = 0.1,
        max_tokens: int = 300,
        format_json: bool = False,
    ) -> LLMResponse:
        """Synchronously generate a completion for ``prompt``.

        Args:
            prompt: The full prompt text — system prompt, few-shot, and user
                turn already concatenated. The adapter does not perform any
                templating.
            temperature: Sampling temperature. ``0.0`` is greedy / deterministic.
            max_tokens: Maximum number of tokens to generate (Ollama: ``num_predict``).
            format_json: If ``True``, instruct the backend to constrain output
                to valid JSON.

        Returns:
            A populated :class:`LLMResponse`.

        Raises:
            LLMConnectionError: backend unreachable or timed out.
            LLMGenerationError: backend returned a non-2xx HTTP status.
            LLMResponseFormatError: backend returned a body that is not parseable.
        """
        ...

    @abstractmethod
    def health_check(self) -> bool:
        """Return ``True`` iff the backend service is currently reachable.

        Implementations should be cheap (a small GET / heartbeat call) and
        must never raise; on any failure they return ``False``.
        """
        ...


# =============================================================================
# Concrete adapter: Ollama
# =============================================================================

class OllamaClient(BaseLLM):
    """Concrete LLM adapter for the Ollama HTTP API.

    Connection parameters (host, port, model, timeouts, ``num_ctx``) are read
    from the injected :class:`~core.config.Settings`. The adapter owns an
    ``httpx.Client`` for connection pooling; close it via :meth:`close` or
    use the adapter as a context manager.

    Example:
        >>> with OllamaClient() as llm:
        ...     resp = llm.generate_response("Say hi.", max_tokens=20)
        ...     print(resp.text)
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        http_client: httpx.Client | None = None,
    ) -> None:
        """Initialise the adapter.

        Args:
            settings: Application settings. If ``None``, fetched via
                :func:`~core.config.get_settings`.
            http_client: Pre-configured ``httpx.Client`` for testing or
                custom transports. If ``None``, a default client is built
                with the ``ollama_request_timeout_s`` from settings.
        """
        self._settings: Settings = settings or get_settings()
        self._http: httpx.Client = http_client or httpx.Client(
            timeout=httpx.Timeout(self._settings.ollama_request_timeout_s),
        )

    # -------------------------------------------------------------- BaseLLM
    def generate_response(
        self,
        prompt: str,
        *,
        temperature: float = 0.1,
        max_tokens: int = 300,
        format_json: bool = False,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self._settings.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": float(temperature),
                "num_predict": int(max_tokens),
                "num_ctx": self._settings.ollama_num_ctx,
            },
        }
        if format_json:
            payload["format"] = "json"

        t0 = time.perf_counter()
        try:
            resp = self._http.post(self._settings.ollama_generate_url, json=payload)
        except httpx.ConnectError as exc:
            raise LLMConnectionError(
                f"Cannot connect to Ollama at {self._settings.ollama_base_url}: {exc}"
            ) from exc
        except httpx.TimeoutException as exc:
            raise LLMConnectionError(
                f"Ollama request to {self._settings.ollama_generate_url} timed out: {exc}"
            ) from exc
        except httpx.HTTPError as exc:  # network-layer fallback
            raise LLMConnectionError(
                f"HTTP transport error talking to Ollama: {exc}"
            ) from exc

        if resp.status_code >= 400:
            raise LLMGenerationError(
                f"Ollama returned HTTP {resp.status_code}: "
                f"{resp.text[:200]!r}"
            )

        try:
            data: dict[str, Any] = resp.json()
        except json.JSONDecodeError as exc:
            raise LLMResponseFormatError(
                f"Ollama response was not valid JSON: {resp.text[:200]!r}"
            ) from exc

        text: str = str(data.get("response", "")).strip()
        return LLMResponse(
            text=text,
            model=self._settings.ollama_model,
            latency_s=round(time.perf_counter() - t0, 4),
            finish_reason=data.get("done_reason"),
            raw=data,
        )

    def health_check(self) -> bool:
        try:
            r = self._http.get(self._settings.ollama_tags_url, timeout=httpx.Timeout(5.0))
            r.raise_for_status()
            return True
        except (httpx.HTTPError, OSError):
            return False

    # ------------------------------------------------------------- Extras
    def list_models(self) -> list[str]:
        """Return the names of the models currently loaded in the Ollama server.

        Useful for the application's startup self-check (verify that
        ``settings.ollama_model`` is actually available before serving traffic).

        Raises:
            LLMConnectionError: if the server is unreachable.
            LLMResponseFormatError: if the response body cannot be parsed.
        """
        try:
            r = self._http.get(self._settings.ollama_tags_url, timeout=httpx.Timeout(10.0))
            r.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMConnectionError(
                f"Cannot list Ollama models at {self._settings.ollama_tags_url}: {exc}"
            ) from exc
        try:
            data = r.json()
        except json.JSONDecodeError as exc:
            raise LLMResponseFormatError(
                f"Ollama /api/tags response not parseable: {r.text[:200]!r}"
            ) from exc
        return [str(m["name"]) for m in data.get("models", [])]

    # --------------------------------------------------------- Lifecycle
    def close(self) -> None:
        """Release the underlying HTTP connection pool."""
        self._http.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
