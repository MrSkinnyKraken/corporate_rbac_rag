#!/usr/bin/env python3
"""Fail-fast environment validation for the Zero-Trust RAG system.

This standalone CLI tool runs the five infrastructure-readiness checks that
must pass before any business-layer code is allowed to execute. Each check
is logged via the standard :mod:`logging` module at ``INFO`` level on
success and ``ERROR`` level on failure. The script terminates as soon as a
single check fails (fail-fast); on full success it exits with status ``0``,
on any failure with status ``1``.

The checks, in order, are:

    1. Config verification — :func:`core.config.get_settings` returns valid
       paths and URLs.
    2. ChromaDB heartbeat — the vector store responds to a lightweight ping.
    3. ChromaDB roundtrip — a smoke-test collection can be created, listed,
       and deleted, exercising the full read / write contract.
    4. Ollama heartbeat & model availability — the LLM server is reachable
       and the model named in ``OLLAMA_MODEL`` is actually downloaded.
    5. Ollama generation end-to-end — a tiny generation completes within
       the configured request timeout and returns non-empty text.

Usage::

    python scripts/healthcheck.py
    # or, when the project root is already on PYTHONPATH:
    python -m scripts.healthcheck

This script is also suitable as a Docker Compose ``healthcheck`` command
or as a CI step that gates deployment.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Callable

# ─────────────────────────────────────────────────────────────────────────────
# Path bootstrap — ensure the project root is importable regardless of CWD.
# Must run BEFORE any project imports.
# ─────────────────────────────────────────────────────────────────────────────
_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ─────────────────────────────────────────────────────────────────────────────
# Project imports
# ─────────────────────────────────────────────────────────────────────────────
from core import (  # noqa: E402  (import after sys.path bootstrap is intentional)
    LLMConnectionError,
    LLMError,
    LLMGenerationError,
    Settings,
    VectorDBConnectionError,
    ZeroTrustBaseError,
    get_settings,
)
from infrastructure import ChromaDBClient, OllamaClient  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
SMOKE_COLLECTION_NAME: str = "smoke_test_collection"
TOTAL_CHECKS: int = 5

_SEPARATOR: str = "-" * 72
_LOG_FORMAT: str = "%(asctime)s  %(levelname)-5s  %(message)s"
_LOG_DATEFMT: str = "%H:%M:%S"

log: logging.Logger = logging.getLogger("healthcheck")


# ─────────────────────────────────────────────────────────────────────────────
# Logging helpers
# ─────────────────────────────────────────────────────────────────────────────
def _setup_logging() -> None:
    """Configure root logging for the script.

    Idempotent: subsequent calls are no-ops thanks to ``basicConfig``'s own
    short-circuit when handlers already exist.
    """
    logging.basicConfig(
        level=logging.INFO,
        format=_LOG_FORMAT,
        datefmt=_LOG_DATEFMT,
    )


def _step_start(idx: int, title: str) -> None:
    log.info("[%d/%d] %s ...", idx, TOTAL_CHECKS, title)


def _step_pass(idx: int, elapsed_s: float) -> None:
    log.info("[%d/%d] PASSED in %.1f ms", idx, TOTAL_CHECKS, elapsed_s * 1000.0)


def _step_fail(idx: int, error: BaseException) -> None:
    log.error(
        "[%d/%d] FAILED -- %s: %s",
        idx,
        TOTAL_CHECKS,
        type(error).__name__,
        error,
    )
    cause: BaseException | None = error.__cause__
    if cause is not None:
        log.error(
            "        caused by %s: %s",
            type(cause).__name__,
            cause,
        )


def _detail(message: str) -> None:
    """Indented INFO line for sub-step diagnostics."""
    log.info("        %s", message)


# ─────────────────────────────────────────────────────────────────────────────
# The orchestrator
# ─────────────────────────────────────────────────────────────────────────────
class HealthCheck:
    """Fail-fast runner for the five infrastructure-readiness checks.

    Encapsulates state shared between checks (the :class:`ChromaDBClient`
    instance built in step 2 is reused in step 3; the :class:`OllamaClient`
    built in step 4 is reused in step 5) so each check function can stay
    small and free of plumbing.

    Attributes:
        settings: The validated :class:`Settings` for this process.
        chroma:   :class:`ChromaDBClient`, populated by step 2.
        llm:      :class:`OllamaClient`, populated by step 4.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings: Settings = settings
        self.chroma: ChromaDBClient | None = None
        self.llm: OllamaClient | None = None

    # ───────────────────────── Public entry point ───────────────────────────
    def run(self) -> int:
        """Execute the five checks sequentially.

        Returns:
            ``0`` if every check passed, ``1`` otherwise.
        """
        steps: list[tuple[str, Callable[[], None]]] = [
            ("Config verification",                       self._check_config),
            ("ChromaDB heartbeat",                        self._check_chroma_heartbeat),
            ("ChromaDB roundtrip (create / list / delete)", self._check_chroma_roundtrip),
            ("Ollama heartbeat & model availability",     self._check_ollama_heartbeat),
            ("Ollama generation end-to-end",              self._check_ollama_generation),
        ]
        try:
            for idx, (title, fn) in enumerate(steps, start=1):
                if not self._run_step(idx, title, fn):
                    return 1
            return 0
        finally:
            self._cleanup()

    # ────────────────────────── Per-step wrapper ────────────────────────────
    @staticmethod
    def _run_step(idx: int, title: str, fn: Callable[[], None]) -> bool:
        """Execute a single check, log entry / outcome, return success."""
        _step_start(idx, title)
        t0: float = time.perf_counter()
        try:
            fn()
        except ZeroTrustBaseError as exc:
            # Expected failure mode — our infrastructure adapters raise these
            # with a clear remediation message and original cause attached.
            _step_fail(idx, exc)
            return False
        except Exception as exc:  # noqa: BLE001  (safety net for unmodelled bugs)
            # Unmodelled error — likely a bug in the script or an environment
            # problem outside our exception taxonomy. Surfaced verbatim.
            _step_fail(idx, exc)
            return False
        _step_pass(idx, time.perf_counter() - t0)
        return True

    # ───────────────────────── Individual checks ────────────────────────────
    def _check_config(self) -> None:
        """Step 1: Settings load + required directories exist on disk."""
        s: Settings = self.settings
        _detail(f"chroma_base_url      = {s.chroma_base_url}")
        _detail(f"ollama_base_url      = {s.ollama_base_url}")
        _detail(f"ollama_model         = {s.ollama_model}")
        _detail(f"embedding_model_name = {s.embedding_model_name}")
        _detail(f"raw_docs_dir         = {s.raw_docs_dir}")
        _detail(f"bm25_index_dir       = {s.bm25_index_dir}")
        _detail(f"artifacts_dir        = {s.artifacts_dir}")

        # The Settings field validators auto-create these dirs at load time;
        # this assertion is a defence in depth in case a future refactor
        # silently drops the validator.
        for attr in ("raw_docs_dir", "bm25_index_dir", "artifacts_dir"):
            path: Path = getattr(s, attr)
            if not path.exists():
                raise FileNotFoundError(
                    f"Required directory '{attr}'={path} does not exist."
                )

    def _check_chroma_heartbeat(self) -> None:
        """Step 2: ChromaDB is reachable and answers heartbeat."""
        # Constructor itself performs an HTTP probe (chromadb 0.4.x is eager),
        # so a transport failure here surfaces as VectorDBConnectionError.
        self.chroma = ChromaDBClient(self.settings)
        if not self.chroma.health_check():
            raise VectorDBConnectionError(
                f"ChromaDB heartbeat returned False at {self.settings.chroma_base_url}. "
                f"Is the chromadb container running?"
            )
        _detail(f"heartbeat OK at {self.settings.chroma_base_url}")

    def _check_chroma_roundtrip(self) -> None:
        """Step 3: full read / write contract on a smoke-test collection."""
        if self.chroma is None:  # defensive — should be set by step 2
            raise VectorDBConnectionError(
                "ChromaDBClient was not initialised; step 2 must precede step 3."
            )

        # Defensive cleanup — a previous aborted run may have left the
        # collection behind, which would mask a subtle creation bug.
        existing: list[str] = self.chroma.list_collections()
        if SMOKE_COLLECTION_NAME in existing:
            self.chroma.delete_collection(SMOKE_COLLECTION_NAME)
            _detail(
                f"pre-existing '{SMOKE_COLLECTION_NAME}' deleted "
                f"(left over from a prior aborted run)"
            )

        # 1) create
        coll = self.chroma.get_or_create_collection(SMOKE_COLLECTION_NAME)
        if coll.name != SMOKE_COLLECTION_NAME:
            raise VectorDBConnectionError(
                f"created collection name mismatch: got '{coll.name}', "
                f"expected '{SMOKE_COLLECTION_NAME}'"
            )

        # 2) verify visibility
        if SMOKE_COLLECTION_NAME not in self.chroma.list_collections():
            raise VectorDBConnectionError(
                f"created collection '{SMOKE_COLLECTION_NAME}' is not visible "
                f"in list_collections()"
            )
        _detail(f"created '{SMOKE_COLLECTION_NAME}'")

        # 3) delete
        self.chroma.delete_collection(SMOKE_COLLECTION_NAME)
        if SMOKE_COLLECTION_NAME in self.chroma.list_collections():
            raise VectorDBConnectionError(
                f"collection '{SMOKE_COLLECTION_NAME}' was not deleted as expected"
            )
        _detail(f"deleted '{SMOKE_COLLECTION_NAME}'")

    def _check_ollama_heartbeat(self) -> None:
        """Step 4: Ollama is reachable and the configured model is loaded."""
        self.llm = OllamaClient(self.settings)
        if not self.llm.health_check():
            raise LLMConnectionError(
                f"Ollama heartbeat failed at {self.settings.ollama_base_url}. "
                f"Is `ollama serve` running?"
            )
        _detail(f"heartbeat OK at {self.settings.ollama_base_url}")

        models: list[str] = self.llm.list_models()
        # Ollama returns names like "llama3.2:latest"; OLLAMA_MODEL is usually
        # given as "llama3.2". Compare both the bare and tagged forms.
        bare_names: list[str] = [m.split(":")[0] for m in models]
        if (
            self.settings.ollama_model not in models
            and self.settings.ollama_model not in bare_names
        ):
            raise LLMConnectionError(
                f"Configured OLLAMA_MODEL='{self.settings.ollama_model}' is not "
                f"present on the Ollama server. Available: {models}. "
                f"Run `ollama pull {self.settings.ollama_model}` to download it."
            )
        _detail(f"available models = {models}")

    def _check_ollama_generation(self) -> None:
        """Step 5: short generation completes within the configured timeout."""
        if self.llm is None:  # defensive — should be set by step 4
            raise LLMConnectionError(
                "OllamaClient was not initialised; step 4 must precede step 5."
            )

        prompt: str = "Reply with just the word OK."
        max_tokens: int = 5

        resp = self.llm.generate_response(
            prompt,
            max_tokens=max_tokens,
            temperature=0.0,
        )

        if not resp.text:
            raise LLMGenerationError(
                "Ollama returned an empty response — the model may be busy "
                "warming up or stuck."
            )

        if resp.latency_s >= self.settings.ollama_request_timeout_s:
            raise LLMGenerationError(
                f"Generation took {resp.latency_s:.2f}s, equal to or above the "
                f"configured timeout of {self.settings.ollama_request_timeout_s}s. "
                f"Either raise OLLAMA_REQUEST_TIMEOUT_S or use a faster model."
            )

        _detail(f"prompt        = {prompt!r}")
        _detail(f"response.text = {resp.text!r}")
        _detail(
            f"latency       = {resp.latency_s:.3f} s  "
            f"(timeout = {self.settings.ollama_request_timeout_s} s)"
        )
        _detail(f"finish_reason = {resp.finish_reason}")

    # ──────────────────────────── Cleanup ───────────────────────────────────
    def _cleanup(self) -> None:
        """Release any HTTP resources held by adapters built during the run."""
        if self.llm is not None:
            self.llm.close()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main() -> int:
    """Run the full healthcheck suite and return the appropriate exit code.

    Returns:
        ``0`` on full success, ``1`` if any check fails.
    """
    _setup_logging()

    log.info(_SEPARATOR)
    log.info("Zero-Trust RAG  --  System Healthcheck")
    log.info(_SEPARATOR)

    try:
        settings: Settings = get_settings()
    except Exception as exc:  # noqa: BLE001
        log.error(
            "Could not load Settings — config is unusable. %s: %s",
            type(exc).__name__,
            exc,
        )
        return 1

    runner = HealthCheck(settings)
    rc: int = runner.run()

    log.info(_SEPARATOR)
    if rc == 0:
        log.info(
            "All %d checks passed. Infrastructure is ready for the domain layer.",
            TOTAL_CHECKS,
        )
    else:
        log.error("Healthcheck FAILED -- see errors above.")
    log.info(_SEPARATOR)
    return rc


if __name__ == "__main__":
    sys.exit(main())
