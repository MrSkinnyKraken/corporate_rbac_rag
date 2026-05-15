"""LLM-based whole-document metadata classifier.

The classifier reads an excerpt of an uploaded document and asks the LLM
for a structured proposal of:

  * ``home_department``    — which real department OWNS the document.
  * ``allowed_departments`` — read scope (may include the ``"all"``
                              wildcard for fully public material).
  * ``clearance_level``    — severity 0-3 (PUBLIC → STRICT).
  * ``ksp_text``           — 1-3 sentence summary used by the KSP
                              Router-Index for semantic routing.
  * ``reasoning``          — short justification surfaced to the user
                              at the human-in-the-loop confirmation
                              step. Never persisted to chunks / KSP.

This is the "LLM global document metadata allocation" step that the
chunker's docstring explicitly defers to the application layer. The
output is **a proposal, not a decision** — the api adapter shows it to
the uploading user via a Streamlit form, who may edit any field before
the orchestrator's :meth:`IngestionApp.commit` writes the final
metadata.

Pure orchestration: depends on the abstract :class:`BaseLLM` (not on
``OllamaClient`` specifically), so swapping to a different backend
requires no change here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from core.exceptions import (
    IngestionError,
    LLMResponseFormatError,
)
from core.security import (
    ALLOWED_DEPARTMENT_WILDCARD,
    AllowedDepartment,
    ClearanceLevel,
    Department,
)
from infrastructure.llm_client import BaseLLM


_MAX_EXCERPT_CHARS: int = 6000
"""Hard cap on the excerpt sent to the LLM.

~1500 tokens with most tokenisers — comfortably below the default
``ollama_num_ctx`` of 8192, leaving plenty of headroom for the system
prompt, the JSON schema instruction, and the generation budget. A
longer excerpt buys diminishing returns: the classifier only needs the
document's *flavour*, not its full text.
"""


_PROMPT_TEMPLATE: str = """You are a document classifier for a Zero-Trust enterprise RAG system.
Read the document excerpt below and return ONE JSON object with EXACTLY these keys:

{{
  "home_department": "<one of: hr, finance, engineering, legal, sales, operations, marketing>",
  "allowed_departments": ["<dept>", ...],
  "clearance_level": <integer 0-3>,
  "ksp_text": "<1-3 sentence summary capturing the key semantic points>",
  "reasoning": "<brief justification, max 2 sentences>"
}}

RULES:
1. "home_department" is the SINGLE department that most naturally owns this document.
   Never use "all" here — every document has exactly one owner.
2. "allowed_departments" is the read-scope list. Always include "home_department" in it.
   Add other departments only if the content is genuinely cross-functional.
   Use "all" ONLY if the content is fully public (employee handbook,
   marketing brochure, public announcement). NEVER mix "all" with other names.
3. "clearance_level" is severity, NOT scope:
     0 = PUBLIC        (handbook, marketing material, FAQs)
     1 = INTERNAL      (org charts, schedules, internal announcements)
     2 = CONFIDENTIAL  (financials, contracts, partner data)
     3 = STRICT        (PII, credentials, salary data, legal disputes)
   Be CONSERVATIVE: when in doubt, prefer the HIGHER level.
4. "ksp_text" must be dense and specific so queries like
   "find documents about <topic>" can match it semantically. Mention the
   document's *subject*, not its *format*.
5. "reasoning" is a short justification (max 2 sentences) shown to the
   uploading user for review. NOT persisted to the index.

Source file: {source_file}

Document excerpt:
---
{excerpt}
---

Return ONLY the JSON object. No prose around it."""


# ─────────────────────────────────────────────────────────────────────────────
# Proposal dataclass (re-exported by `application/__init__.py`)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class ProposedMetadata:
    """The LLM's (possibly user-edited) classification of a document.

    Constructed twice during ingestion: once by
    :meth:`LLMDocumentClassifier.classify` (the raw LLM proposal), once
    by the api adapter after the user has reviewed and possibly edited
    the fields. The :class:`IngestionApp.commit` method treats the
    second instance as authoritative.
    """

    home_department: Department
    allowed_departments: list[AllowedDepartment]
    clearance_level: ClearanceLevel
    ksp_text: str
    reasoning: str = ""

    def __post_init__(self) -> None:
        """Validate the proposal is internally consistent.

        Raised here (not at LLM-parse time) so a user editing the
        proposal in the UI can also be caught by the same rules.

        Raises:
            IngestionError: any invariant breach.
        """
        if not isinstance(self.home_department, Department):
            raise IngestionError(
                f"ProposedMetadata.home_department must be a Department enum, "
                f"got {type(self.home_department).__name__}={self.home_department!r}."
            )
        if not isinstance(self.clearance_level, ClearanceLevel):
            raise IngestionError(
                f"ProposedMetadata.clearance_level must be a ClearanceLevel enum, "
                f"got {type(self.clearance_level).__name__}={self.clearance_level!r}."
            )
        if not self.allowed_departments:
            raise IngestionError(
                "ProposedMetadata.allowed_departments must be non-empty."
            )
        for d in self.allowed_departments:
            if not isinstance(d, AllowedDepartment):
                raise IngestionError(
                    f"ProposedMetadata.allowed_departments entries must be "
                    f"AllowedDepartment enums; got "
                    f"{type(d).__name__}={d!r}."
                )

        # home_department MUST be in allowed_departments (or the list must
        # be the wildcard "all"). The chunker's home-collection routing
        # would silently mis-route otherwise.
        home_str = self.home_department.value
        allowed_strs = [d.value for d in self.allowed_departments]
        wildcard_present = ALLOWED_DEPARTMENT_WILDCARD in allowed_strs
        if not wildcard_present and home_str not in allowed_strs:
            raise IngestionError(
                f"ProposedMetadata.home_department={home_str!r} must appear "
                f"in allowed_departments={allowed_strs!r} (or the list must "
                f"contain the 'all' wildcard)."
            )
        if wildcard_present and len(allowed_strs) > 1:
            raise IngestionError(
                f"ProposedMetadata.allowed_departments contains the 'all' "
                f"wildcard alongside other entries: {allowed_strs!r}. "
                f"The wildcard must appear alone."
            )
        if not self.ksp_text or not self.ksp_text.strip():
            raise IngestionError(
                "ProposedMetadata.ksp_text must be non-empty."
            )


# ─────────────────────────────────────────────────────────────────────────────
# Classifier
# ─────────────────────────────────────────────────────────────────────────────

class LLMDocumentClassifier:
    """Run the metadata-classification prompt and return a :class:`ProposedMetadata`."""

    def __init__(
        self,
        llm: BaseLLM,
        *,
        max_excerpt_chars: int = _MAX_EXCERPT_CHARS,
    ) -> None:
        """Initialise the classifier.

        Args:
            llm: An LLM adapter satisfying :class:`BaseLLM`. Injected so
                the demo can swap Ollama for a remote backend later
                without touching the classifier.
            max_excerpt_chars: Truncation threshold applied to the
                document text BEFORE it is interpolated into the
                prompt. Defaults to :data:`_MAX_EXCERPT_CHARS`.
        """
        if max_excerpt_chars < 200:
            raise IngestionError(
                f"max_excerpt_chars must be >= 200 to give the LLM a "
                f"chance at classifying; got {max_excerpt_chars}."
            )
        self._llm: BaseLLM = llm
        self._max_excerpt_chars: int = max_excerpt_chars

    # ─────────────────────────── Public API ─────────────────────────────────
    def classify(
        self,
        text: str,
        source_file: str,
    ) -> tuple[ProposedMetadata, float]:
        """Run the classifier and return ``(proposed_metadata, latency_s)``.

        Args:
            text:        The full text already extracted from the file
                         by :class:`~domain.chunking.parsers.DocumentParser`.
            source_file: Original filename (passed to the LLM as context
                         and propagated to the audit trail).

        Returns:
            ``(metadata, latency_s)``. The latency is the wall-clock
            time the LLM call took, useful for the audit log and demo
            UI.

        Raises:
            IngestionError: empty text, LLM produced invalid JSON, the
                JSON missed required fields, or the parsed values
                violate the :class:`ProposedMetadata` invariants.
            LLMConnectionError / LLMGenerationError: forwarded from the
                LLM adapter.
        """
        if not text or not text.strip():
            raise IngestionError(
                f"Cannot classify empty document (source_file={source_file!r})."
            )

        excerpt = text[: self._max_excerpt_chars]
        prompt = _PROMPT_TEMPLATE.format(
            excerpt=excerpt, source_file=source_file,
        )

        response = self._llm.generate_response(
            prompt,
            temperature=0.1,
            max_tokens=500,
            format_json=True,
        )

        try:
            payload = json.loads(response.text)
        except json.JSONDecodeError as exc:
            raise LLMResponseFormatError(
                f"Classifier LLM did not return valid JSON for "
                f"{source_file!r}: {response.text[:200]!r}"
            ) from exc

        if not isinstance(payload, dict):
            raise LLMResponseFormatError(
                f"Classifier LLM returned a non-object JSON for "
                f"{source_file!r}: top-level type {type(payload).__name__}."
            )

        metadata = _coerce_payload(payload, source_file)
        return metadata, response.latency_s


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _coerce_payload(payload: dict[str, Any], source_file: str) -> ProposedMetadata:
    """Map a raw LLM JSON object into a typed :class:`ProposedMetadata`.

    Each conversion that can fail raises :class:`IngestionError` with a
    diagnostic naming the offending field — operations + UI can show
    the user exactly what the LLM produced and which part is wrong.
    """
    # 1) home_department
    home_raw = payload.get("home_department")
    if not isinstance(home_raw, str):
        raise IngestionError(
            f"Classifier output for {source_file!r}: 'home_department' "
            f"must be a string, got {type(home_raw).__name__}={home_raw!r}."
        )
    try:
        home = Department(home_raw.strip().lower())
    except ValueError as exc:
        valid = sorted(d.value for d in Department)
        raise IngestionError(
            f"Classifier output for {source_file!r}: 'home_department'="
            f"{home_raw!r} is not a valid Department; expected one of {valid}."
        ) from exc

    # 2) allowed_departments
    allowed_raw = payload.get("allowed_departments")
    if not isinstance(allowed_raw, list) or not allowed_raw:
        raise IngestionError(
            f"Classifier output for {source_file!r}: 'allowed_departments' "
            f"must be a non-empty list, got {allowed_raw!r}."
        )
    allowed: list[AllowedDepartment] = []
    for entry in allowed_raw:
        if not isinstance(entry, str):
            raise IngestionError(
                f"Classifier output for {source_file!r}: "
                f"'allowed_departments' entries must be strings; got "
                f"{type(entry).__name__}={entry!r}."
            )
        try:
            allowed.append(AllowedDepartment(entry.strip().lower()))
        except ValueError as exc:
            valid = sorted(d.value for d in AllowedDepartment)
            raise IngestionError(
                f"Classifier output for {source_file!r}: "
                f"'allowed_departments' contains {entry!r}, not in {valid}."
            ) from exc

    # 3) clearance_level
    cl_raw = payload.get("clearance_level")
    try:
        clearance = ClearanceLevel.from_int(int(cl_raw))  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise IngestionError(
            f"Classifier output for {source_file!r}: 'clearance_level'="
            f"{cl_raw!r} must be an integer in 0..3."
        ) from exc

    # 4) ksp_text
    ksp_raw = payload.get("ksp_text")
    if not isinstance(ksp_raw, str) or not ksp_raw.strip():
        raise IngestionError(
            f"Classifier output for {source_file!r}: 'ksp_text' must be "
            f"a non-empty string, got {ksp_raw!r}."
        )

    # 5) reasoning (optional)
    reasoning_raw = payload.get("reasoning", "")
    reasoning = reasoning_raw.strip() if isinstance(reasoning_raw, str) else ""

    return ProposedMetadata(
        home_department=home,
        allowed_departments=allowed,
        clearance_level=clearance,
        ksp_text=ksp_raw.strip(),
        reasoning=reasoning,
    )
