"""PII regex vault for the Custom RBAC Chunker.

Migrated from ``strategy-analysis/notebooks/ph1-chunking-metadata-strategy/
03_local_metadata_rbac.ipynb`` (Phase 1) and hardened with the lessons learned
during Phases 2 and 3:

  * ``swift_code`` originally caught any uppercase 4-letter acronym followed
    by 2 letters and 2-5 alphanumerics — words like ``"SWIFT MEMORY ACCESS"``
    would trip it. The hardened version requires the ``SWIFT`` keyword as a
    prefix AND enforces the ISO 9362 length: 4-letter bank code + 2-letter
    country code + 2 alphanumeric location chars + optional 3-char branch.
  * ``phone`` originally accepted any string of digits with optional dashes,
    which matched dates ``"2026-04-15"``, version numbers, port numbers,
    invoice IDs, etc. The hardened version requires either a contextual
    keyword (``phone``, ``tel``, ``mobile`` …) within the line OR an
    international ``+`` prefix.
  * Every pattern now uses ``\\b`` word boundaries (or negative
    look-arounds for the email local-part) to avoid matching mid-token.
  * Length constraints are enforced: IBANs to ISO 13616 (BBAN 12-28
    alphanumerics), DNI/EMP-style IDs to a fixed digit count, etc.

----

ENTERPRISE-PRODUCTION NOTE
==========================

These regex rules are a *floor*, not a ceiling. They catch the well-formed
patterns that appear in the corpus we used to validate the architecture
during Phases 1-3. For a true production release inside a customer
environment, this module must be supplemented with a proper Named-Entity
Recognition layer such as **Microsoft Presidio**
(`<https://microsoft.github.io/presidio/>`_) or a fine-tuned NER model
(spaCy, HuggingFace transformers) to catch the entities that resist regex:

    * Social-security numbers / national IDs in *every* country format
    * Physical addresses (mixed letters, digits, postal codes)
    * Dates of birth distinguished from any other date
    * Personal names without a courtesy title in front
    * Credit-card numbers with Luhn validation
    * Account numbers in non-IBAN regions (US ABA, UK sort code, etc.)
    * Health-record identifiers (HL7, FHIR resource IDs)

The :class:`PIIPattern` interface defined here is intentionally agnostic
of the matcher used: a future ``NERPattern`` or ``PresidioPattern`` can
expose the same dataclass shape (``name``, ``min_clearance``,
``description``) and the chunker will compose them into the existing
detection loop without changes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final, Pattern

from core.security import ClearanceLevel


@dataclass(frozen=True, slots=True)
class PIIPattern:
    """A single regex-based PII detection rule.

    Attributes:
        name:           Stable identifier emitted into a chunk's
                        ``sensitivity_types`` list. MUST be ``snake_case`` and
                        unique within :data:`PII_PATTERNS`.
        compiled:       Pre-compiled :class:`re.Pattern` object. Compiling at
                        import time means the chunker pays the regex
                        compilation cost once per process, not once per chunk.
        min_clearance:  The :class:`~core.security.ClearanceLevel` a chunk
                        MUST hold once this pattern is detected inside it.
                        The chunker takes the *maximum* over every pattern
                        that fires plus the parent document's clearance.
        description:    One-line human-readable rationale. Surfaced into
                        audit logs when the pattern triggers.
    """

    name: str
    compiled: Pattern[str]
    min_clearance: ClearanceLevel
    description: str


# =============================================================================
# 1. Email — RFC-5322-ish with strict word-boundary anchoring.
# =============================================================================
# Negative look-behind: not preceded by another local-part character (avoids
# splitting "x.foo+bar@host" from a longer literal). Negative look-ahead at
# the tail to avoid matching a partial TLD inside a longer token.
EMAIL: Final[Pattern[str]] = re.compile(
    r"(?<![\w.+-])"
    r"[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.[A-Za-z]{2,24}"
    r"(?![\w])"
)

# =============================================================================
# 2. IBAN — ISO 13616 (4-7 groups of 4 alphanumerics after the country/check).
# =============================================================================
# Optional ``IBAN`` keyword prefix. Word boundaries on both sides. Length is
# bounded to the legal range without enumerating every country's BBAN width.
IBAN: Final[Pattern[str]] = re.compile(
    r"(?:\bIBAN\s*[:#]?\s*)?"
    r"\b[A-Z]{2}\d{2}\s?(?:[A-Z0-9]{4}\s?){3,7}[A-Z0-9]{0,4}\b"
)

# =============================================================================
# 3. IPv4 — strict octets in 0-255.
# =============================================================================
# The canonical strict-IPv4 regex. Each octet matches one of:
#   * 250-255      (25[0-5])
#   * 200-249      (2[0-4]\d)
#   * 100-199      (1\d{2})
#   * 0-99         ([1-9]?\d)
IP_ADDRESS: Final[Pattern[str]] = re.compile(
    r"\b(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)"
    r"(?:\.(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)){3}\b"
)

# =============================================================================
# 4. Password — keyword + delimiter + value.
# =============================================================================
# Requires a credential keyword (case-insensitive) followed by ``:`` or ``=``
# and a value of at least 3 non-whitespace characters. Avoids matching
# placeholder text like ``password = ?`` or ``pwd: ""``.
PASSWORD: Final[Pattern[str]] = re.compile(
    r"(?i)\b(?:password|contrase[ñn]a|pwd|"
    r"override[_\s]?password|secret|api[_\s]?key|token)\b"
    r"\s*[:=]\s*['\"]?[^\s'\"]{3,}['\"]?"
)

# =============================================================================
# 5. Phone — context-anchored to defeat numeric-sequence false positives.
# =============================================================================
# Either a contextual keyword (phone / tel / mobile / WhatsApp / fax) precedes
# the number, OR the number starts with an international ``+``. Bare 9-digit
# sequences (dates, version numbers, port numbers) NEVER match.
PHONE: Final[Pattern[str]] = re.compile(
    r"""
    (?:
        \b(?i:phone|tel|t[ée]l|mobile|m[óo]vil|whatsapp|fax|ext)\b
        \s*[:.\s]\s*
      |
        (?<![\w.])\+
    )
    \(?\d{1,4}\)?
    (?:[\s.-]?\d{2,4}){2,4}
    \b
    """,
    re.VERBOSE,
)

# =============================================================================
# 6. Person name (with courtesy-title context).
# =============================================================================
# Requires a courtesy title or "Contacto" prefix to anchor the match. Bare
# capitalised double-words (e.g., "New York", "Black Friday") never match.
PERSON_NAME_CONTEXT: Final[Pattern[str]] = re.compile(
    r"\b(?:D\.|D[ñn]a\.|Mr\.|Mrs\.|Ms\.|Mx\.|Dra?\.?|Sr\.|Sra\.|"
    r"Lic\.|Ing\.|Prof\.|Contacto:?)\s+"
    r"[A-Z][a-záéíóúñç]+"
    r"(?:\s+[A-Z][a-záéíóúñç]+){1,2}\b"
)

# =============================================================================
# 7. Client ID — corporate identifier of the form CLI-NNN(N).
# =============================================================================
CLIENT_ID: Final[Pattern[str]] = re.compile(r"\bCLI-\d{3,6}\b")

# =============================================================================
# 8. SWIFT / BIC code — ISO 9362 with mandatory ``SWIFT`` keyword prefix.
# =============================================================================
# Format: 4-letter bank code + 2-letter country + 2 alphanumeric location
# + optional 3-char branch. The ``SWIFT[:\\s]+`` prefix prevents matching
# generic 8-12 char uppercase tokens.
SWIFT_CODE: Final[Pattern[str]] = re.compile(
    r"\bSWIFT[\s:]+[A-Z]{6}[A-Z0-9]{2}(?:[A-Z0-9]{3})?\b"
)

# =============================================================================
# 9. Monetary value — context-anchored to a finance keyword.
# =============================================================================
# Required nearby keyword (cost, salary, revenue, budget, …) within 80
# non-newline characters of the € amount. Eliminates matches in product specs
# ("kit costs ~5 hours of training" — false positive in Phase 1).
MONETARY_VALUE: Final[Pattern[str]] = re.compile(
    r"""
    (?ix)
    \b(?:cost|coste|precio|price|salary|salario|amount|importe|fee|
        invoice|factura|budget|presupuesto|revenue|ingreso|margin|margen|
        payroll|n[oó]mina|expense|gasto|valor|wage|fund(?:ing)?|
        investment|inversi[oó]n)\b
    [^.\n]{0,80}
    €\s?\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?
    (?:\s?(?:million|mil(?:l[oó]n(?:es)?)?|k|m|thousand|billion|bn))?
    """,
    re.VERBOSE,
)


# =============================================================================
# Aggregate — the canonical 9-pattern set.
# =============================================================================
# Order is meaningful only insofar as the chunker reports detection in this
# order; all patterns run on every chunk. The (min_clearance) field maps
# each detection to the *minimum* clearance level the host chunk must hold,
# following the Phase-1-validated severity ladder.
PII_PATTERNS: Final[list[PIIPattern]] = [
    PIIPattern(
        name="password",
        compiled=PASSWORD,
        min_clearance=ClearanceLevel.STRICT,
        description="Literal credential value (password, API key, token).",
    ),
    PIIPattern(
        name="iban",
        compiled=IBAN,
        min_clearance=ClearanceLevel.STRICT,
        description="ISO 13616 IBAN account number.",
    ),
    PIIPattern(
        name="swift_code",
        compiled=SWIFT_CODE,
        min_clearance=ClearanceLevel.STRICT,
        description="ISO 9362 SWIFT/BIC code with mandatory keyword prefix.",
    ),
    PIIPattern(
        name="email",
        compiled=EMAIL,
        min_clearance=ClearanceLevel.CONFIDENTIAL,
        description="Personal or corporate email address.",
    ),
    PIIPattern(
        name="ip_address",
        compiled=IP_ADDRESS,
        min_clearance=ClearanceLevel.CONFIDENTIAL,
        description="Strict IPv4 address (octets 0-255).",
    ),
    PIIPattern(
        name="client_id",
        compiled=CLIENT_ID,
        min_clearance=ClearanceLevel.CONFIDENTIAL,
        description="Corporate client identifier of the form CLI-NNN.",
    ),
    PIIPattern(
        name="person_name_context",
        compiled=PERSON_NAME_CONTEXT,
        min_clearance=ClearanceLevel.CONFIDENTIAL,
        description="Personal name preceded by a courtesy title or 'Contacto'.",
    ),
    PIIPattern(
        name="monetary_value",
        compiled=MONETARY_VALUE,
        min_clearance=ClearanceLevel.CONFIDENTIAL,
        description="Euro amount adjacent to a finance keyword.",
    ),
    PIIPattern(
        name="phone",
        compiled=PHONE,
        min_clearance=ClearanceLevel.INTERNAL,
        description="Phone number with required keyword or +intl prefix.",
    ),
]


def get_pattern(name: str) -> PIIPattern:
    """Look up a :class:`PIIPattern` by its ``name`` field.

    Raises:
        KeyError: if no pattern with that name is registered.
    """
    for p in PII_PATTERNS:
        if p.name == name:
            return p
    raise KeyError(f"Unknown PII pattern name: {name!r}")
