"""Intent classification for dynamic α-vector selection in RRF fusion.

The asymmetric ensemble retriever weights its two rankers — dense
embedding (vector) and BM25 (lexical) — with a parameter α ∈ [0, 1]
chosen per query. The optimal α depends on what the user is asking for:

  * **PII-exact retrieval** (e.g., "give me Joan Pla's IBAN")
    Dense models paraphrase too aggressively to be reliable for exact
    PII strings; BM25's literal token match wins. α = 0.15.
  * **Lexical look-up** (e.g., "find the report titled INV-2024-001")
    Names, codes, identifiers — again BM25-leaning. α = 0.30.
  * **Mixed** queries with no clear lean. α = 0.50 (balanced).
  * **Conceptual** queries ("explain the warranty policy", "summarise
    the Q4 results") need semantic paraphrase — dense embeddings win.
    α = 0.80.

This module classifies a query into one of four
:class:`QueryIntent` values and maps that to the corresponding α.

The classifier is a **pure deterministic function** built on multilingual
regex patterns (EN / ES / CA). It deliberately does NOT call an LLM:
the routing layer must stay cheap enough to run in front of *every*
retrieval request. A future iteration may upgrade to an LLM-based
classifier behind a feature flag if the regex patterns prove too coarse.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Iterable

from core.exceptions import PipelineError


def _compile_word_boundary(keywords: Iterable[str]) -> list[re.Pattern[str]]:
    """Compile each keyword as a case-insensitive word-boundary regex.

    Plain ``substring in lowered_query`` checks were the original
    approach but fired false positives on short identifiers — e.g.
    ``"cif"`` (Spanish tax ID) matches inside ``"specific"``. Wrapping
    each keyword in ``\\b…\\b`` enforces token boundaries, including
    across multi-word keywords like ``"email address"``.

    Keywords with leading or trailing whitespace in the source bank are
    stripped before compilation; the ``\\b`` anchors handle the boundary
    semantics they were doing manually.
    """
    out: list[re.Pattern[str]] = []
    for raw in keywords:
        kw = raw.strip()
        if not kw:
            continue
        out.append(re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE))
    return out


class QueryIntent(str, Enum):
    """Coarse intent class assigned to a user query."""

    PII_EXACT = "pii_exact"
    """Query targets a specific PII string (IBAN, email, phone, …)."""

    LEXICAL = "lexical"
    """Query targets a specific named entity, code, or quoted phrase."""

    MIXED = "mixed"
    """Query carries no strong signal in either direction (default)."""

    CONCEPTUAL = "conceptual"
    """Query is paraphrasable / asks for explanation / summary."""


# α weight applied to the vector branch in RRF (lexical receives 1-α).
# Tuned from the Phase 3 Step 4 ablation experiments.
_INTENT_ALPHA: dict[QueryIntent, float] = {
    QueryIntent.PII_EXACT:  0.15,
    QueryIntent.LEXICAL:    0.30,
    QueryIntent.MIXED:      0.50,
    QueryIntent.CONCEPTUAL: 0.80,
}


# ─────────────────────────────────────────────────────────────────────────────
# Pattern banks (lower-cased substring matching unless a regex is supplied)
# ─────────────────────────────────────────────────────────────────────────────

# A query mentioning any of these is almost certainly asking for a PII
# value to be retrieved verbatim. Multilingual: EN / ES / CA.
_PII_KEYWORDS: tuple[str, ...] = (
    # Banking / payment
    "iban", "swift", "bic", "account number", "bank account",
    "número de cuenta", "numero de cuenta", "núm. cuenta", "núm cuenta",
    "número de compte", "num. compte", "num compte",
    "credit card", "card number",
    "tarjeta de credito", "tarjeta de crédito", "número de tarjeta",
    "targeta de crèdit", "targeta de credit", "número de targeta",
    # Authentication
    "password", "passphrase", "pwd",
    "contraseña", "clave de acceso",
    "contrasenya", "clau d'accés", "clau d'acces",
    # Contact
    "email address", "e-mail address",
    "correo electrónico", "correo electronico", "dirección de correo",
    "correu electrònic", "correu electronic", "adreça de correu",
    "phone number", "telephone number", "mobile number",
    "número de teléfono", "numero de telefono", "número móvil",
    "número de telèfon", "numero de telefon", "número mòbil",
    # Government IDs
    "ssn", "social security",
    "tax id", "vat number",
    "dni", "nie", "cif", "nif",
    # Internal customer / employee IDs
    "client id", "customer id", "employee id",
    "id del cliente", "id del empleado",
    "id del client", "id de l'empleat",
)

# Regexes that detect a *format* (not a keyword) suggestive of PII-exact intent.
_PII_FORMAT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b[A-Z]{2}\d{2}[\sA-Z0-9]{11,30}\b"),                       # IBAN-ish
    re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),     # email literal
    re.compile(r"(?:^|[\s(])\+\d{1,3}[\s\-]?\d{6,12}\b"),                    # +ES phone
    re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b"),                              # IPv4
    re.compile(r"\b[A-Z]{6}[A-Z2-9][A-NP-Z0-9](?:[A-Z0-9]{3})?\b"),          # SWIFT/BIC
)

# Pre-compiled word-boundary regexes for the keyword banks. Done once at
# module load so the per-query path stays a linear sweep of pre-built patterns.
_PII_KEYWORD_PATTERNS: list[re.Pattern[str]]        # populated below
_LEXICAL_KEYWORD_PATTERNS: list[re.Pattern[str]]
_CONCEPTUAL_KEYWORD_PATTERNS: list[re.Pattern[str]]

# Keywords that drag intent towards LEXICAL (specific token / identifier search).
_LEXICAL_KEYWORDS: tuple[str, ...] = (
    # Action verbs of literal retrieval
    "find ", "search for ", "look up ", "look for ", "show me ", "list ",
    "buscar ", "encuentra ", "encontrar ", "lístame ", "lista de ",
    "cerca ", "troba ", "trobar ", "mostra'm ", "llista ", "llista de ",
    # Naming
    " named ", " called ", " titled ",
    " llamado ", " titulado ",
    " anomenat ", " titulat ",
    # Explicit reference markers
    " ref:", " ref ", " id ", "id:", "code:", "código:", "codi:",
    "named after",
    "filename ", "file name ",
    "nombre de archivo", "nom de fitxer",
    # Operational lookups
    "specific ", "exact ",
    "específico ", "específica ", "exacto ", "exacta ",
    "específic ", "específica ", "exacte ", "exacta ",
)

_LEXICAL_FORMAT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b[A-Z]{2,5}-\d{2,6}\b"),    # internal codes: INV-2024-001
    re.compile(r"\b[A-Z0-9]{3,}_\d{3,}\b"),   # alt code forms: REF_001234
    re.compile(r'"[^"\n]+"'),                  # quoted exact phrase
    re.compile(r"«[^»\n]+»"),  # spanish/cat quotation marks « »
)

# Keywords that drag intent towards CONCEPTUAL (paraphrase / explanation).
_CONCEPTUAL_KEYWORDS: tuple[str, ...] = (
    # WH-style explanation requests
    "what is ", "what are ", "what does ", "what do ",
    "qué es ", "qué son ", "que es ", "que son ", "qué hace ", "qué significa ",
    "què és ", "què són ", "que es", "que son", "què fa ", "què significa ",
    # Imperative explanation
    "explain", "describe", "tell me about", "give me an overview",
    "expli", "describ", "cuéntame", "explícame", "descríbeme",
    "explica", "descriu", "explica'm", "descriu-me",
    # Summary
    "summary", "summarise", "summarize", "overview",
    "resumen", "resumir",
    "resum", "resumir",
    # Reasoning
    "why ", "how does", "how is ", "how can ", "in what way",
    "por qué", "porque ", "cómo ", "como ",
    "per què ", "perque ", "com ",
    # Comparison
    "compare", "comparison", "difference between", "differences between",
    "comparar", "comparación", "diferencia entre", "diferencias entre",
    "comparació", "diferència entre", "diferències entre",
    # High-level analysis
    "implications", "trade-offs", "tradeoffs", "consequences",
    "implicaciones", "consecuencias",
    "implicacions", "conseqüències",
    # Narrative / context
    "history of ", "background of ", "purpose of ", "goal of ", "objective of ",
    "historia de", "objetivo de", "finalidad de",
    "història de", "objectiu de", "finalitat de",
)


# ─────────────────────────────────────────────────────────────────────────────
# Compile keyword banks
# ─────────────────────────────────────────────────────────────────────────────

_PII_KEYWORD_PATTERNS = _compile_word_boundary(_PII_KEYWORDS)
_LEXICAL_KEYWORD_PATTERNS = _compile_word_boundary(_LEXICAL_KEYWORDS)
_CONCEPTUAL_KEYWORD_PATTERNS = _compile_word_boundary(_CONCEPTUAL_KEYWORDS)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def classify_intent(query: str) -> QueryIntent:
    """Return the dominant :class:`QueryIntent` of ``query``.

    Decision order (strongest signal first):

      1. Any **PII keyword** OR **PII format** match → :attr:`PII_EXACT`.
      2. Count of lexical hits vs conceptual hits:

         * lexical > conceptual          → :attr:`LEXICAL`
         * conceptual > lexical          → :attr:`CONCEPTUAL`
         * both zero or tied             → :attr:`MIXED`

    PII signals dominate because they are the most unambiguous: a query
    mentioning "IBAN" or matching an email regex is virtually certain to
    want a literal value, regardless of conceptual phrasing around it
    ("what is the IBAN of …" still benefits from BM25).

    Args:
        query: The raw user query. Must be non-empty after stripping.

    Returns:
        The dominant intent class.

    Raises:
        PipelineError: ``query`` is empty or whitespace-only.
    """
    if not query or not query.strip():
        raise PipelineError("Empty query passed to classify_intent.")

    # Strongest signal: PII keyword or PII format match
    for pat in _PII_KEYWORD_PATTERNS:
        if pat.search(query):
            return QueryIntent.PII_EXACT
    for pat in _PII_FORMAT_PATTERNS:
        if pat.search(query):
            return QueryIntent.PII_EXACT

    # Tally the other two intents
    lex_score: int = 0
    for pat in _LEXICAL_KEYWORD_PATTERNS:
        if pat.search(query):
            lex_score += 1
    for pat in _LEXICAL_FORMAT_PATTERNS:
        if pat.search(query):
            lex_score += 1

    con_score: int = 0
    for pat in _CONCEPTUAL_KEYWORD_PATTERNS:
        if pat.search(query):
            con_score += 1

    if lex_score == 0 and con_score == 0:
        return QueryIntent.MIXED
    if lex_score > con_score:
        return QueryIntent.LEXICAL
    if con_score > lex_score:
        return QueryIntent.CONCEPTUAL
    return QueryIntent.MIXED  # tie


def dynamic_alpha(query: str) -> tuple[float, QueryIntent]:
    """Convenience wrapper: classify ``query`` and return ``(alpha, intent)``.

    Args:
        query: The raw user query.

    Returns:
        ``(alpha, intent)`` where ``alpha`` ∈ [0, 1] is the vector-branch
        weight to pass to
        :func:`~domain.retrieval.rrf_fusion.calculate_dynamic_rrf`.

    Raises:
        PipelineError: forwarded from :func:`classify_intent`.
    """
    intent = classify_intent(query)
    return _INTENT_ALPHA[intent], intent


def alpha_for(intent: QueryIntent) -> float:
    """Return the configured α for a given :class:`QueryIntent`.

    Useful when an upstream component (a test, the orchestrator)
    already knows the intent and just wants the α mapping.
    """
    return _INTENT_ALPHA[intent]
