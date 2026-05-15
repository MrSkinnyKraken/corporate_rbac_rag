"""Cross-department visibility rules (per-chunk monotonic expansion).

Every chunk emitted by :class:`~domain.chunking.core_chunker.CustomRBACChunker`
inherits the document's ``home_department`` as its default
``allowed_departments`` value. The application orchestrator (
:class:`~application.ingestion_app.IngestionApp`) then overlays the
LLM-proposed (and user-confirmed) **baseline** ``allowed_departments``
on every chunk, AND — this module's job — applies a registry of regex
rules per chunk that may *extend* the baseline with additional
departments whose content footprint appears in the chunk's text.

The expansion is **monotonic by design**:

    final_chunk.allowed_departments  ⊇  baseline_from_LLM

A regex rule can never *remove* a department inherited from the
baseline; it can only *add*. This makes the broadening safe by
construction — the LLM marks the floor, rules can only lift the
ceiling — and aligns with the chunker docstring's TODO ("apply rule-
based heuristics in the orchestrator").

The rules are deliberately pure regex (no LLM, no I/O) so:

* Ingestion latency stays bounded — no per-chunk LLM call.
* Behaviour is reproducible and auditable: a rule that fires is a
  deterministic property of the chunk's text.
* The same patterns can be reused by a future "explain why this user
  saw this chunk" diagnostic without touching infrastructure.

Each rule names the departments it *grants visibility to* — never the
inverse. The convention parallels :data:`~domain.chunking.patterns.PII_PATTERNS`
in spirit: a regex match adds a *capability tag*, the chunker / orchestrator
maps it to a policy decision.

Multilingual coverage (EN / ES / CA) mirrors what we already do in
:mod:`~domain.routing.intent_classifier`: the demo corpus mixes the
three languages and the patterns must catch all of them. Word-boundary
matching (``\\b…\\b``) prevents the "cif inside specific" class of
false positives discussed during the routing-step development.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final, Pattern

from core.security import ALLOWED_DEPARTMENT_WILDCARD, AllowedDepartment


@dataclass(frozen=True, slots=True)
class CrossDepartmentRule:
    """One cross-department visibility rule.

    Attributes:
        name:        Stable identifier (snake_case). Logged in the
            chunk's ``cross_dept_rules_fired`` audit field and used as
            the key in :attr:`~application.ingestion_app.IngestionReport.cross_dept_rules_summary`.
        compiled:    A pre-compiled regex with ``re.IGNORECASE``. The
            rule fires when :meth:`Pattern.search` returns a match
            anywhere in the chunk's text.
        grants_to:   The department whose users gain read visibility on
            chunks where the rule fires. Always a real
            :class:`~core.security.AllowedDepartment` member — never
            the ``ALL`` wildcard (wildcards are absorbing and bypass
            the rules, see :func:`expand_allowed_departments`).
        description: Free-form explanation surfaced to operators
            reviewing the rule set; never persisted to chunk metadata.
    """

    name: str
    compiled: Pattern[str]
    grants_to: AllowedDepartment
    description: str


# ─────────────────────────────────────────────────────────────────────────────
# Rule registry
# ─────────────────────────────────────────────────────────────────────────────
#
# Conventions:
#  * One rule per department-as-grantee. Multiple rules for the same
#    grantee are allowed when their patterns are semantically distinct
#    enough to be auditable separately.
#  * Each pattern uses word boundaries on every alternation arm.
#  * Multilingual: EN / ES / CA. Accented characters are included
#    literally — Python's `re` with default flags treats `\b` as a
#    Unicode word boundary, so "nómina" matches but "nóminas-cobrar"
#    does not (no boundary between "ó" and "n").
#  * No rule grants `ALL`. ALL is a *resource* property the LLM sets on
#    fully-public documents; it bypasses the rules entirely.

CROSS_DEPT_RULES: Final[list[CrossDepartmentRule]] = [
    CrossDepartmentRule(
        name="hr_topics",
        compiled=re.compile(
            r"\b("
            r"employee|employees|salary|salaries|wage|wages|payroll|"
            r"benefit|benefits|vacation|leave|onboarding|hiring|hire|"
            r"termination|terminate|terminated|"
            r"empleado|empleados|empleada|empleadas|sueldo|sueldos|salario|salarios|"
            r"n[oó]mina|n[oó]minas|vacaciones|contrataci[oó]n|despido|despidos|"
            r"empleat|empleats|empleada|empleades|sou|sous|salari|salaris|"
            r"vacances|contractaci[oó]|acomiadament|acomiadaments"
            r")\b",
            re.IGNORECASE,
        ),
        grants_to=AllowedDepartment.HR,
        description=(
            "Grants HR visibility on chunks mentioning employees, "
            "compensation, benefits, hiring or terminations (EN/ES/CA)."
        ),
    ),
    CrossDepartmentRule(
        name="legal_compliance",
        compiled=re.compile(
            r"\b("
            r"GDPR|HIPAA|compliance|liability|warranty|warranties|clause|clauses|"
            r"jurisdiction|contractual|contract|contracts|"
            r"cumplimiento|responsabilidad|garant[ií]a|garant[ií]as|cl[áa]usula|cl[áa]usulas|"
            r"jurisdicci[oó]n|contrato|contratos|"
            r"compliment|responsabilitat|garantia|garanties|cl[àa]usula|cl[àa]usules|"
            r"jurisdicci[oó]|contracte|contractes"
            r")\b",
            re.IGNORECASE,
        ),
        grants_to=AllowedDepartment.LEGAL,
        description=(
            "Grants Legal visibility on chunks mentioning compliance, "
            "warranty, contracts, clauses or jurisdiction (EN/ES/CA)."
        ),
    ),
    CrossDepartmentRule(
        name="finance_topics",
        compiled=re.compile(
            r"\b("
            r"invoice|invoices|budget|budgets|revenue|revenues|EBITDA|forecast|"
            r"expense|expenses|payment|payments|refund|refunds|cost|costs|"
            r"factura|facturas|presupuesto|presupuestos|ingreso|ingresos|"
            r"coste|costes|gasto|gastos|pago|pagos|reembolso|reembolsos|"
            r"factura|factures|pressupost|pressupostos|ingr[eé]s|ingressos|"
            r"cost|costos|despesa|despeses|pagament|pagaments|reembors|reemborsos"
            r")\b",
            re.IGNORECASE,
        ),
        grants_to=AllowedDepartment.FINANCE,
        description=(
            "Grants Finance visibility on chunks mentioning invoicing, "
            "budgeting, revenue, costs or payments (EN/ES/CA)."
        ),
    ),
    CrossDepartmentRule(
        name="engineering_topics",
        compiled=re.compile(
            r"\b("
            r"API|endpoint|endpoints|deployment|infrastructure|server|servers|"
            r"database|databases|codebase|repository|repo|microservice|microservices|"
            r"despliegue|infraestructura|servidor|servidores|base\ de\ datos|"
            r"despliegues|c[oó]digo\ fuente|repositorio|repositorios|"
            r"desplegament|infraestructura|servidor|servidors|base\ de\ dades|"
            r"desplegaments|codi\ font|repositori|repositoris"
            r")\b",
            re.IGNORECASE,
        ),
        grants_to=AllowedDepartment.ENGINEERING,
        description=(
            "Grants Engineering visibility on chunks mentioning APIs, "
            "deployments, infrastructure, servers or databases (EN/ES/CA)."
        ),
    ),
    CrossDepartmentRule(
        name="operations_topics",
        compiled=re.compile(
            r"\b("
            r"logistics|supply\ chain|inventory|procurement|vendor|vendors|supplier|suppliers|"
            r"fulfillment|warehouse|warehouses|shipment|shipments|"
            r"log[ií]stica|cadena\ de\ suministro|inventario|inventarios|"
            r"proveedor|proveedores|aprovisionamiento|almac[eé]n|almacenes|env[ií]o|env[ií]os|"
            r"log[ií]stica|cadena\ de\ subministrament|inventari|inventaris|"
            r"prove[ïi]dor|prove[ïi]dors|aprovisionament|magatzem|magatzems|enviament|enviaments"
            r")\b",
            re.IGNORECASE,
        ),
        grants_to=AllowedDepartment.OPERATIONS,
        description=(
            "Grants Operations visibility on chunks mentioning logistics, "
            "supply chain, vendors, inventory or fulfillment (EN/ES/CA)."
        ),
    ),
    CrossDepartmentRule(
        name="sales_topics",
        compiled=re.compile(
            r"\b("
            r"customer|customers|client|clients|lead|leads|prospect|prospects|"
            r"pipeline|opportunity|opportunities|quota|quotas|"
            r"cliente|clientes|prospecto|prospectos|oportunidad|oportunidades|cuota|cuotas|"
            r"client|clients|prospecte|prospectes|oportunitat|oportunitats|quota|quotes"
            r")\b",
            re.IGNORECASE,
        ),
        grants_to=AllowedDepartment.SALES,
        description=(
            "Grants Sales visibility on chunks mentioning customers, leads, "
            "prospects, pipeline or sales quotas (EN/ES/CA)."
        ),
    ),
    CrossDepartmentRule(
        name="marketing_topics",
        compiled=re.compile(
            r"\b("
            r"campaign|campaigns|brand|branding|audience|audiences|"
            r"conversion|conversions|advertising|advertisement|advertisements|"
            r"campa[ñn]a|campa[ñn]as|marca|audiencia|audiencias|conversi[oó]n|conversiones|"
            r"publicidad|anuncio|anuncios|"
            r"campanya|campanyes|marca|audi[eè]ncia|audi[eè]ncies|conversi[oó]|conversions|"
            r"publicitat|anunci|anuncis"
            r")\b",
            re.IGNORECASE,
        ),
        grants_to=AllowedDepartment.MARKETING,
        description=(
            "Grants Marketing visibility on chunks mentioning campaigns, "
            "branding, audience, conversion or advertising (EN/ES/CA)."
        ),
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# Pure expansion function
# ─────────────────────────────────────────────────────────────────────────────

def expand_allowed_departments(
    chunk_text: str,
    baseline: list[AllowedDepartment],
    rules: list[CrossDepartmentRule] = CROSS_DEPT_RULES,
) -> tuple[list[AllowedDepartment], list[str]]:
    """Return the chunk's final ``allowed_departments`` plus the rules that fired.

    The function is **monotonic in the baseline**: every department in
    ``baseline`` is present in the returned list, regardless of which
    rules match. Only *new* departments may be added.

    Wildcard short-circuit: if ``baseline`` already contains
    :attr:`~core.security.AllowedDepartment.ALL`, the list is absorbing
    — no rule can broaden a document that is already fully public —
    and the function returns ``(baseline, [])`` unchanged.

    Args:
        chunk_text: The chunk's raw text. Searched once per rule.
        baseline:   The LLM-proposed (and user-confirmed) document-
            level ``allowed_departments`` list. Order is preserved
            in the result; additions are appended in the order their
            rules appear in ``rules``.
        rules:      The rule registry to apply. Defaults to
            :data:`CROSS_DEPT_RULES`. Tests may pass a smaller set.

    Returns:
        ``(final, fired)``:
          * ``final``: the effective ``allowed_departments`` for the
            chunk — always ⊇ ``baseline``, with potential additions.
            Returned in a deterministic order (baseline first, then
            sorted alphabetical for the new additions) so the JSON
            encoding stays stable across runs.
          * ``fired``: the names of every rule that matched, in the
            order they were evaluated. Persisted to the chunk's
            ``cross_dept_rules_fired`` metadata field for audit.

    Raises:
        TypeError: ``baseline`` is empty or contains non-AllowedDepartment
            entries — the caller violated the upstream contract.
    """
    if not baseline:
        raise TypeError(
            "expand_allowed_departments: baseline must be a non-empty list "
            "of AllowedDepartment values."
        )
    for d in baseline:
        if not isinstance(d, AllowedDepartment):
            raise TypeError(
                f"expand_allowed_departments: baseline entries must be "
                f"AllowedDepartment members; got {type(d).__name__}={d!r}."
            )

    # Wildcard short-circuit: ALL is absorbing.
    baseline_values: set[str] = {d.value for d in baseline}
    if ALLOWED_DEPARTMENT_WILDCARD in baseline_values:
        return list(baseline), []

    granted: set[AllowedDepartment] = set(baseline)
    fired: list[str] = []

    for rule in rules:
        if rule.grants_to in granted:
            continue  # already in baseline — no rule firing recorded
        if rule.compiled.search(chunk_text):
            granted.add(rule.grants_to)
            fired.append(rule.name)

    additions: list[AllowedDepartment] = sorted(
        granted - set(baseline), key=lambda d: d.value
    )
    return list(baseline) + additions, fired
