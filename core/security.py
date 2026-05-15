"""Security taxonomy for the Zero-Trust RAG system.

This module defines the canonical types used by the RBAC layer:

- :class:`ClearanceLevel` — the 0-to-3 severity ladder governing the
  ``chunk.clearance_level <= user.clearance_level`` predicate.
- :class:`Department` — real corporate departments. Used as the value of the
  ``department`` metadata field, which determines a chunk's *physical home*
  (the unique collection in which it is indexed and from which it is
  retrieved). Never contains the ``"all"`` wildcard.
- :class:`AllowedDepartment` — values valid inside the ``allowed_departments``
  list. Includes every :class:`Department` member plus the ``ALL`` wildcard,
  which grants read permission to any user regardless of department.

The split between :class:`Department` and :class:`AllowedDepartment` encodes
at the type level the architectural decision adopted at the end of Phase 3
Step 4: physical storage is partitioned by concrete department, while read
permission can extend to any subset of departments — including everyone.
"""

from __future__ import annotations

from enum import Enum, IntEnum
from typing import Final


class ClearanceLevel(IntEnum):
    """Confidentiality severity of a chunk's content (or a user's privilege).

    The Zero-Trust filter at the retrieval layer enforces:

    .. code-block:: text

        chunk.clearance_level <= user.clearance_level

    A user with :attr:`PUBLIC` clearance may only see :attr:`PUBLIC` chunks;
    a user with :attr:`STRICT` clearance may see chunks at every level.
    """

    PUBLIC = 0
    """Safe to expose to anyone, internal or external. Examples: product
    specifications, user manuals, marketing material."""

    INTERNAL = 1
    """Internal-only but not sensitive. Examples: company-wide policies,
    onboarding procedures, password complexity *rules* (not values)."""

    CONFIDENTIAL = 2
    """Department-scoped sensitive material. Examples: client contracts,
    server logs, incident reports, internal code names, moderate financial
    detail."""

    STRICT = 3
    """Highly sensitive content whose leak would cause legal, financial, or
    market damage. Examples: raw credentials, IBAN/SWIFT, DNI/national IDs,
    salary bands, M&A or layoff plans, board meeting minutes."""

    @classmethod
    def from_int(cls, value: int) -> ClearanceLevel:
        """Build a :class:`ClearanceLevel` from a raw integer.

        Args:
            value: an integer in the closed range ``[0, 3]``.

        Returns:
            The matching :class:`ClearanceLevel` member.

        Raises:
            ValueError: if ``value`` is not one of ``{0, 1, 2, 3}``.
        """
        try:
            return cls(value)
        except ValueError as exc:
            raise ValueError(
                f"Invalid clearance_level={value!r}; expected one of "
                f"{[m.value for m in cls]}."
            ) from exc

    def __str__(self) -> str:  # pragma: no cover  (cosmetic)
        return f"{self.name} ({self.value})"


class Department(str, Enum):
    """A real corporate department.

    Used as the value of the ``department`` metadata field on documents and
    chunks: this determines the *physical home* of the chunk — the unique
    ChromaDB collection (and matching BM25 index) in which it is stored.

    The hierarchical router returns one of these values per top document; it
    never returns the ``"all"`` wildcard, because there is intentionally no
    ``"all"`` collection in the production architecture.
    """

    HR = "hr"
    FINANCE = "finance"
    ENGINEERING = "engineering"
    LEGAL = "legal"
    SALES = "sales"
    OPERATIONS = "operations"
    MARKETING = "marketing"

    def __str__(self) -> str:  # pragma: no cover  (cosmetic)
        return self.value


class AllowedDepartment(str, Enum):
    """A value valid inside the ``allowed_departments`` metadata list.

    Includes every :class:`Department` member plus the special :attr:`ALL`
    wildcard, which grants read permission to every user. The chunk-level
    RBAC predicate evaluated by the retrieval layer is:

    .. code-block:: text

        allowed_for_user(chunk, user) :=
            chunk.clearance_level <= user.clearance_level
            AND
            (
                user.department.value in chunk.allowed_departments
                OR
                AllowedDepartment.ALL.value in chunk.allowed_departments
            )

    A chunk's ``allowed_departments`` may be, for example,
    ``["finance", "hr"]`` (the document is owned by finance but readable by
    HR too) or ``["all"]`` (company-wide content such as a public manual).
    Both shapes are expressible with this enum.
    """

    HR = "hr"
    FINANCE = "finance"
    ENGINEERING = "engineering"
    LEGAL = "legal"
    SALES = "sales"
    OPERATIONS = "operations"
    MARKETING = "marketing"
    ALL = "all"

    def __str__(self) -> str:  # pragma: no cover  (cosmetic)
        return self.value


#: Convenience constant for code paths that handle raw strings rather than
#: typed enums (e.g., the Custom RBAC chunker, JSON serialisation, ChromaDB
#: ``where`` filters constructed dynamically).
ALLOWED_DEPARTMENT_WILDCARD: Final[str] = AllowedDepartment.ALL.value
