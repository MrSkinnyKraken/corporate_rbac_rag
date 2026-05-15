"""User identity as a domain value object.

The :class:`User` dataclass represents a *resolved* identity that the
application orchestrators consume. It has no relationship with how the
identity was authenticated — the api adapter is responsible for
translating whatever credential reaches it (a JSON-fixture `user_id` in
the demo phase, an OAuth bearer token as future "TO-DO" implementation) into a populated
:class:`User`. Domain code therefore stays clean of auth concerns.

The class is immutable (``frozen=True``) and uses the strongly-typed
:class:`~core.security.ClearanceLevel` and :class:`~core.security.Department`
enums for its security fields, so:

* No string drift across the call chain — passing the wrong literal
  ``"all"`` as a department is caught at construction.
* The router's :meth:`~domain.routing.router.HierarchicalRouter.route`
  and the retriever's
  :meth:`~domain.retrieval.ensemble_retriever.AsymmetricEnsembleRetriever.retrieve_secure_context`
  can keep their existing signatures: ``user.clearance_level`` and
  ``user.department`` plug straight in.

Construction failures raise :class:`~core.exceptions.MetadataValidationError`
(a :class:`SecurityError`), consistent with how the chunker validates
its own metadata — a malformed user is, like a malformed chunk, a Zero-
Trust violation that must never silently fall through.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.exceptions import MetadataValidationError
from core.security import ClearanceLevel, Department


@dataclass(frozen=True, slots=True)
class User:
    """A resolved identity ready for use by the application orchestrators.

    Attributes:
        user_id: Opaque, stable identifier. Persisted in audit logs so
            "who uploaded this document" / "who issued that query"
            survives the demo phase. Trim of surrounding whitespace and
            non-empty enforced at construction.
        username: Human-readable display name shown in UI dropdowns and
            audit reports. Free-form.
        clearance_level: The user's security clearance. Drives the
            severity gate of :func:`~domain.retrieval.rbac_filter.apply_zero_trust_filter`.
        department: The user's home department. Drives the scope gate
            of the same filter. Must be a real
            :class:`~core.security.Department` enum value — the wildcard
            ``"all"`` of :class:`~core.security.AllowedDepartment` is a
            *resource* attribute and must never appear on the subject
            side. Enforced at construction.
        email: Optional contact address. Reserved for the audit log and
            for future notification flows; never used as an identity
            key (the api adapter looks users up by ``user_id``).
    """

    user_id: str
    username: str
    clearance_level: ClearanceLevel
    department: Department
    email: str | None = None

    def __post_init__(self) -> None:
        """Validate the security-relevant fields at construction time.

        Mirrors :class:`~domain.chunking.core_chunker.ChunkInput.__post_init__`:
        anything that touches the Zero-Trust predicate is checked the
        moment the object is built, so a malformed ``User`` cannot
        reach the retriever / router as a string-typed department or
        an out-of-range clearance integer.
        """
        if not self.user_id or not self.user_id.strip():
            raise MetadataValidationError(
                "User.user_id must be a non-empty string."
            )
        if not self.username or not self.username.strip():
            raise MetadataValidationError(
                f"User.username must be a non-empty string (user_id={self.user_id!r})."
            )
        if not isinstance(self.clearance_level, ClearanceLevel):
            raise MetadataValidationError(
                f"User.clearance_level must be a ClearanceLevel enum, got "
                f"{type(self.clearance_level).__name__}={self.clearance_level!r} "
                f"(user_id={self.user_id!r})."
            )
        if not isinstance(self.department, Department):
            raise MetadataValidationError(
                f"User.department must be a Department enum (not "
                f"AllowedDepartment / str); got "
                f"{type(self.department).__name__}={self.department!r} "
                f"(user_id={self.user_id!r})."
            )
        if self.email is not None and not self.email.strip():
            raise MetadataValidationError(
                f"User.email, when provided, must be non-empty "
                f"(user_id={self.user_id!r})."
            )
