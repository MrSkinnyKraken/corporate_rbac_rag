"""User store adapter: identity persistence behind an interface.

Two pieces:

  * :class:`UserStore`     — abstract contract every implementation
    must satisfy. Two methods: :meth:`get` (resolve a single
    ``user_id``) and :meth:`list_all` (enumerate every available
    identity, used by the demo Streamlit dropdown).
  * :class:`JsonUserStore` — a flat-file implementation that reads a
    JSON fixture at construction time and serves lookups from memory.

The abstraction keeps domain code clean of identity persistence
concerns: tomorrow an ``OAuthUserStore`` or ``DatabaseUserStore`` will
slot in via constructor injection without touching
``application/`` or ``domain/``. The api adapter is the only layer
that selects which implementation to bind.

Failure model:

  * The fixture path missing / unreadable / malformed → :class:`ConfigurationError`
    (this is a deployment-time misconfiguration, not a per-request event).
  * A requested ``user_id`` that doesn't exist → :class:`UserNotFoundError`
    (the api adapter translates this to 401/403 — never a 500).
  * A fixture entry whose field types are invalid (e.g. clearance "9",
    department "all") → :class:`ConfigurationError` raised at load
    time, NOT lazily on the per-request `get()` — operations needs to
    know on startup that the fixture is broken.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from core.config import Settings, get_settings
from core.exceptions import (
    ConfigurationError,
    MetadataValidationError,
    UserNotFoundError,
)
from core.security import ClearanceLevel, Department
from domain.users import User


class UserStore(ABC):
    """Abstract user store. Every backend must satisfy this contract."""

    @abstractmethod
    def get(self, user_id: str) -> User:
        """Resolve a single identity.

        Raises:
            UserNotFoundError: ``user_id`` is not registered.
        """

    @abstractmethod
    def list_all(self) -> list[User]:
        """Return every registered identity.

        Order is stable across calls within a process. Used by the
        Streamlit demo to populate its user-picker dropdown.
        """


class JsonUserStore(UserStore):
    """A user store backed by a flat JSON fixture file.

    Expected file shape — a top-level list of objects, each carrying::

        {
          "user_id":         "u-001",
          "username":        "Anna Garcia",
          "clearance_level": 1,
          "department":      "hr",
          "email":           "anna@demo.local"
        }

    * ``clearance_level`` is an int 0-3 (``PUBLIC``, ``INTERNAL``,
      ``CONFIDENTIAL``, ``STRICT``); coerced via
      :meth:`ClearanceLevel.from_int`.
    * ``department`` is a string matching one of
      :class:`~core.security.Department` values (case-insensitive).
      The wildcard ``"all"`` is rejected — subjects never carry it.
    * ``email`` is optional.

    The fixture is read **once at construction**. Hot-reload is not
    supported — the demo restarts the process on fixture changes. This
    avoids file-system race conditions during requests and keeps
    :meth:`get` a pure dict lookup.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """Load and validate the fixture.

        Args:
            settings: Application settings. The fixture path is read
                from ``settings.users_fixture_path``.

        Raises:
            ConfigurationError: the fixture is missing, unreadable, or
                contains a malformed entry.
        """
        self._settings: Settings = settings or get_settings()
        self._users: dict[str, User] = {}
        self._load()

    # ─────────────────────────── Public API ─────────────────────────────────
    def get(self, user_id: str) -> User:
        try:
            return self._users[user_id]
        except KeyError as exc:
            raise UserNotFoundError(user_id) from exc

    def list_all(self) -> list[User]:
        # dict insertion order is preserved on every supported Python ≥3.7
        return list(self._users.values())

    @property
    def fixture_path(self) -> Path:
        """The absolute path the fixture was read from."""
        return self._settings.users_fixture_path

    # ────────────────────────────── Internals ───────────────────────────────
    def _load(self) -> None:
        """Parse the fixture file into :class:`User` records.

        Each step that can fail attributes its own actionable error
        message — operations should be able to read the message alone
        and know which entry of which file is broken.
        """
        path: Path = self._settings.users_fixture_path
        try:
            raw_text = path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise ConfigurationError(
                f"User fixture not found at {path}. "
                f"Set USERS_FIXTURE_PATH in .env or place the demo file there."
            ) from exc
        except OSError as exc:
            raise ConfigurationError(
                f"User fixture at {path} is unreadable: {exc}"
            ) from exc

        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise ConfigurationError(
                f"User fixture at {path} is not valid JSON: "
                f"line {exc.lineno}, column {exc.colno}: {exc.msg}"
            ) from exc

        if not isinstance(parsed, list):
            raise ConfigurationError(
                f"User fixture at {path} must be a JSON array of objects; "
                f"got top-level {type(parsed).__name__}."
            )

        for idx, raw in enumerate(parsed):
            if not isinstance(raw, dict):
                raise ConfigurationError(
                    f"User fixture at {path}, entry #{idx}: expected a JSON "
                    f"object, got {type(raw).__name__}."
                )
            user = self._build_user(path, idx, raw)
            if user.user_id in self._users:
                raise ConfigurationError(
                    f"User fixture at {path}: duplicate user_id "
                    f"{user.user_id!r} (entry #{idx})."
                )
            self._users[user.user_id] = user

    @staticmethod
    def _build_user(path: Path, idx: int, raw: dict[str, Any]) -> User:
        """Convert one raw dict into a typed :class:`User`."""
        def _require(key: str) -> Any:
            if key not in raw:
                raise ConfigurationError(
                    f"User fixture at {path}, entry #{idx}: missing "
                    f"required field {key!r}."
                )
            return raw[key]

        user_id = _require("user_id")
        username = _require("username")
        cl_raw = _require("clearance_level")
        dept_raw = _require("department")
        email_raw = raw.get("email")

        # Clearance: int → ClearanceLevel
        try:
            clearance = ClearanceLevel.from_int(int(cl_raw))
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(
                f"User fixture at {path}, entry #{idx} (user_id={user_id!r}): "
                f"invalid clearance_level {cl_raw!r} — must be int in 0..3."
            ) from exc

        # Department: string → Department enum (reject wildcard)
        try:
            department = Department(str(dept_raw).strip().lower())
        except ValueError as exc:
            valid = ", ".join(sorted(d.value for d in Department))
            raise ConfigurationError(
                f"User fixture at {path}, entry #{idx} (user_id={user_id!r}): "
                f"invalid department {dept_raw!r}. Subjects must carry a real "
                f"Department (not the 'all' wildcard); valid values: {valid}."
            ) from exc

        # Email: None | non-empty string
        email: str | None
        if email_raw is None:
            email = None
        elif isinstance(email_raw, str) and email_raw.strip():
            email = email_raw.strip()
        else:
            raise ConfigurationError(
                f"User fixture at {path}, entry #{idx} (user_id={user_id!r}): "
                f"invalid email {email_raw!r} — must be a non-empty string or omitted."
            )

        try:
            return User(
                user_id=str(user_id).strip(),
                username=str(username).strip(),
                clearance_level=clearance,
                department=department,
                email=email,
            )
        except MetadataValidationError as exc:
            # Bubble up as configuration: the fixture is the configuration source.
            raise ConfigurationError(
                f"User fixture at {path}, entry #{idx} (user_id={user_id!r}): "
                f"{exc}"
            ) from exc
