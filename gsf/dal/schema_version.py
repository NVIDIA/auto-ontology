# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Is the database's schema the one this code expects?

Two callers, one rule. The server refuses to start when the answer is no, and
``/api/health`` reports it — because the failure is otherwise invisible in the
worst way: the process starts, ``/api/health`` returns ``200 {"status": "ok"}``
on the strength of a working connection, and every catalog endpoint answers
``500 UndefinedTable`` with nothing pointing at the cause.

This is the same rule the chart's ``gsf.waitForCatalogSchema`` init container
applies before letting the backend start, restated in-process for the people not
deploying with Helm: ``python -m gsf.server``, docker compose, a bare container.

It deliberately does **not** migrate. Applying DDL from an application process
races every other replica doing the same, which is why the chart runs migrations
as their own Job. Reporting the mismatch is safe from any number of processes;
fixing it is a decision the operator makes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from sqlalchemy import text

from gsf.dal.session import get_engine

logger = logging.getLogger(__name__)

#: Where the migrations live, relative to the installed package. The image
#: copies `alembic/` and `alembic.ini` next to `gsf/` (see the Dockerfile), and
#: a source checkout has the same shape, so one path serves both.
_REPO_ROOT = Path(__file__).resolve().parents[2]

_UPGRADE_COMMAND = "uv run alembic upgrade head"


@dataclass(frozen=True)
class SchemaState:
    """What the database says, against what the code expects."""

    expected: str | None
    applied: str | None
    #: Set when the revision could not be read at all. Kept apart from
    #: ``applied is None`` because the two need different advice: an
    #: unmigrated database is fixed by running the migration, an unreachable
    #: one is not, and telling an operator to migrate a database nothing can
    #: connect to sends them down the wrong path.
    unreachable: str | None = None

    @property
    def current(self) -> bool:
        """True only when both are known and equal.

        An unknown *expected* (no migrations found on disk) counts as not
        current rather than as fine: a packaging mistake that hides the
        migrations should be loud, not silently permissive.
        """
        return bool(self.expected) and self.expected == self.applied

    @property
    def detail(self) -> str:
        if self.unreachable is not None:
            return f"could not read the applied revision: {self.unreachable}"
        if self.expected is None:
            return (
                "cannot determine the expected revision — no alembic migrations "
                f"found under {_REPO_ROOT / 'alembic'}"
            )
        if self.applied is None:
            return (
                "the database has no alembic_version table, so no migration has "
                f"ever been applied to it. Run: {_UPGRADE_COMMAND}"
            )
        return (
            f"the database is at revision {self.applied}, but this build expects "
            f"{self.expected}. Run: {_UPGRADE_COMMAND}"
        )


@lru_cache(maxsize=1)
def head_revision() -> str | None:
    """The revision this build's migrations end at, or ``None`` if unreadable.

    Cached: the migration scripts cannot change under a running process, and
    reading them costs a directory scan plus an import of every version file.
    """
    try:
        from alembic.config import Config
        from alembic.script import ScriptDirectory

        config = Config(str(_REPO_ROOT / "alembic.ini"))
        config.set_main_option("script_location", str(_REPO_ROOT / "alembic"))
        heads = ScriptDirectory.from_config(config).get_heads()
    except Exception:
        logger.warning("Could not read the alembic head revision", exc_info=True)
        return None

    if len(heads) != 1:
        # Branched history: nothing single to compare against, and the operator
        # has a merge to do before any of this is meaningful.
        logger.warning("Expected exactly one alembic head, found %s", heads or "none")
        return None
    return heads[0]


def applied_revision() -> tuple[str | None, str | None]:
    """``(revision, unreachable_reason)`` for the database's stamped revision.

    Never raises — the health endpoint has to answer even with the database
    down — but it does separate the two ways of getting nothing back. A missing
    ``alembic_version`` table is a migration that has not run yet
    (``UndefinedTable``); anything else means the answer is unknown, and the
    two want different advice.
    """
    try:
        with get_engine().connect() as connection:
            revision = connection.execute(
                text("SELECT version_num FROM public.alembic_version")
            ).scalar()
        return revision, None
    except Exception as exc:
        if _is_missing_table(exc):
            return None, None
        return None, (str(exc) or type(exc).__name__)[:200]


def _is_missing_table(exc: BaseException) -> bool:
    """Is this "the table isn't there", as opposed to "the database isn't"?

    Matched on psycopg's ``UndefinedTable`` by name rather than by import so a
    driver swap degrades to "unreachable" — the conservative reading — instead
    of raising a new error from inside the error path.
    """
    seen = exc
    while seen is not None:
        if type(seen).__name__ == "UndefinedTable":
            return True
        seen = seen.__cause__ or seen.__context__
    return False


def schema_state() -> SchemaState:
    """Compare what the database has against what this build expects."""
    applied, unreachable = applied_revision()
    return SchemaState(
        expected=head_revision(), applied=applied, unreachable=unreachable
    )


class SchemaOutOfDateError(RuntimeError):
    """The database's schema is not the one this build expects."""


def _raise_for(state: SchemaState) -> None:
    """The refusal message, as its own function so it can be tested.

    Split from :func:`require_current_schema` because the wording is the part
    that matters — it is the only thing an operator sees — and asserting on it
    should not require standing up a database in each of the failing states.
    """
    raise SchemaOutOfDateError(f"Refusing to start: {state.detail}")


def require_current_schema() -> None:
    """Raise unless the database is at this build's head revision.

    Called from the server's lifespan, so an operator gets one actionable line
    at startup instead of a working ``/api/health`` and 500s everywhere else.
    """
    state = schema_state()
    if state.current:
        logger.info("Database schema is at the expected revision %s", state.applied)
        return
    _raise_for(state)
