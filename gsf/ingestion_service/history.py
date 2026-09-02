# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Postgres-backed history of semantic compilation passes.

Unlike the ``configurations`` table (owned by the frontend's Prisma schema),
this table is GSF's own: written and read only by Python services. It is
declared in :mod:`gsf.dal.schema` and created by a migration like every other
GSF table, so nothing here creates it — the reads and writes below assume it
exists, and degrade to no-ops if it somehow doesn't.

The scheduler records a start/end row for every pass (see
``semantic_scheduler.py``); the ingestion service's status endpoint reads the
last successful — and, if more recent, the last failed — one back so the
settings page can show either without a second, frontend-owned source of
truth.

``status`` is a nullable string outcome — ``RUN_SUCCEEDED``, ``RUN_FAILED``,
or ``NULL``. A row is only ever written to when the pass reaches one of those
two terminal outcomes; there is no "aborted" value. That means ``NULL`` covers
every other case — still running, explicitly stopped mid-run, disabled
mid-run, or orphaned by a crash (SIGKILL, OOM, power loss — anything that
skips the ``finally`` in ``semantic_scheduler.py``) — and stays that way
forever for a row nobody ever finishes. Callers that need to know whether a
pass is *currently* running use the scheduler's own in-memory state instead
(see ``semantic_status`` in ``router.py``), so this table never needs to
distinguish "still going" from "gave up without a verdict".
"""

from __future__ import annotations

import logging

import psycopg

from gsf.dal.schema import RUN_FAILED, RUN_SUCCEEDED
from gsf.infra.postgres import get_postgres_connection_string

logger = logging.getLogger(__name__)

# The only two outcomes ever written to the ``status`` column — see the module
# docstring on why there's no "aborted" value. Re-exported from the schema that
# declares the CHECK constraint enforcing them, rather than restated here: two
# copies of these strings differ only when one is wrong, and the write would
# then fail against the constraint at runtime.
__all__ = [
    "RUN_FAILED",
    "RUN_SUCCEEDED",
    "get_last_failure_if_most_recent",
    "get_last_successful_run",
    "record_run_finish",
    "record_run_start",
]

_VALID_OUTCOMES = frozenset({RUN_SUCCEEDED, RUN_FAILED})


def record_run_start() -> int | None:
    """Insert a row for a pass starting now; returns its id.

    ``status`` takes its column default (``NULL``) and stays that way unless
    ``record_run_finish`` later overwrites it with a terminal outcome — see
    the module docstring.

    Returns ``None`` on failure (best-effort — a DB hiccup must not block a
    compilation pass). Callers should skip ``record_run_finish`` in that case,
    since there is no row to update.
    """
    try:
        with psycopg.connect(
            get_postgres_connection_string(), connect_timeout=3
        ) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO semantic_compilation_history (started_at)
                    VALUES (now())
                    RETURNING id
                    """
                )
                row = cur.fetchone()
            conn.commit()
        return int(row[0]) if row else None
    except Exception:
        logger.exception("Failed to record semantic compilation run start")
        return None


def record_run_finish(run_id: int | None, outcome: str | None) -> None:
    """Mark a previously-started run as finished with *outcome*.

    *outcome* must be ``RUN_SUCCEEDED``, ``RUN_FAILED``, or ``None``. Passing
    ``None`` (an explicit stop, a mid-run disable, or anything else short of
    an actual success/failure) is a no-op: the row is left exactly as
    ``record_run_start`` inserted it, i.e. ``finished_at``/``status`` both
    stay ``NULL`` — see the module docstring on why that's indistinguishable
    from "still running" on purpose.

    Also no-ops when *run_id* is ``None`` — either the start was never
    recorded, or the table wasn't reachable at the time. Best-effort like the
    rest of this module: a DB error here must not fail the compilation pass
    itself.
    """
    if run_id is None or outcome is None:
        return
    if outcome not in _VALID_OUTCOMES:
        raise ValueError(f"invalid semantic compilation run outcome: {outcome!r}")
    try:
        with psycopg.connect(
            get_postgres_connection_string(), connect_timeout=3
        ) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE semantic_compilation_history
                    SET finished_at = now(), status = %s
                    WHERE id = %s
                    """,
                    (outcome, run_id),
                )
            conn.commit()
    except Exception:
        logger.exception(
            "Failed to record semantic compilation run finish (id=%s)", run_id
        )


def get_last_successful_run() -> str | None:
    """Return the ISO 8601 timestamp of the most recently *finished* successful pass.

    Best-effort: any DB error (including the table not existing yet) is
    treated as "unknown" rather than raised, so a missing/unreachable table
    never breaks the status endpoint.
    """
    try:
        with psycopg.connect(
            get_postgres_connection_string(), connect_timeout=3
        ) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT finished_at FROM semantic_compilation_history
                    WHERE status = '{RUN_SUCCEEDED}'
                    ORDER BY finished_at DESC
                    LIMIT 1
                    """
                )
                row = cur.fetchone()
    except Exception:
        logger.exception("Failed to read last successful semantic compilation run")
        return None

    return row[0].isoformat() if row and row[0] is not None else None


def get_last_failure_if_most_recent() -> str | None:
    """Return the ISO 8601 timestamp of the last failed pass, but only if it is
    more recent than the last *successful* one.

    "Failed" means the pass actually ran and raised — a row with
    ``status = 'failed'``, as opposed to one still ``NULL`` (still running,
    explicitly stopped, disabled mid-run, or orphaned by a crash; see the
    module docstring). Those all stay invisible here on purpose: an aborted
    run isn't a verdict, so it shouldn't read as a failure on the settings
    page.

    Scoping to "more recent than the last success" means a stale failure from
    days ago stops being surfaced the moment a later run succeeds, without the
    caller having to compare two timestamps itself.

    Best-effort, like the rest of this module: any DB error is treated as "no
    failure to report" rather than raised.
    """
    try:
        with psycopg.connect(
            get_postgres_connection_string(), connect_timeout=3
        ) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT finished_at FROM semantic_compilation_history
                    WHERE status = '{RUN_FAILED}'
                    AND finished_at > COALESCE(
                        (
                            SELECT MAX(finished_at) FROM semantic_compilation_history
                            WHERE status = '{RUN_SUCCEEDED}'
                        ),
                        '-infinity'
                    )
                    ORDER BY finished_at DESC
                    LIMIT 1
                    """
                )
                row = cur.fetchone()
    except Exception:
        logger.exception("Failed to read last failed semantic compilation run")
        return None

    return row[0].isoformat() if row and row[0] is not None else None
