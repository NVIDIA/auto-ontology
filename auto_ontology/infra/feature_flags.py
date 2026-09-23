# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read instance-wide feature flags from the frontend's ``configurations`` table.

The frontend owns that key/value table through Prisma; the Python services read
it directly with psycopg so they can gate behaviour without a frontend
round-trip. Values are the strings ``"true"`` / ``"false"``.

Each flag carries its own default for the "row absent or unreadable" case, and
the defaults genuinely differ: semantic compilation is opt-in (absent means
off), while column profiling is opt-out (absent means on, so existing instances
keep profiling after an upgrade that adds the toggle).
"""

from __future__ import annotations

import logging

import psycopg

from auto_ontology.infra.postgres import FRONTEND_SCHEMA, get_postgres_connection_string

logger = logging.getLogger(__name__)

# Written by the Semantic Compilation settings tab. Opt-out: an instance that
# has never touched the toggle still probes, which is what every deployment
# predating the toggle did.
DISTINCT_VALUE_PROBING_ENABLED_KEY = "distinct_value_probing_enabled"


def read_configuration_flag(key: str, *, default: bool) -> bool:
    """Return the boolean value of *key*, or *default* when it cannot be read.

    Best-effort by design: any DB error (including the table not existing yet,
    which is the case before the frontend's first migration) falls back to
    *default* rather than raising, so a missing or unreachable config never
    crashes a service at startup.
    """
    try:
        with psycopg.connect(
            get_postgres_connection_string(), connect_timeout=3
        ) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT value FROM {FRONTEND_SCHEMA}.configurations WHERE key = %s",
                    (key,),
                )
                row = cur.fetchone()
    except Exception:
        logger.exception("Failed to read %s; falling back to %s", key, default)
        return default

    if not row:
        return default
    # Only the two values the frontend writes count as a decision. Anything
    # else -- an empty string, "1", "yes", a hand-edited or seeded row -- falls
    # back to *default*, because the alternative is a silent disagreement: the
    # frontend reads an opt-out flag as `value !== 'false'`, so a junk value
    # renders the toggle ON while this returned False and stopped the work.
    value = str(row[0]).strip().lower()
    if value == "true":
        return True
    if value == "false":
        return False
    logger.warning(
        "Configuration %s holds unrecognised value %r; using default %s",
        key,
        row[0],
        default,
    )
    return default


def is_distinct_value_probing_enabled() -> bool:
    """Whether compilation may run per-column ``SELECT DISTINCT`` probes.

    Scoped to those probes only. The bounded ``SELECT * ... LIMIT 1000`` row
    sample always runs -- it is one cheap query per table. The DISTINCT probes
    are the unbounded ones: one per low-cardinality text column, and a full
    column scan on warehouses where ``DISTINCT`` + ``LIMIT`` does not
    early-stop, which is what makes them worth switching off on a large or
    metered warehouse.
    """
    return read_configuration_flag(DISTINCT_VALUE_PROBING_ENABLED_KEY, default=True)
