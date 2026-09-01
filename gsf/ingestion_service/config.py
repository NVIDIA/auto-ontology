# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read ingestion-service feature flags from the GSF metadata DB.

The frontend manages these via Prisma (the ``configurations`` key/value table in
the same Postgres instance the backend uses). We read them here with psycopg so
the ingestion service can gate behaviour on them without a frontend round-trip.

The history of individual compilation passes (start/end, success) is *not*
kept here — it lives in the GSF-owned ``semantic_compilation_history`` table,
see ``history.py``.
"""

from __future__ import annotations

import logging

import psycopg

from gsf.infra.postgres import get_postgres_connection_string

logger = logging.getLogger(__name__)

# Key stored in the ``configurations`` table by the Semantic Compilation
# settings tab. Its value is the string ``"true"`` or ``"false"``.
SEMANTIC_COMPILATION_ENABLED_KEY = "semantic_compilation_enabled"


def is_semantic_compilation_enabled() -> bool:
    """Return whether semantic compilation is enabled in settings.

    Best-effort: any DB error (including the table not existing yet) is treated
    as "disabled" so a missing/unreachable config never crashes startup.
    """
    try:
        with psycopg.connect(
            get_postgres_connection_string(), connect_timeout=3
        ) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT value FROM configurations WHERE key = %s",
                    (SEMANTIC_COMPILATION_ENABLED_KEY,),
                )
                row = cur.fetchone()
    except Exception:
        logger.exception(
            "Failed to read %s; treating semantic compilation as disabled",
            SEMANTIC_COMPILATION_ENABLED_KEY,
        )
        return False

    return bool(row) and str(row[0]).strip().lower() == "true"
