# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read user-configured acronyms and custom prompts from the GSF metadata DB.

The frontend manages these via Prisma (`acronyms` and `prompts` tables in the
same Postgres instance the backend uses); the chat router pulls them on every
request and injects them into the text-to-SQL agent payload, so the client no
longer needs to ship them on the wire.
"""

from __future__ import annotations

import logging

import psycopg
from psycopg.rows import dict_row

from gsf.infra.postgres import get_postgres_connection_string

logger = logging.getLogger(__name__)


def fetch_acronyms() -> list[dict[str, str]]:
    """Return acronyms as ``[{"name": ..., "description": ...}, ...]``.

    Shape matches ``nemo_retriever`` ``AgentState.acronyms`` / the dicts
    consumed by ``rules_to_text`` in the text-to-SQL pipeline.
    """
    try:
        with psycopg.connect(
            get_postgres_connection_string(), connect_timeout=3
        ) as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    "SELECT name, description FROM acronyms ORDER BY created_at"
                )
                return [dict(row) for row in cur.fetchall()]
    except Exception:
        # Settings are best-effort: a DB hiccup here must not break chat.
        logger.exception("Failed to fetch acronyms; continuing with empty list")
        return []


def fetch_custom_prompts() -> str:
    """Return all stored prompts joined into one string (blank if none).

    The agent embeds this verbatim into the system prompt
    (see ``gsf.retrieval.text_to_sql.main._build_state``).
    """
    try:
        with psycopg.connect(
            get_postgres_connection_string(), connect_timeout=3
        ) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT content FROM prompts")
                rows = cur.fetchall()
    except Exception:
        logger.exception("Failed to fetch custom prompts; continuing with empty string")
        return ""

    contents = [row[0] for row in rows if row[0]]
    return "\n\n".join(contents)
