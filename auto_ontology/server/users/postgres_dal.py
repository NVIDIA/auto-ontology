# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PostgreSQL-backed lookup of a user's role (Better Auth ``user`` table).

PostgreSQL is the single source of truth for whether a user is an ``admin``
or a ``viewer`` — the catalog only stores a single, role-agnostic
``User`` node (see ``auto_ontology.dal.users``).  Anything that needs to gate access
(e.g. "can this user manage zones?", "does this user see every zone?")
must go through :func:`get_user_role` / :func:`is_admin` instead of reading
a label or property off the graph node.
"""

from __future__ import annotations

import logging

import psycopg
from psycopg.rows import dict_row

from auto_ontology.infra.postgres import get_postgres_connection_string

logger = logging.getLogger(__name__)


def get_user_role(user_id: str) -> str | None:
    """Return the Postgres ``role`` column for *user_id*.

    Returns ``None`` when the user does not exist or the database is
    unreachable — callers should treat that as "not an admin" rather than
    raising, so a transient Postgres hiccup fails closed instead of
    granting elevated access.
    """
    try:
        with psycopg.connect(
            get_postgres_connection_string(), connect_timeout=3
        ) as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute('SELECT role FROM "user" WHERE id = %s', (user_id,))
                row = cur.fetchone()
                return row["role"] if row else None
    except Exception:
        logger.exception("Failed to fetch role for user %r from Postgres", user_id)
        return None


def is_admin(user_id: str) -> bool:
    """Return whether *user_id* currently has the ``admin`` role in Postgres."""
    return get_user_role(user_id) == "admin"
