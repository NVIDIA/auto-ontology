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

from auto_ontology.dal.schema import GroupRole
from auto_ontology.infra.postgres import FRONTEND_SCHEMA, get_postgres_connection_string

logger = logging.getLogger(__name__)


def get_user_role(user_id: str, *, strict: bool = False) -> str | None:
    """Return the Postgres ``role`` column for *user_id*.

    ``None`` when the account does not exist. A database failure also
    returns ``None``, so a permission check fails closed instead of
    granting access. ``strict`` re-raises that failure: ownership cleanup
    must not treat an outage as a missing account.
    """
    try:
        with psycopg.connect(
            get_postgres_connection_string(), connect_timeout=3
        ) as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                # Schema-qualified. The role's search_path is `public`, and
                # the Better Auth account table lives in `frontend`. An
                # unqualified `"user"` raises, this function swallows that
                # and returns None, and every admin then fails the check.
                cur.execute(
                    f'SELECT role FROM {FRONTEND_SCHEMA}."user" WHERE id = %s',
                    (user_id,),
                )
                row = cur.fetchone()
                return row["role"] if row else None
    except Exception:
        logger.exception("Failed to fetch role for user %r from Postgres", user_id)
        if strict:
            raise
        return None


def is_admin(user_id: str) -> bool:
    """Return whether *user_id* currently has the ``admin`` role in Postgres."""
    return get_user_role(user_id) == GroupRole.ADMIN
