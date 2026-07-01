# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Neo4j data access for User nodes.

User nodes mirror the authoritative records kept in PostgreSQL (Better Auth).
They are created here so that other graph entities (e.g. conversations,
analytics) can be linked to a user node without crossing the SQL/graph
boundary at query time.

The ``id`` field is the PostgreSQL user id and acts as the natural key.
"""

from __future__ import annotations

import logging
from typing import Any

from nemo_retriever.tabular_data.neo4j import get_neo4j_conn

from gsf.semantic.constants import LABEL_ZONE, REL_PARTICIPANT_OF

logger = logging.getLogger(__name__)

LABEL_USER = "User"


def get_user(user_id: str) -> dict[str, Any] | None:
    """Return a User node by its id, or *None* if it does not exist."""
    rows = get_neo4j_conn().query_read(
        f"""
        MATCH (u:{LABEL_USER} {{id: $user_id}})
        RETURN u.id    AS id,
               u.email AS email,
               u.name  AS name,
               u.role  AS role
        LIMIT 1
        """,
        {"user_id": user_id},
    )
    return dict(rows[0]) if rows else None


def list_users() -> list[dict[str, Any]]:
    """Return all User nodes ordered by email."""
    rows = get_neo4j_conn().query_read(
        f"""
        MATCH (u:{LABEL_USER})
        RETURN u.id    AS id,
               u.email AS email,
               u.name  AS name,
               u.role  AS role
        ORDER BY u.email
        """
    )
    return [dict(r) for r in rows]


def upsert_user(
    *,
    user_id: str,
    email: str,
    name: str,
    role: str,
) -> dict[str, Any]:
    """Create or update a User node identified by *user_id*.

    Uses MERGE so that calling this with the same ``user_id`` is idempotent
    (safe to call on every login / session refresh).  After upserting the
    node, the user is linked to every existing Zone via
    ``PARTICIPANT_OF`` (idempotent thanks to MERGE).  Returns the node's
    stored properties.
    """
    conn = get_neo4j_conn()
    rows = conn.query_write(
        f"""
        MERGE (u:{LABEL_USER} {{id: $user_id}})
        SET u.email = $email,
            u.name  = $name,
            u.role  = $role
        RETURN u.id    AS id,
               u.email AS email,
               u.name  AS name,
               u.role  AS role
        """,
        {
            "user_id": user_id,
            "email": email,
            "name": name,
            "role": role,
        },
    )
    assert rows
    # Admins automatically gain access to every existing zone.
    # Viewers start with no zone access — access must be explicitly granted.
    if role == "admin":
        conn.query_write(
            f"""
            MATCH (u:{LABEL_USER} {{id: $user_id}})
            MATCH (z:{LABEL_ZONE})
            MERGE (u)-[:{REL_PARTICIPANT_OF}]->(z)
            """,
            {"user_id": user_id},
        )
    return dict(rows[0])
