# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Neo4j data access for User nodes.

User nodes mirror the authoritative records kept in PostgreSQL (Better Auth).
They are created here so that other graph entities (e.g. conversations,
analytics) can be linked to a user node without crossing the SQL/graph
boundary at query time.

The ``id`` field is the PostgreSQL user id and acts as the natural key.

Each node carries exactly one role label — either ``Admin`` or ``Viewer`` —
that mirrors the ``role`` property.  When a user's role changes the old
label is removed and the new one is set so that the graph label always
reflects current access level.
"""

from __future__ import annotations

import logging
from typing import Any

from nemo_retriever.tabular_data.ingestion.model.reserved_words import Edges, Labels
from nemo_retriever.tabular_data.neo4j import get_neo4j_conn

from gsf.server.zones.constants import (
    LABEL_ZONE,
    REL_HAS_DIRECT_ACCESS,
    REL_PARTICIPANT_OF,
    REL_ZONE_OF,
)


logger = logging.getLogger(__name__)

# Role labels — each user node carries exactly one of these.
LABEL_ADMIN = "Admin"
LABEL_VIEWER = "Viewer"

# Cypher label-union pattern used in MATCH clauses to find any user node
# regardless of its current role label.
LABEL_USER_MATCH = f"{LABEL_ADMIN}|{LABEL_VIEWER}"


def get_accessible_catalog_ids_for_zones(
    zone_ids: list[str],
) -> dict[str, set[str]]:
    """Return the catalog node IDs reachable via *zone_ids*.

    Receives a pre-resolved list of zone IDs (already scoped to the requesting
    user).  Does NOT consult the User node — caller is responsible for passing
    only the zones the user has access to.

    Returns a dict with three sets:

    * ``"db_ids"``     — Database nodes reachable through the given zones.
    * ``"schema_ids"`` — Schema nodes reachable through the given zones.
    * ``"table_ids"``  — Table nodes reachable through the given zones.

    Parent nodes are expanded automatically: if a zone grants access to a Table,
    the parent Schema and grandparent Database are added so the catalog tree can
    be rendered correctly on the client.  Likewise, if a zone covers a DB, all
    descendant schemas and tables are included.
    """
    conn = get_neo4j_conn()

    # Collect items directly linked via zone → zone_of → item.
    item_rows = conn.query_read(
        f"""
        UNWIND $zone_ids AS zone_id
        MATCH (z:{LABEL_ZONE} {{id: zone_id}})-[:{REL_ZONE_OF}]->(item)
        RETURN labels(item)[0] AS label, item.id AS id
        """,
        {"zone_ids": zone_ids},
    )

    direct_db_ids: set[str] = set()
    direct_schema_ids: set[str] = set()
    direct_table_ids: set[str] = set()
    for r in item_rows:
        lbl, nid = r["label"], r["id"]
        if lbl == Labels.DB:
            direct_db_ids.add(nid)
        elif lbl == Labels.SCHEMA:
            direct_schema_ids.add(nid)
        elif lbl == Labels.TABLE:
            direct_table_ids.add(nid)

    all_db_ids: set[str] = set(direct_db_ids)
    all_schema_ids: set[str] = set(direct_schema_ids)
    all_table_ids: set[str] = set(direct_table_ids)

    # DB-level zone → expand to all descendant schemas and tables.
    if direct_db_ids:
        rows = conn.query_read(
            f"""
            UNWIND $db_ids AS db_id
            MATCH (db:{Labels.DB} {{id: db_id}})-[:{Edges.CONTAINS}]->(s:{Labels.SCHEMA})
            OPTIONAL MATCH (s)-[:{Edges.CONTAINS}]->(t:{Labels.TABLE})
            RETURN DISTINCT s.id AS schema_id, t.id AS table_id
            """,
            {"db_ids": list(direct_db_ids)},
        )
        for r in rows:
            if r["schema_id"]:
                all_schema_ids.add(r["schema_id"])
            if r["table_id"]:
                all_table_ids.add(r["table_id"])

    # Schema-level zone → find parent DB + descendant tables.
    if direct_schema_ids:
        rows = conn.query_read(
            f"""
            UNWIND $schema_ids AS schema_id
            MATCH (db:{Labels.DB})-[:{Edges.CONTAINS}]->(s:{Labels.SCHEMA} {{id: schema_id}})
            OPTIONAL MATCH (s)-[:{Edges.CONTAINS}]->(t:{Labels.TABLE})
            RETURN DISTINCT db.id AS db_id, t.id AS table_id
            """,
            {"schema_ids": list(direct_schema_ids)},
        )
        for r in rows:
            if r["db_id"]:
                all_db_ids.add(r["db_id"])
            if r["table_id"]:
                all_table_ids.add(r["table_id"])

    # Table-level zone → find parent Schema and grandparent DB.
    if direct_table_ids:
        rows = conn.query_read(
            f"""
            UNWIND $table_ids AS table_id
            MATCH (db:{Labels.DB})-[:{Edges.CONTAINS}]->(s:{Labels.SCHEMA})
                  -[:{Edges.CONTAINS}]->(t:{Labels.TABLE} {{id: table_id}})
            RETURN DISTINCT db.id AS db_id, s.id AS schema_id
            """,
            {"table_ids": list(direct_table_ids)},
        )
        for r in rows:
            if r["db_id"]:
                all_db_ids.add(r["db_id"])
            if r["schema_id"]:
                all_schema_ids.add(r["schema_id"])

    return {
        "db_ids": all_db_ids,
        "schema_ids": all_schema_ids,
        "table_ids": all_table_ids,
    }


def resolve_table_filter(
    zone_ids: list[str] | None,
    column_ref: str,
    *,
    extra_params: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Build a Cypher ``WHERE`` clause restricting *column_ref* to accessible tables.

    *column_ref* is the Cypher expression to filter on, e.g. ``"t.id"`` or
    ``"attr.table_id"``.  When *zone_ids* is ``None`` no filter is applied
    (admin / internal callers who see the full unfiltered catalog).
    *extra_params* are merged into the returned params dict unchanged (e.g.
    query parameters unrelated to zone scoping).

    Returns ``(where_clause, params)`` where *where_clause* is either an
    empty string or a full ``WHERE <column_ref> IN $table_ids`` clause ready
    to interpolate into an f-string query.
    """
    params = dict(extra_params or {})
    if zone_ids is None:
        return "", params
    table_ids = list(get_accessible_catalog_ids_for_zones(zone_ids)["table_ids"])
    params["table_ids"] = table_ids
    return f"WHERE {column_ref} IN $table_ids", params


def sync_admin_direct_access() -> None:
    """Ensure every Admin node has a HAS_DIRECT_ACCESS edge to every Database.

    Admin nodes already reach zoned data through the
    ``participant_of -> zone -> zone_of -> item`` path.  This function adds an
    explicit ``HAS_DIRECT_ACCESS`` edge from every Admin node to **all**
    Database nodes so that the full catalog hierarchy (Database -> Schema ->
    Table via CONTAINS) is reachable from admin in Neo4j regardless of zone
    assignments.

    The function is idempotent — safe to call repeatedly without side-effects.
    It should be called when a user is promoted to admin and after bulk catalog
    ingestion to cover newly-added databases.
    """
    get_neo4j_conn().query_write(
        f"""
        MATCH (u:{LABEL_ADMIN})
        MATCH (db:{Labels.DB})
        MERGE (u)-[:{REL_HAS_DIRECT_ACCESS}]->(db)
        """,
    )


def get_user(user_id: str) -> dict[str, Any] | None:
    """Return a User node by its id, or *None* if it does not exist."""
    rows = get_neo4j_conn().query_read(
        f"""
        MATCH (u:{LABEL_USER_MATCH} {{id: $user_id}})
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
        MATCH (u:{LABEL_USER_MATCH})
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

    Each node carries exactly one role label (``Admin`` or ``Viewer``).
    On create the correct label is set immediately.  On update the old
    label is removed and the new one applied, so a role change is handled
    atomically without leaving stale labels.

    Returns the node's stored properties.
    """
    conn = get_neo4j_conn()
    label = LABEL_ADMIN if role == "admin" else LABEL_VIEWER

    existing = conn.query_read(
        f"""
        MATCH (u:{LABEL_USER_MATCH} {{id: $user_id}})
        RETURN u.id AS id
        LIMIT 1
        """,
        {"user_id": user_id},
    )

    if existing:
        # Node exists: update properties and swap role label if it changed.
        rows = conn.query_write(
            f"""
            MATCH (u:{LABEL_USER_MATCH} {{id: $user_id}})
            REMOVE u:{LABEL_ADMIN}, u:{LABEL_VIEWER}
            SET u:{label},
                u.email = $email,
                u.name  = $name,
                u.role  = $role
            RETURN u.id    AS id,
                   u.email AS email,
                   u.name  AS name,
                   u.role  AS role
            """,
            {"user_id": user_id, "email": email, "name": name, "role": role},
        )
    else:
        # Node does not exist yet: create with the correct role label.
        rows = conn.query_write(
            f"""
            CREATE (u:{label} {{id: $user_id, email: $email, name: $name, role: $role}})
            RETURN u.id    AS id,
                   u.email AS email,
                   u.name  AS name,
                   u.role  AS role
            """,
            {"user_id": user_id, "email": email, "name": name, "role": role},
        )

    assert rows
    # Admins are linked to every existing zone so that the graph reflects full
    # access (visible in Neo4j as Admin->zone->data paths).  Viewers start with
    # no zone access — access must be granted explicitly via grant_zone_access.
    # The list_zone_users query filters admins out so they never appear in the
    # UI access list (their access is implicit, not an explicit assignment).
    if role == "admin":
        conn.query_write(
            f"""
            MATCH (u:{LABEL_ADMIN} {{id: $user_id}})
            MATCH (z:{LABEL_ZONE})
            MERGE (u)-[:{REL_PARTICIPANT_OF}]->(z)
            """,
            {"user_id": user_id},
        )
        # Ensure admin can reach catalog items not assigned to any zone.
        sync_admin_direct_access()
    else:
        # When demoted from admin to viewer, remove all zone relationships so
        # the user starts with no zone access (access must be re-granted
        # explicitly by an admin via grant_zone_access).  Also remove any
        # HAS_DIRECT_ACCESS edges that were created when the user was an admin.
        conn.query_write(
            f"""
            MATCH (u:{LABEL_VIEWER} {{id: $user_id}})
            OPTIONAL MATCH (u)-[rp:{REL_PARTICIPANT_OF}]->()
            OPTIONAL MATCH (u)-[rd:{REL_HAS_DIRECT_ACCESS}]->()
            DELETE rp, rd
            """,
            {"user_id": user_id},
        )
    return dict(rows[0])
