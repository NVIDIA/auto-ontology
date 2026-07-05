# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data Access Layer — Neo4j queries for Zone nodes."""

from __future__ import annotations

from typing import Any

from nemo_retriever.tabular_data.neo4j import get_neo4j_conn

from gsf.dal.users import LABEL_ADMIN, LABEL_USER_MATCH
from gsf.server.zones.constants import REL_PARTICIPANT_OF, REL_ZONE_OF
from gsf.server.zones.utils import (
    LABEL_ZONE,
    REL_CONTAINS,
    ZONE_DATA_LABELS,
    _resolve_api_name,
    format_zone,
)

_DATA_ITEM_PATTERN = "|".join(ZONE_DATA_LABELS)


def _resolve_db_ids(conn, item_ids: list[str]) -> list[str]:
    """Return the IDs of root Database nodes that own the given items.

    If the item is already a Database node its ID is used directly.
    For Schema and Table nodes the function walks up the CONTAINS
    hierarchy (1–2 hops) to find the ancestor Database.
    """
    if not item_ids:
        return []
    rows = conn.query_read(
        f"""
        UNWIND $item_ids AS item_id
        MATCH (item:{_DATA_ITEM_PATTERN} {{id: item_id}})
        WITH item,
             CASE WHEN item:{ZONE_DATA_LABELS[0]} THEN item.id ELSE null END AS direct_db_id
        OPTIONAL MATCH (db:{ZONE_DATA_LABELS[0]})-[:{REL_CONTAINS}*1..2]->(item)
        WHERE direct_db_id IS NULL
        WITH coalesce(direct_db_id, db.id) AS resolved_db_id
        WHERE resolved_db_id IS NOT NULL
        RETURN DISTINCT resolved_db_id AS db_id
        """,
        {"item_ids": item_ids},
    )
    return [row["db_id"] for row in rows]


def list_zones(user_id: str) -> list[dict[str, Any]]:
    """Return zones visible to *user_id*.

    Admins see every zone.  Viewers see only zones to which they have been
    explicitly granted access via a ``PARTICIPANT_OF`` relationship.
    Returns an empty list when *user_id* is unknown.
    """
    rows = get_neo4j_conn().query_read(
        f"""
        MATCH (u:{LABEL_USER_MATCH} {{id: $user_id}})
        WITH u, u.role AS role
        MATCH (z:{LABEL_ZONE})
        WHERE role = 'admin'
           OR EXISTS {{ MATCH (u)-[:{REL_PARTICIPANT_OF}]->(z) }}
        RETURN z.id          AS id,
               z.name        AS name,
               z.description AS description,
               z.color       AS color
        ORDER BY z.name
        """,
        {"user_id": user_id},
    )
    return [format_zone(dict(r)) for r in rows]


def get_zone_by_id(
    zone_id: str, *, user_id: str | None = None
) -> dict[str, Any] | None:
    """Return one zone with its linked catalog data items.

    When *user_id* is supplied the zone is only returned if the user has
    access (admin role or explicit ``PARTICIPANT_OF`` relationship).
    Passing ``user_id=None`` skips the access check (internal / admin use).
    """
    conn = get_neo4j_conn()
    if user_id is not None:
        access_rows = conn.query_read(
            f"""
            MATCH (u:{LABEL_USER_MATCH} {{id: $user_id}})
            MATCH (z:{LABEL_ZONE} {{id: $zone_id}})
            WHERE u.role = 'admin'
               OR EXISTS {{ MATCH (u)-[:{REL_PARTICIPANT_OF}]->(z) }}
            RETURN z.id AS id
            LIMIT 1
            """,
            {"user_id": user_id, "zone_id": zone_id},
        )
        if not access_rows:
            return None

    rows = conn.query_read(
        f"""
        MATCH (z:{LABEL_ZONE} {{id: $zone_id}})
        OPTIONAL MATCH (z)-[:{REL_ZONE_OF}]->(item:{_DATA_ITEM_PATTERN})
        WHERE coalesce(item.deleted, false) = false
        WITH z,
             [i IN collect(DISTINCT item)
              WHERE i IS NOT NULL
              | {{
                  id: i.id,
                  name: i.name,
                  gsf_label: head(labels(i))
                }}] AS raw_items
        RETURN z.id AS id,
               z.name AS name,
               z.description AS description,
               z.color AS color,
               raw_items AS items
        """,
        {"zone_id": zone_id},
    )
    if not rows:
        return None

    row = dict(rows[0])
    items = [
        {
            "id": item["id"],
            "name": item["name"],
            "label": _resolve_api_name(item["gsf_label"]),
        }
        for item in row.pop("items", [])
    ]
    return format_zone(row, items=items)


def _zone_name_exists(
    conn,
    name: str,
    *,
    exclude_id: str | None = None,
    item_ids: list[str] | None = None,
) -> bool:
    """Return whether a zone with the same name already exists (case-insensitive).

    Pass ``exclude_id`` to ignore a specific zone (useful during updates).
    Pass ``item_ids`` to scope the check to zones that share the same databases
    as those items; DB IDs are resolved internally via ``_resolve_db_ids``.
    """
    exclude_clause = "AND z.id <> $exclude_id" if exclude_id is not None else ""
    db_ids = _resolve_db_ids(conn, item_ids) if item_ids else None
    db_clause = (
        f"""
          AND EXISTS {{
            MATCH (z)-[:{REL_ZONE_OF}]->(item)
            MATCH (db:{ZONE_DATA_LABELS[0]})-[:{REL_CONTAINS}*0..2]->(item)
            WHERE db.id IN $db_ids
          }}"""
        if db_ids
        else ""
    )
    params: dict[str, Any] = {"name": name}
    if exclude_id is not None:
        params["exclude_id"] = exclude_id
    if db_ids:
        params["db_ids"] = db_ids
    rows = conn.query_read(
        f"""
        MATCH (z:{LABEL_ZONE})
        WHERE toLower(trim(z.name)) = toLower(trim($name))
          {exclude_clause}{db_clause}
        RETURN count(z) > 0 AS exists
        """,
        params,
    )
    return bool(rows and rows[0].get("exists"))


def create_zone(
    *,
    name: str,
    description: str | None,
    color: str,
    item_ids: list[str],
) -> dict[str, Any]:
    """Create a zone and link it to catalog data nodes (db/schema/table).

    Raises ``ValueError`` if a zone with the same name already exists in the
    same databases as the given items.
    """
    conn = get_neo4j_conn()
    if _zone_name_exists(conn, name, item_ids=item_ids):
        raise ValueError(f"Zone with name {name!r} already exists")
    zone_rows = conn.query_write(
        f"""
        CREATE (z:{LABEL_ZONE})
        SET z.id = randomUUID(),
            z.name = $name,
            z.description = $description,
            z.color = $color
        RETURN z.id AS id,
               z.name AS name,
               z.description AS description,
               z.color AS color
        """,
        {
            "name": name,
            "description": description,
            "color": color,
        },
    )
    zone = format_zone(dict(zone_rows[0]), items=[])

    if item_ids:
        linked_rows = conn.query_write(
            f"""
            UNWIND $item_ids AS item_id
            MATCH (z:{LABEL_ZONE} {{id: $zone_id}})
            MATCH (item:{_DATA_ITEM_PATTERN} {{id: item_id}})
            WHERE coalesce(item.deleted, false) = false
            MERGE (z)-[:{REL_ZONE_OF}]->(item)
            RETURN DISTINCT item.id AS id
            """,
            {"zone_id": zone["id"], "item_ids": item_ids},
        )
        linked_ids = {row["id"] for row in linked_rows}
        ordered_linked_ids: list[str] = []
        seen: set[str] = set()
        for item_id in item_ids:
            if item_id not in linked_ids or item_id in seen:
                continue
            seen.add(item_id)
            ordered_linked_ids.append(item_id)
        zone["items"] = ordered_linked_ids

    # Link all admin users to the new zone so the graph reflects full admin
    # access (User->zone->data paths visible in Neo4j).  Viewers are granted
    # access explicitly via grant_zone_access.  Admins are filtered out of
    # list_zone_users so they never appear in the UI access list.
    conn.query_write(
        f"""
        MATCH (z:{LABEL_ZONE} {{id: $zone_id}})
        MATCH (u:{LABEL_ADMIN})
        MERGE (u)-[:{REL_PARTICIPANT_OF}]->(z)
        """,
        {"zone_id": zone["id"]},
    )

    return zone


def update_zone(
    *,
    zone_id: str,
    updates: dict[str, Any],
    item_ids: list[str] | None,
) -> dict[str, Any] | None:
    """Update zone properties and optional full items list, then return zone detail.

    When *item_ids* is provided, the DB IDs are derived from the items themselves.
    Only existing links that belong to those databases are removed, leaving items
    from other databases untouched.
    Name uniqueness is scoped to the databases associated with the zone's items.
    """
    conn = get_neo4j_conn()
    existing_rows = conn.query_read(
        f"""
        MATCH (z:{LABEL_ZONE} {{id: $zone_id}})
        RETURN z.id AS id
        LIMIT 1
        """,
        {"zone_id": zone_id},
    )
    if not existing_rows:
        return None

    if "name" in updates:
        if item_ids is not None:
            name_check_item_ids = item_ids
        else:
            current_items = conn.query_read(
                f"""
                MATCH (z:{LABEL_ZONE} {{id: $zone_id}})-[:{REL_ZONE_OF}]->(item)
                RETURN item.id AS id
                """,
                {"zone_id": zone_id},
            )
            name_check_item_ids = [r["id"] for r in current_items]
        if _zone_name_exists(
            conn, updates["name"], exclude_id=zone_id, item_ids=name_check_item_ids
        ):
            raise ValueError(f"Zone with name {updates['name']!r} already exists")

    if updates:
        set_clauses = ", ".join(f"z.{field} = ${field}" for field in updates)
        conn.query_write(
            f"""
            MATCH (z:{LABEL_ZONE} {{id: $zone_id}})
            SET {set_clauses}
            """,
            {"zone_id": zone_id, **updates},
        )

    if item_ids is not None:
        if item_ids:
            db_ids = _resolve_db_ids(conn, item_ids)
            if db_ids:
                conn.query_write(
                    f"""
                    MATCH (z:{LABEL_ZONE} {{id: $zone_id}})-[r:{REL_ZONE_OF}]->(item)
                    WHERE EXISTS {{
                        MATCH (db:{ZONE_DATA_LABELS[0]})-[:{REL_CONTAINS}*0..2]->(item)
                        WHERE db.id IN $db_ids
                    }}
                    DELETE r
                    """,
                    {"zone_id": zone_id, "db_ids": db_ids},
                )
            conn.query_write(
                f"""
                UNWIND $item_ids AS item_id
                MATCH (z:{LABEL_ZONE} {{id: $zone_id}})
                MATCH (item:{_DATA_ITEM_PATTERN} {{id: item_id}})
                WHERE coalesce(item.deleted, false) = false
                MERGE (z)-[:{REL_ZONE_OF}]->(item)
                """,
                {"zone_id": zone_id, "item_ids": item_ids},
            )
        else:
            conn.query_write(
                f"""
                MATCH (z:{LABEL_ZONE} {{id: $zone_id}})-[r:{REL_ZONE_OF}]->()
                DELETE r
                """,
                {"zone_id": zone_id},
            )
    return get_zone_by_id(zone_id)


def delete_zone(zone_id: str) -> bool:
    """Delete a zone by id. Return whether a node was deleted."""
    conn = get_neo4j_conn()
    existing_rows = conn.query_read(
        f"""
        MATCH (z:{LABEL_ZONE} {{id: $zone_id}})
        RETURN z.id AS id
        LIMIT 1
        """,
        {"zone_id": zone_id},
    )
    if not existing_rows:
        return False

    conn.query_write(
        f"""
        MATCH (z:{LABEL_ZONE} {{id: $zone_id}})
        DETACH DELETE z
        """,
        {"zone_id": zone_id},
    )
    return True


def list_zone_users(zone_id: str) -> list[dict[str, Any]]:
    """Return viewer users that have explicit access to *zone_id*.

    Admins access all zones via role check and are excluded from the result —
    only explicitly-granted viewer relationships are returned.
    """
    rows = get_neo4j_conn().query_read(
        f"""
        MATCH (u:{LABEL_USER_MATCH})-[:{REL_PARTICIPANT_OF}]->(z:{LABEL_ZONE} {{id: $zone_id}})
        WHERE u.role <> 'admin'
        RETURN u.id    AS id,
               u.email AS email,
               u.name  AS name,
               u.role  AS role
        ORDER BY u.email
        """,
        {"zone_id": zone_id},
    )
    return [dict(r) for r in rows]


def grant_zone_access(zone_id: str, user_id: str) -> bool:
    """Create a ``PARTICIPANT_OF`` edge from *user_id* to *zone_id*.

    Returns ``False`` when either the zone or the user does not exist.
    """
    conn = get_neo4j_conn()
    rows = conn.query_write(
        f"""
        MATCH (u:{LABEL_USER_MATCH} {{id: $user_id}})
        MATCH (z:{LABEL_ZONE} {{id: $zone_id}})
        MERGE (u)-[:{REL_PARTICIPANT_OF}]->(z)
        RETURN u.id AS user_id, z.id AS zone_id
        """,
        {"user_id": user_id, "zone_id": zone_id},
    )
    return bool(rows)


def revoke_zone_access(zone_id: str, user_id: str) -> bool:
    """Delete the ``PARTICIPANT_OF`` edge between *user_id* and *zone_id*.

    Returns ``False`` when the edge (or either node) does not exist.
    Admins always keep their implicit access — this only removes the stored
    relationship, which is harmless for admins since their role check always
    passes in ``list_zones`` / ``get_zone_by_id``.
    """
    conn = get_neo4j_conn()
    rows = conn.query_write(
        f"""
        MATCH (u:{LABEL_USER_MATCH} {{id: $user_id}})-[r:{REL_PARTICIPANT_OF}]->(z:{LABEL_ZONE} {{id: $zone_id}})
        DELETE r
        RETURN u.id AS user_id
        """,
        {"user_id": user_id, "zone_id": zone_id},
    )
    return bool(rows)
