# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Neo4j helpers for applying a zone scope to catalog queries.

Users are not represented in Neo4j.  Zone membership is not used to
authorize catalog access: every authenticated role has the same unrestricted
catalog scope.
"""

from __future__ import annotations

from typing import Any

from gsf.catalog.constants import Edges, Labels
from gsf.catalog.store.neo4j.connection import get_neo4j_conn

from gsf.server.zones.constants import LABEL_ZONE, REL_ZONE_OF


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

    # Collect items directly linked via Zone → ZONE_OF → item.
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


def resolve_accessible_catalog_ids(
    zone_ids: list[str] | None,
    data_ids_by_zone: dict[str, set[str]] | None = None,
) -> dict[str, set[str]] | None:
    """Resolve *zone_ids* to accessible catalog ids, reusing *data_ids_by_zone* if given.

    ``get_accessible_catalog_ids_for_zones`` costs several Neo4j round trips.
    Callers that need the resolved ids in more than one place for the same
    request (e.g. the Exploration graph builders, which combine node,
    zone-map and edge queries) should resolve it once via this helper and
    thread the result through every downstream call as *data_ids_by_zone*,
    instead of letting each one re-resolve the same *zone_ids* independently.

    Returns ``None`` when *zone_ids* is ``None`` (no zone scoping — admin /
    internal callers). Returns *data_ids_by_zone* unchanged when already
    supplied.
    """
    if zone_ids is None:
        return None
    if data_ids_by_zone is not None:
        return data_ids_by_zone
    return get_accessible_catalog_ids_for_zones(zone_ids)


def resolve_table_filter(
    zone_ids: list[str] | None,
    column_ref: str,
    *,
    extra_params: dict[str, Any] | None = None,
    data_ids_by_zone: dict[str, set[str]] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Build a Cypher ``WHERE`` clause restricting *column_ref* to accessible tables.

    *column_ref* is the Cypher expression to filter on, e.g. ``"t.id"`` or
    ``"attr.table_id"``.  When *zone_ids* is ``None`` no filter is applied
    (admin / internal callers who see the full unfiltered catalog).
    *extra_params* are merged into the returned params dict unchanged (e.g.
    query parameters unrelated to zone scoping). Pass a pre-resolved
    *data_ids_by_zone* (see ``resolve_accessible_catalog_ids``) to avoid a
    repeat Neo4j round trip when the caller already has it for this request.

    Returns ``(where_clause, params)`` where *where_clause* is either an
    empty string or a full ``WHERE <column_ref> IN $table_ids`` clause ready
    to interpolate into an f-string query.
    """
    params = dict(extra_params or {})
    if zone_ids is None:
        return "", params
    resolved = resolve_accessible_catalog_ids(zone_ids, data_ids_by_zone)
    params["table_ids"] = list(resolved["table_ids"])
    return f"WHERE {column_ref} IN $table_ids", params
