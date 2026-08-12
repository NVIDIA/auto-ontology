# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Neo4j data access for the data- and semantic-layer Exploration graphs.

The Postgres counterpart is :mod:`gsf.dal.pg.exploration`; `gsf.dal.exploration`
selects between them. Phase 11 deletes this file.

Contains only functions that call ``get_neo4j_conn()`` directly (aside from
the internal calls each graph builder makes to its own sibling functions
below). All read functions use the ``fetch_*`` prefix.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from gsf.catalog.constants import Edges, Labels
from gsf.catalog.store.neo4j.connection import get_neo4j_conn

from gsf.dal.cypher_fragments import paging_clause, table_description_expr

# Cypher, so imported from the Neo4j implementation directly rather than
# through the selector: the selector carries only the backend-neutral function
# surface. Phase 9 ports this module and the import goes with it.
from gsf.dal.neo4j.datasources import TABLE_COUNTS_SUBQUERY
from gsf.dal.neo4j.sql_attributes import fetch_sql_attribute_counts
from gsf.dal.neo4j.terms import (
    build_term_table_maps,
    fetch_all_terms,
    fetch_column_attribute_counts,
    fetch_related_term_ids,
    fetch_related_terms_counts,
    fetch_term_table_pairs,
    fetch_terms_by_ids,
    term_is_in_scope,
)
from gsf.dal.neo4j.users import resolve_accessible_catalog_ids, resolve_table_filter
from gsf.semantic.constants import (
    LABEL_COLUMN_ATTRIBUTE,
    LABEL_TERM,
    REL_HAS_ATTRIBUTE,
    REL_PROPERTY_OF,
    REL_REPRESENTS,
    REL_SEMANTIC_FK,
)
from gsf.server.zones.constants import (
    LABEL_ZONE_DISABLED,
    REL_ZONE_OF,
    ZONE_LABEL_PATTERN,
)

# Hard ceiling on nodes returned by an Exploration graph endpoint, regardless
# of what a caller requests — keeps the response bounded on large catalogs.
MAX_EXPLORATION_GRAPH_NODES = 500

# A Sql node with no query text is not a data-layer edge. Every read that
# derives table-to-table relationships has to apply this, or the same pair of
# tables is connected in one view and unconnected in another. Expects `sql`.
_NON_EMPTY_SQL = (
    "sql.sql_full_query IS NOT NULL AND trim(toString(sql.sql_full_query)) <> ''"
)


def fetch_data_exploration_edges(
    zone_ids: list[str] | None = None,
    data_ids_by_zone: dict[str, set[str]] | None = None,
) -> list[dict[str, Any]]:
    """Return table pairs connected by a shared SQL query or a foreign key.

    A query becomes an exploration edge when its ``Sql`` node references at
    least two visible tables — each such edge includes the SQL text shown
    when the user selects that connection. A foreign key between two
    tables' columns also becomes an edge (flagged ``via_foreign_key``), even
    when no stored SQL query ever referenced both tables together; that
    edge carries an empty ``queries`` list unless a shared SQL query also
    connects the same pair, in which case the two are merged into one edge.
    Every edge also carries a ``foreign_keys`` list — one entry per FK
    column pair between the two tables (``source_column``,
    ``target_column``, and each column's ``sample_values``) — so the
    client can show which columns join the pair even when there's no
    stored SQL query to display. Empty for edges that are SQL-only.
    Pass a pre-resolved *data_ids_by_zone* (see
    ``resolve_accessible_catalog_ids``) when the caller already resolved
    *zone_ids* for this request, to skip a repeat Neo4j round trip.
    """
    data_ids_by_zone = resolve_accessible_catalog_ids(zone_ids, data_ids_by_zone)
    if data_ids_by_zone is not None:
        table_ids = list(data_ids_by_zone["table_ids"])
        if not table_ids:
            return []
        table_filter = "source.id IN $table_ids AND target.id IN $table_ids AND "
        params: dict[str, Any] = {"table_ids": table_ids}
    else:
        table_filter = ""
        params = {}

    conn = get_neo4j_conn()
    sql_rows = conn.query_read(
        f"""
        MATCH (source:{Labels.TABLE})<-[:{Edges.SQL}]-(sql:{Labels.SQL})
              -[:{Edges.SQL}]->(target:{Labels.TABLE})
        WHERE {table_filter}source.id < target.id AND {_NON_EMPTY_SQL}
        RETURN source.id AS source,
               target.id AS target,
               collect(DISTINCT sql.sql_full_query) AS queries
        """,
        params,
    )
    fk_rows = conn.query_read(
        f"""
        MATCH (source:{Labels.TABLE})-[:{Edges.CONTAINS}]->(src_col:{Labels.COLUMN})
              -[:{Edges.FOREIGN_KEY}]->(tgt_col:{Labels.COLUMN})<-[:{Edges.CONTAINS}]-
              (target:{Labels.TABLE})
        WHERE {table_filter}source.id <> target.id
        RETURN DISTINCT source.id AS source, target.id AS target,
               src_col.name AS source_column, tgt_col.name AS target_column,
               src_col.sample_values AS source_sample_values,
               tgt_col.sample_values AS target_sample_values
        """,
        params,
    )

    edges: dict[tuple[str, str], dict[str, Any]] = {}
    for row in sql_rows:
        source, target = row.get("source"), row.get("target")
        if not source or not target:
            continue
        edges[(source, target)] = {
            "source": source,
            "target": target,
            "queries": list(row.get("queries") or []),
            "via_foreign_key": False,
            "foreign_keys": [],
        }
    for row in fk_rows:
        a, b = row.get("source"), row.get("target")
        if not a or not b:
            continue
        # Edge keys are normalized to (min id, max id) above; when a FK row's
        # (source, target) came in reversed relative to that order, the
        # column pair it carries must be swapped along with it so
        # `foreign_keys[].source_column` always names a column on
        # `edge["source"]`, never on `edge["target"]`.
        flipped = a > b
        key = (b, a) if flipped else (a, b)
        fk_detail = {
            "source_column": row.get("target_column" if flipped else "source_column"),
            "target_column": row.get("source_column" if flipped else "target_column"),
            "source_sample_values": row.get(
                "target_sample_values" if flipped else "source_sample_values"
            ),
            "target_sample_values": row.get(
                "source_sample_values" if flipped else "target_sample_values"
            ),
        }
        edge = edges.get(key)
        if edge is None:
            edge = {
                "source": key[0],
                "target": key[1],
                "queries": [],
                "via_foreign_key": True,
                "foreign_keys": [],
            }
            edges[key] = edge
        else:
            edge["via_foreign_key"] = True
        if fk_detail not in edge["foreign_keys"]:
            edge["foreign_keys"].append(fk_detail)

    return sorted(edges.values(), key=lambda edge: (edge["source"], edge["target"]))


def fetch_table_exploration_details(
    table_id: str,
    zone_ids: list[str] | None = None,
    *,
    skip: int = 0,
    limit: int | None = None,
) -> dict[str, Any]:
    """Return SQL queries and one ordered page of Terms linked to a visible Table."""
    data_ids_by_zone = resolve_accessible_catalog_ids(zone_ids)
    if data_ids_by_zone is not None and table_id not in data_ids_by_zone["table_ids"]:
        return {"queries": [], "terms": [], "terms_total": 0}

    conn = get_neo4j_conn()
    queries = conn.query_read(
        f"""
        MATCH (sql:{Labels.SQL})-[:{Edges.SQL}]->
              (t:{Labels.TABLE} {{id: $table_id}})
        RETURN DISTINCT sql.id AS id, sql.sql_full_query AS sql
        ORDER BY id
        """,
        {"table_id": table_id},
    )
    term_params: dict[str, Any] = {"table_id": table_id}
    paging = paging_clause(skip, limit, term_params)
    terms = conn.query_read(
        f"""
        CALL () {{
            MATCH (t:{Labels.TABLE} {{id: $table_id}})
                  -[:{REL_REPRESENTS}]->(term:{LABEL_TERM})
            RETURN term.id AS id, term.name AS name, term.description AS description
            UNION
            MATCH (t:{Labels.TABLE} {{id: $table_id}})
                  -[:{Edges.CONTAINS}]->(:{Labels.COLUMN})
                  -[:{REL_HAS_ATTRIBUTE}|{REL_SEMANTIC_FK}]->
                  (:{LABEL_COLUMN_ATTRIBUTE})-[:{REL_PROPERTY_OF}]->
                  (term:{LABEL_TERM})
            RETURN term.id AS id, term.name AS name, term.description AS description
        }}
        WITH DISTINCT id, name, description
        ORDER BY name, id
        {paging}
        RETURN id, name, description
        """,
        term_params,
    )
    total_rows = conn.query_read(
        f"""
        CALL () {{
            MATCH (t:{Labels.TABLE} {{id: $table_id}})
                  -[:{REL_REPRESENTS}]->(term:{LABEL_TERM})
            RETURN term.id AS id
            UNION
            MATCH (t:{Labels.TABLE} {{id: $table_id}})
                  -[:{Edges.CONTAINS}]->(:{Labels.COLUMN})
                  -[:{REL_HAS_ATTRIBUTE}|{REL_SEMANTIC_FK}]->
                  (:{LABEL_COLUMN_ATTRIBUTE})-[:{REL_PROPERTY_OF}]->
                  (term:{LABEL_TERM})
            RETURN term.id AS id
        }}
        RETURN count(DISTINCT id) AS total
        """,
        {"table_id": table_id},
    )
    return {
        "queries": [
            {"id": row.get("id") or "", "sql": row.get("sql") or ""}
            for row in queries
            if row.get("sql")
        ],
        "terms": [dict(row) for row in terms if row.get("id")],
        "terms_total": total_rows[0]["total"] if total_rows else 0,
    }


def fetch_exploration_related_nodes(
    node_id: str,
    layer: str,
    zone_ids: list[str] | None = None,
    *,
    skip: int = 0,
    limit: int | None = None,
) -> dict[str, Any]:
    """Return one page of nodes related to an Exploration node.

    This deliberately resolves relationships from the uncapped graph
    definitions rather than from either Exploration graph endpoint. Data nodes
    share SQL or a foreign key; semantic nodes share a table through any of
    the three Term-to-table paths used by ``fetch_related_terms``.
    """
    if layer == "data":
        return _fetch_data_exploration_related_nodes(
            node_id, zone_ids=zone_ids, skip=skip, limit=limit
        )
    if layer == "semantic":
        return _fetch_semantic_exploration_related_nodes(
            node_id, zone_ids=zone_ids, skip=skip, limit=limit
        )
    raise ValueError(f"Unsupported Exploration layer: {layer}")


def _data_related_tables_match(other_filter: str) -> str:
    """Cypher matching every Table related to ``$node_id``, binding ``t``/``db``/``s``.

    Related means what ``fetch_data_exploration_edges`` means by it: a shared
    SQL query with text, or a foreign key in either direction — so a node's
    degree on the graph and the size of its related list are the same number.
    The catalog path is OPTIONAL for that reason: requiring it would drop a
    related table that has no Schema/Database above it from the page while
    the graph still counted it.
    """
    return f"""
        CALL () {{
            MATCH (:{Labels.TABLE} {{id: $node_id}})<-[:{Edges.SQL}]-(sql:{Labels.SQL})
                  -[:{Edges.SQL}]->(other:{Labels.TABLE})
            WHERE other.id <> $node_id AND {_NON_EMPTY_SQL} {other_filter}
            RETURN DISTINCT other.id AS related_id
            UNION
            MATCH (:{Labels.TABLE} {{id: $node_id}})-[:{Edges.CONTAINS}]->
                  (:{Labels.COLUMN})-[:{Edges.FOREIGN_KEY}]->(:{Labels.COLUMN})
                  <-[:{Edges.CONTAINS}]-(other:{Labels.TABLE})
            WHERE other.id <> $node_id {other_filter}
            RETURN DISTINCT other.id AS related_id
            UNION
            MATCH (other:{Labels.TABLE})-[:{Edges.CONTAINS}]->(:{Labels.COLUMN})
                  -[:{Edges.FOREIGN_KEY}]->(:{Labels.COLUMN})
                  <-[:{Edges.CONTAINS}]-(:{Labels.TABLE} {{id: $node_id}})
            WHERE other.id <> $node_id {other_filter}
            RETURN DISTINCT other.id AS related_id
        }}
        WITH DISTINCT related_id
        MATCH (t:{Labels.TABLE} {{id: related_id}})
        OPTIONAL MATCH (db:{Labels.DB})-[:{Edges.CONTAINS}]->
              (s:{Labels.SCHEMA})-[:{Edges.CONTAINS}]->(t)
    """


def _fetch_data_exploration_related_nodes(
    node_id: str,
    zone_ids: list[str] | None,
    *,
    skip: int,
    limit: int | None,
) -> dict[str, Any]:
    """Page Tables related to *node_id* through a shared SQL query or a foreign key.

    Every query here is scoped to ``$node_id`` (or, for
    ``_fetch_table_relationship_counts``, to the tables on the page just
    fetched) rather than building the entire data-layer edge set through
    ``fetch_data_exploration_edges`` and filtering it down to one node in
    Python.
    """
    data_ids_by_zone = resolve_accessible_catalog_ids(zone_ids)
    if data_ids_by_zone is not None and node_id not in data_ids_by_zone["table_ids"]:
        return {"nodes": [], "total": 0}

    other_filter = ""
    zone_table_ids: list[str] | None = None
    params: dict[str, Any] = {"node_id": node_id}
    if data_ids_by_zone is not None:
        zone_table_ids = list(data_ids_by_zone["table_ids"])
        other_filter = "AND other.id IN $zone_table_ids"
        params["zone_table_ids"] = zone_table_ids

    conn = get_neo4j_conn()
    related_match = _data_related_tables_match(other_filter)
    total_rows = conn.query_read(
        f"""
        {related_match}
        RETURN count(DISTINCT related_id) AS total
        """,
        params,
    )
    total = total_rows[0]["total"] if total_rows else 0
    if not total:
        return {"nodes": [], "total": 0}

    paging = paging_clause(skip, limit, params)
    rows = conn.query_read(
        f"""
        {related_match}
        WITH related_id, t, db, s
        ORDER BY db.id, s.id
        WITH related_id     AS id,
             t.name         AS name,
             t.table_type   AS table_type,
             head(collect(db.id)) AS database_id,
             head(collect(s.id))  AS schema_id
        RETURN id, name, table_type, database_id, schema_id
        ORDER BY name, id
        {paging}
        """,
        params,
    )
    relationship_counts = _fetch_table_relationship_counts(
        [row["id"] for row in rows], zone_table_ids=zone_table_ids
    )
    return {
        "nodes": [
            {
                **dict(row),
                "relationship_count": relationship_counts.get(row["id"], 0),
            }
            for row in rows
        ],
        "total": total,
    }


def _fetch_table_relationship_counts(
    table_ids: list[str],
    zone_table_ids: list[str] | None,
) -> dict[str, int]:
    """Return ``{table_id: relationship_count}`` for a small, known set of tables.

    Companion to ``_fetch_data_exploration_related_nodes``: rendering its
    "Relationships" column needs each *returned* table's own total degree,
    under the same definition of an edge, which this re-derives with three
    ``UNWIND``-scoped scans bounded by the page just fetched.
    """
    if not table_ids:
        return {}
    conn = get_neo4j_conn()
    params: dict[str, Any] = {"table_ids": table_ids}
    other_filter = ""
    if zone_table_ids is not None:
        other_filter = "AND other.id IN $zone_table_ids"
        params["zone_table_ids"] = zone_table_ids

    sql_rows = conn.query_read(
        f"""
        UNWIND $table_ids AS tid
        MATCH (:{Labels.TABLE} {{id: tid}})<-[:{Edges.SQL}]-(sql:{Labels.SQL})
              -[:{Edges.SQL}]->(other:{Labels.TABLE})
        WHERE other.id <> tid AND {_NON_EMPTY_SQL} {other_filter}
        RETURN DISTINCT tid AS id, other.id AS other_id
        """,
        params,
    )
    fk_out_rows = conn.query_read(
        f"""
        UNWIND $table_ids AS tid
        MATCH (:{Labels.TABLE} {{id: tid}})-[:{Edges.CONTAINS}]->(:{Labels.COLUMN})
              -[:{Edges.FOREIGN_KEY}]->(:{Labels.COLUMN})<-[:{Edges.CONTAINS}]-
              (other:{Labels.TABLE})
        WHERE other.id <> tid {other_filter}
        RETURN DISTINCT tid AS id, other.id AS other_id
        """,
        params,
    )
    fk_in_rows = conn.query_read(
        f"""
        UNWIND $table_ids AS tid
        MATCH (other:{Labels.TABLE})-[:{Edges.CONTAINS}]->(:{Labels.COLUMN})
              -[:{Edges.FOREIGN_KEY}]->(:{Labels.COLUMN})<-[:{Edges.CONTAINS}]-
              (:{Labels.TABLE} {{id: tid}})
        WHERE other.id <> tid {other_filter}
        RETURN DISTINCT tid AS id, other.id AS other_id
        """,
        params,
    )
    neighbours: dict[str, set[str]] = {}
    for row in (*sql_rows, *fk_out_rows, *fk_in_rows):
        neighbours.setdefault(row["id"], set()).add(row["other_id"])
    return {tid: len(others) for tid, others in neighbours.items()}


def _fetch_semantic_exploration_related_nodes(
    node_id: str,
    zone_ids: list[str] | None,
    *,
    skip: int,
    limit: int | None,
) -> dict[str, Any]:
    """Page Terms related under the existing three-path, zone-safe semantics.

    ``term_is_in_scope`` checks just ``node_id`` rather than building the
    whole in-scope set via ``fetch_all_terms``, and the relationship
    counts are scoped to the page just fetched.

    Which terms are related has to be resolved in full — it is what ``total``
    counts — but only the page's own rows are read: ``fetch_related_term_ids``
    yields ids, and ``fetch_terms_by_ids`` orders and pages them in Neo4j
    under the same rule ``fetch_related_terms`` uses, so a term can't move
    between pages.

    *zone_ids* is resolved to accessible catalog ids exactly once here (see
    ``resolve_accessible_catalog_ids``) and threaded through the visibility
    check, the related-id pass and the counts — five separate resolutions of
    the same zones, otherwise, for one request.
    """
    data_ids_by_zone = resolve_accessible_catalog_ids(zone_ids)
    if not term_is_in_scope(node_id, zone_ids, data_ids_by_zone):
        return {"nodes": [], "total": 0}

    related_ids = fetch_related_term_ids(node_id, zone_ids, data_ids_by_zone)
    if not related_ids:
        return {"nodes": [], "total": 0}

    page = fetch_terms_by_ids(related_ids, skip=skip, limit=limit)
    page_ids = [term["id"] for term in page]

    relationship_counts = _fetch_term_relationship_counts(
        page_ids, zone_ids=zone_ids, data_ids_by_zone=data_ids_by_zone
    )
    return {
        "nodes": [
            {
                "id": term["id"],
                "name": term.get("name") or "",
                "relationship_count": relationship_counts.get(term["id"], 0),
            }
            for term in page
        ],
        "total": len(related_ids),
    }


def _fetch_term_relationship_counts(
    term_ids: list[str],
    zone_ids: list[str] | None,
    data_ids_by_zone: dict[str, set[str]] | None = None,
) -> dict[str, int]:
    """Return ``{term_id: relationship_count}`` for a small, known set of terms.

    Companion to ``_fetch_semantic_exploration_related_nodes``: rendering its
    "Relationships" column needs each *returned* term's own count of related
    terms. Reuses the Terms list's own count so a term's number is the same
    wherever it is shown.
    """
    return {
        row["term_id"]: row["count"]
        for row in fetch_related_terms_counts(
            zone_ids, term_ids=term_ids, data_ids_by_zone=data_ids_by_zone
        )
    }


def fetch_table_zones_map(
    zone_ids: list[str] | None = None,
    data_ids_by_zone: dict[str, set[str]] | None = None,
    table_ids: list[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Return ``{table_id: [zone, ...]}`` for every visible Table.

    Zones are resolved via ``Zone -[ZONE_OF]-> item`` where *item* is the
    table itself or one of its ancestors (Schema, Database), matching the
    resolution used by ``get_full_term_by_id`` / ``get_full_sql_attribute_by_id``.
    ``item`` is found by walking ``CONTAINS`` backwards from ``t`` (0..2 hops,
    0 meaning ``item`` is ``t`` itself) rather than matching ``t`` and ``item``
    independently and filtering afterwards — keeping every ``MATCH`` chained
    through a shared variable avoids a cartesian product between all tables
    and all zone/item pairs on large catalogs.
    Used to render Zone chips in the Exploration graph without a per-node
    request. When *zone_ids* is supplied, both the visible tables and the
    returned zone names/colors are restricted to that set. Disabled zones
    are included (with ``enabled: False``) so admins can see and manage
    them; they never grant access since *zone_ids* itself is computed from
    enabled zones only. Pass ``None`` to return zones for every table
    (admin / internal callers). Pass a pre-resolved *data_ids_by_zone*
    (see ``resolve_accessible_catalog_ids``) when the caller already
    resolved *zone_ids* for this request, to skip a repeat Neo4j round trip.
    Pass *table_ids* to further restrict the scan to a known set of tables
    (e.g. the owning tables of a batch of ColumnAttributes); it is
    intersected with the zone-accessible tables so it never widens access.
    """
    data_ids_by_zone = resolve_accessible_catalog_ids(zone_ids, data_ids_by_zone)
    params: dict[str, Any] = {}
    filter_ids: set[str] | None = None
    if data_ids_by_zone is not None:
        filter_ids = set(data_ids_by_zone["table_ids"])
    if table_ids is not None:
        filter_ids = (
            set(table_ids) if filter_ids is None else filter_ids & set(table_ids)
        )
    table_filter = ""
    if filter_ids is not None:
        if not filter_ids:
            return {}
        table_filter = "WHERE t.id IN $table_ids"
        params["table_ids"] = list(filter_ids)

    # A viewer never sees a disabled zone's chip, even if its id ended up in
    # zone_ids (e.g. access granted before the zone was disabled) — admins
    # (zone_ids=None) still see disabled zones so they can manage them.
    zone_filter = (
        ""
        if zone_ids is None
        else f"WHERE z.id IN $zone_ids AND NOT z:{LABEL_ZONE_DISABLED}"
    )
    if zone_ids is not None:
        params["zone_ids"] = zone_ids

    rows = get_neo4j_conn().query_read(
        f"""
        MATCH (t:{Labels.TABLE})
        {table_filter}
        MATCH (item)-[:{Edges.CONTAINS}*0..2]->(t)
        MATCH (z:{ZONE_LABEL_PATTERN})-[:{REL_ZONE_OF}]->(item)
        {zone_filter}
        RETURN DISTINCT t.id   AS table_id,
                        z.id    AS id,
                        z.name  AS name,
                        z.color AS color,
                        NOT z:{LABEL_ZONE_DISABLED} AS enabled
        ORDER BY t.id, z.name
        """,
        params,
    )
    result: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        result.setdefault(row["table_id"], []).append(
            {
                "id": row["id"],
                "name": row["name"],
                "color": row["color"],
                "enabled": row["enabled"],
            }
        )
    return result


def fetch_data_exploration_graph(
    zone_ids: list[str] | None = None,
    limit: int = MAX_EXPLORATION_GRAPH_NODES,
) -> dict[str, list[dict[str, Any]]]:
    """Return the whole data-layer Exploration graph in one payload.

    Builds ``{"nodes": [...], "links": [...]}`` server-side so the client
    renders the data graph from a single request instead of walking the
    catalog tree (databases → schemas → tables) with one request per level.

    Each node is a visible Table with its column / SQL / Term counts, its
    relationship count, owning Database and Schema ids and names, and
    resolved Zone chips. Links are the SQL- and foreign-key-backed table
    connections from ``fetch_data_exploration_edges``. When *zone_ids* is
    supplied both nodes and links are restricted to tables reachable through
    those zones.

    *limit* caps the number of nodes returned — always clamped to
    ``MAX_EXPLORATION_GRAPH_NODES`` regardless of what's passed in, so the
    response stays bounded on catalogs with many tables. Nodes are ordered
    by name before truncating for a deterministic result, and links are
    filtered to only pairs where both ends are among the returned nodes.
    Relationship counts are computed before that truncation, so a table's
    number matches the related-nodes endpoint rather than the subset drawn.

    *zone_ids* is resolved to accessible catalog ids exactly once (see
    ``resolve_accessible_catalog_ids``) and threaded through the node query,
    ``fetch_table_zones_map`` and ``fetch_data_exploration_edges``, each of
    which would otherwise re-resolve it for itself.
    """
    limit = max(1, min(limit, MAX_EXPLORATION_GRAPH_NODES))
    data_ids_by_zone = resolve_accessible_catalog_ids(zone_ids)
    where_clause, params = resolve_table_filter(
        zone_ids, "t.id", data_ids_by_zone=data_ids_by_zone
    )
    nodes = get_neo4j_conn().query_read(
        f"""
        MATCH (db:{Labels.DB})-[:{Edges.CONTAINS}]->
              (s:{Labels.SCHEMA})-[:{Edges.CONTAINS}]->
              (t:{Labels.TABLE})
        {where_clause}
        {TABLE_COUNTS_SUBQUERY}
        RETURN t.id AS id,
               t.name AS name,
               t.table_type AS table_type,
               db.id AS database_id,
               db.name AS database_name,
               s.id AS schema_id,
               s.name AS schema_name,
               {table_description_expr("t")} AS description,
               columns_count,
               sql_count,
               size(unique_term_ids) AS terms_count
        ORDER BY name
        LIMIT $limit
        """,
        {**params, "limit": limit},
    )
    node_ids = {row["id"] for row in nodes}
    zones_by_table = fetch_table_zones_map(zone_ids, data_ids_by_zone=data_ids_by_zone)
    edges = fetch_data_exploration_edges(
        zone_ids=zone_ids, data_ids_by_zone=data_ids_by_zone
    )
    # Degree over the whole accessible edge set, not just the links returned:
    # `limit` can leave a neighbour out of the payload, and this has to agree
    # with what ``fetch_exploration_related_nodes`` reports for the same table.
    relationship_counts: Counter[str] = Counter()
    for edge in edges:
        relationship_counts[edge["source"]] += 1
        relationship_counts[edge["target"]] += 1
    return {
        "nodes": [
            {
                **dict(row),
                "zones": zones_by_table.get(row["id"], []),
                "relationship_count": relationship_counts[row["id"]],
            }
            for row in nodes
        ],
        "links": [
            edge
            for edge in edges
            if edge["source"] in node_ids and edge["target"] in node_ids
        ],
    }


def fetch_semantic_exploration_graph(
    zone_ids: list[str] | None = None,
    limit: int = MAX_EXPLORATION_GRAPH_NODES,
) -> dict[str, list[dict[str, Any]]]:
    """Return the whole semantic-layer Exploration graph in one payload.

    Builds ``{"nodes": [...], "links": [...]}`` server-side so the client
    renders the semantic graph from a single request instead of fetching
    related terms once per node (an N+1 over every term).

    Each node is a Term with its resolved Zone chips, related-term count and
    ColumnAttribute / SqlAttribute counts. Links are the undirected
    term↔term relationships (two terms sharing at least one table) between
    visible terms. All counts reuse the same helpers as the ``/terms`` list
    and the per-term counts endpoints, so numbers match across pages. When
    *zone_ids* is supplied nodes, counts and links are all zone-scoped.

    *limit* caps the number of nodes returned — always clamped to
    ``MAX_EXPLORATION_GRAPH_NODES`` regardless of what's passed in, so the
    response stays bounded on catalogs with many terms. Relationship counts
    are computed from the full (untruncated) graph so numbers stay accurate
    even when a related term itself falls outside the returned set; nodes
    are then ordered by name and truncated, and links are filtered to only
    pairs where both ends are among the returned nodes.

    *zone_ids* is resolved to accessible catalog ids exactly once (see
    ``resolve_accessible_catalog_ids``) and threaded through every sub-query
    below, each of which would otherwise re-resolve it for itself.
    """
    limit = max(1, min(limit, MAX_EXPLORATION_GRAPH_NODES))
    data_ids_by_zone = resolve_accessible_catalog_ids(zone_ids)

    terms = fetch_all_terms(zone_ids=zone_ids, data_ids_by_zone=data_ids_by_zone)
    column_counts = {
        row["term_id"]: row["count"]
        for row in fetch_column_attribute_counts(
            zone_ids=zone_ids, data_ids_by_zone=data_ids_by_zone
        )
    }
    sql_counts = {
        row["term_id"]: row["count"]
        for row in fetch_sql_attribute_counts(
            zone_ids=zone_ids, data_ids_by_zone=data_ids_by_zone
        )
    }

    term_tables, table_terms = build_term_table_maps(
        fetch_term_table_pairs(zone_ids, data_ids_by_zone=data_ids_by_zone)
    )
    visible_term_ids = {term["id"] for term in terms}
    relationship_counts: dict[str, int] = {}
    link_keys: set[tuple[str, str]] = set()
    for term_id, tables in term_tables.items():
        related: set[str] = set()
        for table_id in tables:
            related.update(table_terms.get(table_id, set()))
        related.discard(term_id)
        relationship_counts[term_id] = len(related)
        if term_id not in visible_term_ids:
            continue
        for other_id in related:
            if other_id in visible_term_ids:
                link_keys.add(tuple(sorted((term_id, other_id))))

    nodes = sorted(
        (
            {
                "id": term["id"],
                "name": term["name"],
                "description": term.get("description"),
                "synonyms": term.get("synonyms") or [],
                "zones": term.get("zones") or [],
                "relationship_count": relationship_counts.get(term["id"], 0),
                "column_attributes_count": column_counts.get(term["id"], 0),
                "sql_attributes_count": sql_counts.get(term["id"], 0),
            }
            for term in terms
        ),
        key=lambda node: node["name"],
    )[:limit]
    kept_ids = {node["id"] for node in nodes}
    links = [
        {"source": source, "target": target}
        for source, target in sorted(link_keys)
        if source in kept_ids and target in kept_ids
    ]
    return {"nodes": nodes, "links": links}
