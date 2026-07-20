# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Neo4j data access for the data- and semantic-layer Exploration graphs.

Contains only functions that call ``get_neo4j_conn()`` directly (aside from
the internal calls each graph builder makes to its own sibling functions
below). All read functions use the ``fetch_*`` prefix.
"""

from __future__ import annotations

from typing import Any

from nemo_retriever.tabular_data.ingestion.model.reserved_words import Edges, Labels
from nemo_retriever.tabular_data.neo4j import get_neo4j_conn

from gsf.dal.datasources import TABLE_COUNTS_SUBQUERY
from gsf.dal.sql_attributes import fetch_sql_attribute_counts
from gsf.dal.terms import (
    build_term_table_maps,
    fetch_all_terms_and_attributes,
    fetch_column_attribute_counts,
    fetch_term_table_pairs,
)
from gsf.dal.users import resolve_accessible_catalog_ids, resolve_table_filter
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
        WHERE {table_filter}source.id < target.id
        WITH source, target,
             collect(DISTINCT sql.sql_full_query) AS raw_queries
        RETURN source.id AS source,
               target.id AS target,
               [query IN raw_queries
                WHERE query IS NOT NULL AND trim(toString(query)) <> ''] AS queries
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
            "queries": [q for q in (row.get("queries") or []) if q],
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

    return sorted(
        (edge for edge in edges.values() if edge["queries"] or edge["via_foreign_key"]),
        key=lambda edge: (edge["source"], edge["target"]),
    )


def fetch_table_exploration_details(
    table_id: str,
    zone_ids: list[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Return SQL queries and Terms linked to one visible Table."""
    data_ids_by_zone = resolve_accessible_catalog_ids(zone_ids)
    if data_ids_by_zone is not None and table_id not in data_ids_by_zone["table_ids"]:
        return {"queries": [], "terms": []}

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
    terms = conn.query_read(
        f"""
        MATCH (t:{Labels.TABLE} {{id: $table_id}})
              -[:{REL_REPRESENTS}]->(term:{LABEL_TERM})
        RETURN DISTINCT term.id AS id, term.name AS name,
                        term.description AS description
        UNION
        MATCH (t:{Labels.TABLE} {{id: $table_id}})
              -[:{Edges.CONTAINS}]->(:{Labels.COLUMN})
              -[:{REL_HAS_ATTRIBUTE}|{REL_SEMANTIC_FK}]->
              (:{LABEL_COLUMN_ATTRIBUTE})-[:{REL_PROPERTY_OF}]->
              (term:{LABEL_TERM})
        RETURN DISTINCT term.id AS id, term.name AS name,
                        term.description AS description
        """,
        {"table_id": table_id},
    )
    unique_terms = {row["id"]: row for row in terms if row.get("id")}
    return {
        "queries": [
            {"id": row.get("id") or "", "sql": row.get("sql") or ""}
            for row in queries
            if row.get("sql")
        ],
        "terms": list(unique_terms.values()),
    }


def fetch_table_zones_map(
    zone_ids: list[str] | None = None,
    data_ids_by_zone: dict[str, set[str]] | None = None,
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
    """
    data_ids_by_zone = resolve_accessible_catalog_ids(zone_ids, data_ids_by_zone)
    params: dict[str, Any] = {}
    table_filter = ""
    if data_ids_by_zone is not None:
        table_ids = list(data_ids_by_zone["table_ids"])
        if not table_ids:
            return {}
        table_filter = "WHERE t.id IN $table_ids"
        params["table_ids"] = table_ids

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

    Each node is a visible Table with its column / SQL / Term counts,
    owning Database and Schema ids and names, and resolved Zone chips.
    Links are the SQL- and foreign-key-backed table connections from
    ``fetch_data_exploration_edges``. When *zone_ids* is supplied both nodes
    and links are restricted to tables reachable through those zones.

    *limit* caps the number of nodes returned — always clamped to
    ``MAX_EXPLORATION_GRAPH_NODES`` regardless of what's passed in, so the
    response stays bounded on catalogs with many tables. Nodes are ordered
    by name before truncating for a deterministic result, and links are
    filtered to only pairs where both ends are among the returned nodes.

    *zone_ids* is resolved to accessible catalog ids exactly once (see
    ``resolve_accessible_catalog_ids``) and threaded through the node query,
    ``fetch_table_zones_map`` and ``fetch_data_exploration_edges`` — those
    three previously each re-resolved the same *zone_ids* independently,
    tripling the Neo4j round trips this endpoint made per request.
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
               t.description AS description,
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
    return {
        "nodes": [
            {**dict(row), "zones": zones_by_table.get(row["id"], [])} for row in nodes
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
    ``resolve_accessible_catalog_ids``) and threaded through every
    sub-query below. Previously each of the four calls below re-resolved
    the same *zone_ids* independently — and ``fetch_all_terms_and_attributes``
    even did so twice internally — for five redundant Neo4j round trips
    collapsed into the one made here.
    """
    limit = max(1, min(limit, MAX_EXPLORATION_GRAPH_NODES))
    data_ids_by_zone = resolve_accessible_catalog_ids(zone_ids)

    terms, _attrs = fetch_all_terms_and_attributes(
        zone_ids=zone_ids, data_ids_by_zone=data_ids_by_zone
    )
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
