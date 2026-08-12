# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The data- and semantic-layer Exploration graphs.

Most of this module composes helpers that already exist — which is why the plan
put it near the end. The semantic graph in particular is almost entirely calls
into ``pg/terms`` and ``pg/sql_attributes``.

**One invariant runs through all of it, and it is the thing worth protecting:**
a table's ``relationship_count`` in the graph payload must equal the ``total``
its related-nodes page reports. They are computed by different code — the graph
counts edges, the page counts neighbours — so "related" has to mean exactly the
same thing in both. :func:`_related_tables` is that definition, and both sides
read it from there.

The counts are also deliberately computed **before** truncation. ``limit`` can
leave a neighbour out of the payload, and a node whose degree shrank because its
neighbour was not drawn would contradict the page behind it.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

from sqlalchemy import func, select

from gsf.dal import schema as s

# Private on purpose, and imported across modules on purpose: these are the one
# definition of a table's column/sql/term counts, and the SQL shared them the
# same way -- `TABLE_COUNTS_SUBQUERY`, imported across modules. A second
# copy here is how the graph's badge and the schema tree's badge start
# disagreeing about the same table.
from gsf.dal.datasources import _count_of, _terms_count
from gsf.dal.session import store
from gsf.dal.sql_attributes import fetch_sql_attribute_counts
from gsf.dal.sql_fragments import table_description_expr
from gsf.dal.terms import (
    build_term_table_maps,
    fetch_all_terms,
    fetch_column_attribute_counts,
    fetch_related_term_ids,
    fetch_related_terms_counts,
    fetch_term_table_pairs,
    fetch_terms_by_ids,
    term_is_in_scope,
)
from gsf.dal.users import resolve_accessible_catalog_ids

# Re-exported, not reimplemented: the Term reads need it too, and two spellings
# of the zone-resolution rule is how a viewer ends up seeing a chip on one
# screen and not another.
from gsf.dal.zones import fetch_table_zones_map  # noqa: F401

logger = logging.getLogger(__name__)

#: Hard ceiling on nodes returned by an Exploration graph endpoint, whatever a
#: caller asks for — keeps the response bounded on large catalogs.
MAX_EXPLORATION_GRAPH_NODES = 500


def _non_empty_sql():
    """A statement with no text is not a data-layer edge.

    Every read deriving table-to-table relationships applies this, or the same
    pair of tables is connected in one view and unconnected in another.
    """
    return func.nullif(func.trim(s.sql_query.c.sql_full_query), "").isnot(None)


# ---------------------------------------------------------------------------
# What "related" means, in one place
# ---------------------------------------------------------------------------


def _shared_sql_pairs(table_filter=None):
    """Table pairs referenced together by one non-empty statement."""
    source = s.sql_query_table.alias("sql_source")
    target = s.sql_query_table.alias("sql_target")
    statement = (
        select(
            source.c.table_id.label("source"),
            target.c.table_id.label("target"),
            s.sql_query.c.sql_full_query,
        )
        .select_from(
            source.join(target, source.c.sql_query_id == target.c.sql_query_id).join(
                s.sql_query, s.sql_query.c.id == source.c.sql_query_id
            )
        )
        .where(_non_empty_sql())
    )
    if table_filter is not None:
        statement = statement.where(
            source.c.table_id.in_(table_filter), target.c.table_id.in_(table_filter)
        )
    return statement, source, target


def _fk_column_pairs(table_filter=None):
    """Foreign-key column pairs whose two columns live on different tables."""
    source_column = s.catalog_column.alias("fk_source_column")
    target_column = s.catalog_column.alias("fk_target_column")
    statement = (
        select(
            source_column.c.table_id.label("source"),
            target_column.c.table_id.label("target"),
            source_column.c.name.label("source_column"),
            target_column.c.name.label("target_column"),
            source_column.c.sample_values.label("source_sample_values"),
            target_column.c.sample_values.label("target_sample_values"),
        )
        .select_from(
            s.column_foreign_key.join(
                source_column,
                source_column.c.id == s.column_foreign_key.c.source_column_id,
            ).join(
                target_column,
                target_column.c.id == s.column_foreign_key.c.target_column_id,
            )
        )
        .where(source_column.c.table_id != target_column.c.table_id)
        .distinct()
    )
    if table_filter is not None:
        statement = statement.where(
            source_column.c.table_id.in_(table_filter),
            target_column.c.table_id.in_(table_filter),
        )
    return statement


def _related_tables(node_id: str, table_filter: list[str] | None):
    """Every table related to *node_id*, as a set of ids.

    **The single definition of a data-layer relationship**, read by both the
    graph's degree count and the related-nodes page. A shared statement with
    text, or a foreign key in *either* direction — the two FK legs are separate
    because a foreign key is stored one way round and relatedness is not.
    """
    sql_pairs, source, target = _shared_sql_pairs(table_filter)
    shared = (
        sql_pairs.with_only_columns(target.c.table_id.label("related_id"))
        .where(source.c.table_id == node_id, target.c.table_id != node_id)
        .distinct()
    )

    fk_pairs = _fk_column_pairs(table_filter).subquery("fk_pairs")
    outgoing = (
        select(fk_pairs.c.target.label("related_id"))
        .where(fk_pairs.c.source == node_id, fk_pairs.c.target != node_id)
        .distinct()
    )
    incoming = (
        select(fk_pairs.c.source.label("related_id"))
        .where(fk_pairs.c.target == node_id, fk_pairs.c.source != node_id)
        .distinct()
    )
    return shared.union(outgoing, incoming)


# ---------------------------------------------------------------------------
# Data layer
# ---------------------------------------------------------------------------


def fetch_data_exploration_edges(
    zone_ids: list[str] | None = None,
    data_ids_by_zone: dict[str, set[str]] | None = None,
) -> list[dict[str, Any]]:
    """Table pairs connected by a shared statement or a foreign key.

    A statement becomes an edge when it references at least two visible tables;
    the edge carries the SQL text so the client can show it. A foreign key also
    makes an edge — flagged ``via_foreign_key`` — even when no stored statement
    ever referenced both tables together, and every edge carries a
    ``foreign_keys`` list so the client can show *which* columns join the pair.
    A pair connected both ways is one merged edge, not two.

    Edge keys are normalised to ``(min id, max id)``. When a foreign-key row
    arrives reversed relative to that order its column pair is swapped with it,
    so ``foreign_keys[].source_column`` always names a column on
    ``edge["source"]`` — getting that wrong renders a join backwards, which
    looks plausible and is wrong.
    """
    resolved = resolve_accessible_catalog_ids(zone_ids, data_ids_by_zone)
    table_filter: list[str] | None = None
    if resolved is not None:
        table_filter = list(resolved["table_ids"])
        if not table_filter:
            return []

    sql_pairs, source, target = _shared_sql_pairs(table_filter)
    sql_rows = store().query_read(
        sql_pairs.where(source.c.table_id < target.c.table_id).distinct()
    )
    fk_rows = store().query_read(_fk_column_pairs(table_filter))

    edges: dict[tuple[str, str], dict[str, Any]] = {}
    for row in sql_rows:
        key = (row["source"], row["target"])
        edge = edges.setdefault(
            key,
            {
                "source": key[0],
                "target": key[1],
                "queries": [],
                "via_foreign_key": False,
                "foreign_keys": [],
            },
        )
        if row["sql_full_query"] not in edge["queries"]:
            edge["queries"].append(row["sql_full_query"])

    for row in fk_rows:
        a, b = row["source"], row["target"]
        flipped = a > b
        key = (b, a) if flipped else (a, b)
        detail = {
            "source_column": row["target_column" if flipped else "source_column"],
            "target_column": row["source_column" if flipped else "target_column"],
            "source_sample_values": row[
                "target_sample_values" if flipped else "source_sample_values"
            ],
            "target_sample_values": row[
                "source_sample_values" if flipped else "target_sample_values"
            ],
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
        if detail not in edge["foreign_keys"]:
            edge["foreign_keys"].append(detail)

    return sorted(edges.values(), key=lambda edge: (edge["source"], edge["target"]))


def fetch_table_exploration_details(
    table_id: str,
    zone_ids: list[str] | None = None,
    *,
    skip: int = 0,
    limit: int | None = None,
) -> dict[str, Any]:
    """The statements referencing a table, and one page of its Terms.

    **The blank-statement filter here is looser than the edge set's**, and that
    is inherited rather than chosen. ``_non_empty_sql`` trims before deciding,
    so a whitespace-only statement makes no edge; this list drops only *falsy*
    text, so the same statement is still listed among the table's queries. The
    visible effect is a query on the detail panel with no line on the graph
    beside it. Left alone — nothing is lost by it, and unifying the two would
    change what the panel shows — but pinned by a test so it cannot be
    "tidied" silently.
    """
    resolved = resolve_accessible_catalog_ids(zone_ids)
    if resolved is not None and table_id not in resolved["table_ids"]:
        return {"queries": [], "terms": [], "terms_total": 0}

    queries = store().query_read(
        select(s.sql_query.c.id, s.sql_query.c.sql_full_query.label("sql"))
        .select_from(
            s.sql_query_table.join(
                s.sql_query, s.sql_query.c.id == s.sql_query_table.c.sql_query_id
            )
        )
        .where(s.sql_query_table.c.table_id == table_id)
        .distinct()
        .order_by(s.sql_query.c.id)
    )

    # Both Term routes a table has, unioned: REPRESENTS, and through a column's
    # attribute by either link. Same three paths `fetch_related_terms` uses.
    represents = (
        select(s.term.c.id, s.term.c.name, s.term.c.description)
        .select_from(s.table_term.join(s.term, s.term.c.id == s.table_term.c.term_id))
        .where(s.table_term.c.table_id == table_id)
    )
    link = (
        select(
            s.column_has_attribute.c.column_id, s.column_has_attribute.c.attribute_id
        )
        .union(
            select(
                s.column_semantic_fk.c.column_id, s.column_semantic_fk.c.attribute_id
            )
        )
        .subquery("attribute_link")
    )
    via_attribute = (
        select(s.term.c.id, s.term.c.name, s.term.c.description)
        .select_from(
            s.catalog_column.join(link, link.c.column_id == s.catalog_column.c.id)
            .join(
                s.column_attribute_term,
                s.column_attribute_term.c.attribute_id == link.c.attribute_id,
            )
            .join(s.term, s.term.c.id == s.column_attribute_term.c.term_id)
        )
        .where(s.catalog_column.c.table_id == table_id)
    )
    all_terms = represents.union(via_attribute).subquery("table_terms")

    page = select(all_terms).order_by(all_terms.c.name, all_terms.c.id).offset(skip)
    if limit is not None:
        page = page.limit(limit)
    terms = store().query_read(page)
    total = store().query_read(
        select(func.count(func.distinct(all_terms.c.id)).label("total")).select_from(
            all_terms
        )
    )

    return {
        "queries": [
            {"id": row["id"] or "", "sql": row["sql"] or ""}
            for row in queries
            if row["sql"]
        ],
        "terms": [dict(row) for row in terms if row["id"]],
        "terms_total": int(total[0]["total"]) if total else 0,
    }


def _fetch_table_relationship_counts(
    table_ids: list[str],
    zone_table_ids: list[str] | None,
) -> dict[str, int]:
    """``{table_id: degree}`` for a small, known set of tables.

    The "Relationships" column of the related-nodes page needs each *returned*
    table's own degree, under the same definition of an edge — so it is
    re-derived here rather than counted from the page, which only holds one
    node's neighbours.
    """
    if not table_ids:
        return {}

    sql_pairs, source, target = _shared_sql_pairs(zone_table_ids)
    shared = (
        sql_pairs.with_only_columns(
            source.c.table_id.label("id"), target.c.table_id.label("other_id")
        )
        .where(
            source.c.table_id.in_(table_ids),
            source.c.table_id != target.c.table_id,
        )
        .distinct()
    )

    fk_pairs = _fk_column_pairs(zone_table_ids).subquery("fk_pairs")
    outgoing = (
        select(fk_pairs.c.source.label("id"), fk_pairs.c.target.label("other_id"))
        .where(fk_pairs.c.source.in_(table_ids))
        .distinct()
    )
    incoming = (
        select(fk_pairs.c.target.label("id"), fk_pairs.c.source.label("other_id"))
        .where(fk_pairs.c.target.in_(table_ids))
        .distinct()
    )

    neighbours: dict[str, set[str]] = {}
    for row in store().query_read(shared.union(outgoing, incoming)):
        neighbours.setdefault(row["id"], set()).add(row["other_id"])
    return {table_id: len(others) for table_id, others in neighbours.items()}


def _fetch_data_exploration_related_nodes(
    node_id: str,
    zone_ids: list[str] | None,
    *,
    skip: int,
    limit: int | None,
) -> dict[str, Any]:
    """Page the tables related to *node_id*.

    Scoped to *node_id* throughout rather than building the whole edge set and
    filtering it down in Python — on a large catalog that is the difference
    between one indexed read and a full scan.

    The catalog path is an **outer** join for a specific reason: requiring it
    would drop a related table with no schema or database above it from the
    page while the graph still counted it, and the two numbers must agree.
    """
    resolved = resolve_accessible_catalog_ids(zone_ids)
    if resolved is not None and node_id not in resolved["table_ids"]:
        return {"nodes": [], "total": 0}

    zone_table_ids = None if resolved is None else list(resolved["table_ids"])
    related = _related_tables(node_id, zone_table_ids).subquery("related")

    total_rows = store().query_read(
        select(
            func.count(func.distinct(related.c.related_id)).label("total")
        ).select_from(related)
    )
    total = int(total_rows[0]["total"]) if total_rows else 0
    if not total:
        return {"nodes": [], "total": 0}

    page = (
        select(
            s.catalog_table.c.id,
            s.catalog_table.c.name,
            s.catalog_table.c.table_type,
            s.catalog_database.c.id.label("database_id"),
            s.catalog_schema.c.id.label("schema_id"),
        )
        .select_from(
            s.catalog_table.outerjoin(
                s.catalog_schema, s.catalog_schema.c.id == s.catalog_table.c.schema_id
            ).outerjoin(
                s.catalog_database,
                s.catalog_database.c.id == s.catalog_schema.c.database_id,
            )
        )
        .where(s.catalog_table.c.id.in_(select(related.c.related_id)))
        .order_by(s.catalog_table.c.name, s.catalog_table.c.id)
        .offset(skip)
    )
    if limit is not None:
        page = page.limit(limit)

    rows = [dict(r) for r in store().query_read(page)]
    counts = _fetch_table_relationship_counts(
        [row["id"] for row in rows], zone_table_ids=zone_table_ids
    )
    return {
        "nodes": [
            {**row, "relationship_count": counts.get(row["id"], 0)} for row in rows
        ],
        "total": total,
    }


# ---------------------------------------------------------------------------
# Semantic layer
# ---------------------------------------------------------------------------


def _fetch_term_relationship_counts(
    term_ids: list[str],
    zone_ids: list[str] | None,
    data_ids_by_zone: dict[str, set[str]] | None = None,
) -> dict[str, int]:
    """``{term_id: related count}``, reusing the Terms list's own count.

    Reused rather than recomputed so a term's number is the same wherever it is
    shown — the card, the column, and the length of the list behind them.
    """
    return {
        row["term_id"]: row["count"]
        for row in fetch_related_terms_counts(
            zone_ids, term_ids=term_ids, data_ids_by_zone=data_ids_by_zone
        )
    }


def _fetch_semantic_exploration_related_nodes(
    node_id: str,
    zone_ids: list[str] | None,
    *,
    skip: int,
    limit: int | None,
) -> dict[str, Any]:
    """Page the Terms related to *node_id*.

    Which terms are related has to be resolved in full — that is what ``total``
    counts — but only the page's rows are read: ``fetch_related_term_ids``
    yields ids and ``fetch_terms_by_ids`` orders and pages them in SQL under the
    same rule the related list uses, so a term cannot move between pages.

    *zone_ids* is resolved **once** here and threaded through the visibility
    check, the related-id pass and the counts. Left to themselves those would
    re-resolve the same zones five times for one request.
    """
    resolved = resolve_accessible_catalog_ids(zone_ids)
    if not term_is_in_scope(node_id, zone_ids, resolved):
        return {"nodes": [], "total": 0}

    related_ids = fetch_related_term_ids(node_id, zone_ids, resolved)
    if not related_ids:
        return {"nodes": [], "total": 0}

    page = fetch_terms_by_ids(related_ids, skip=skip, limit=limit)
    counts = _fetch_term_relationship_counts(
        [term["id"] for term in page], zone_ids=zone_ids, data_ids_by_zone=resolved
    )
    return {
        "nodes": [
            {
                "id": term["id"],
                "name": term.get("name") or "",
                "relationship_count": counts.get(term["id"], 0),
            }
            for term in page
        ],
        "total": len(related_ids),
    }


def fetch_exploration_related_nodes(
    node_id: str,
    layer: str,
    zone_ids: list[str] | None = None,
    *,
    skip: int = 0,
    limit: int | None = None,
) -> dict[str, Any]:
    """One page of nodes related to an Exploration node.

    Resolves relationships from the **uncapped** graph definitions rather than
    from either graph endpoint's payload: a neighbour dropped by
    ``MAX_EXPLORATION_GRAPH_NODES`` is still a neighbour, and a page that
    disagreed with the degree drawn on the node would be worse than either.
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


# ---------------------------------------------------------------------------
# Whole-graph payloads
# ---------------------------------------------------------------------------


def fetch_data_exploration_graph(
    zone_ids: list[str] | None = None,
    limit: int = MAX_EXPLORATION_GRAPH_NODES,
) -> dict[str, list[dict[str, Any]]]:
    """The whole data-layer graph in one payload.

    Built server-side so the client renders it from one request instead of
    walking databases → schemas → tables a level at a time.

    *limit* is clamped to :data:`MAX_EXPLORATION_GRAPH_NODES` whatever is
    passed. Nodes are ordered by name before truncating, so the subset is
    deterministic, and links are filtered to pairs with both ends drawn.

    **Relationship counts are computed before truncation**, over the whole
    accessible edge set — so a table's number matches what the related-nodes
    endpoint reports rather than shrinking because a neighbour was not drawn.
    """
    limit = max(1, min(limit, MAX_EXPLORATION_GRAPH_NODES))
    resolved = resolve_accessible_catalog_ids(zone_ids)

    statement = (
        select(
            s.catalog_table.c.id,
            s.catalog_table.c.name,
            s.catalog_table.c.table_type,
            s.catalog_database.c.id.label("database_id"),
            s.catalog_database.c.name.label("database_name"),
            s.catalog_schema.c.id.label("schema_id"),
            s.catalog_schema.c.name.label("schema_name"),
            table_description_expr().label("description"),
            _count_of(
                s.catalog_column, s.catalog_column.c.table_id == s.catalog_table.c.id
            ).label("columns_count"),
            _count_of(
                s.sql_query_table, s.sql_query_table.c.table_id == s.catalog_table.c.id
            ).label("sql_count"),
            _terms_count(s.catalog_table.c.id).label("terms_count"),
        )
        .select_from(
            s.catalog_table.join(
                s.catalog_schema, s.catalog_schema.c.id == s.catalog_table.c.schema_id
            ).join(
                s.catalog_database,
                s.catalog_database.c.id == s.catalog_schema.c.database_id,
            )
        )
        .order_by(s.catalog_table.c.name)
        .limit(limit)
    )
    if resolved is not None:
        statement = statement.where(
            s.catalog_table.c.id.in_(list(resolved["table_ids"]))
        )

    nodes = [dict(r) for r in store().query_read(statement)]
    node_ids = {row["id"] for row in nodes}
    zones_by_table = fetch_table_zones_map(zone_ids, data_ids_by_zone=resolved)
    edges = fetch_data_exploration_edges(zone_ids=zone_ids, data_ids_by_zone=resolved)

    degree: Counter[str] = Counter()
    for edge in edges:
        degree[edge["source"]] += 1
        degree[edge["target"]] += 1

    return {
        "nodes": [
            {
                **row,
                "zones": zones_by_table.get(row["id"], []),
                "relationship_count": degree[row["id"]],
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
    """The whole semantic-layer graph in one payload.

    Built server-side so the client renders it from one request instead of
    fetching related terms once per node — an N+1 over the whole glossary.

    Every count reuses the helper the corresponding list endpoint uses, so the
    numbers match across pages. As above, counts come from the untruncated
    graph; nodes are then ordered by name and cut to *limit*, and links are kept
    only where both ends survived.
    """
    limit = max(1, min(limit, MAX_EXPLORATION_GRAPH_NODES))
    resolved = resolve_accessible_catalog_ids(zone_ids)

    terms = fetch_all_terms(zone_ids=zone_ids, data_ids_by_zone=resolved)
    column_counts = {
        row["term_id"]: row["count"]
        for row in fetch_column_attribute_counts(
            zone_ids=zone_ids, data_ids_by_zone=resolved
        )
    }
    sql_counts = {
        row["term_id"]: row["count"]
        for row in fetch_sql_attribute_counts(
            zone_ids=zone_ids, data_ids_by_zone=resolved
        )
    }

    term_tables, table_terms = build_term_table_maps(
        fetch_term_table_pairs(zone_ids, data_ids_by_zone=resolved)
    )
    visible = {term["id"] for term in terms}
    degree: dict[str, int] = {}
    link_keys: set[tuple[str, str]] = set()
    for term_id, tables in term_tables.items():
        related: set[str] = set()
        for table_id in tables:
            related.update(table_terms.get(table_id, set()))
        related.discard(term_id)
        degree[term_id] = len(related)
        if term_id not in visible:
            continue
        for other_id in related:
            if other_id in visible:
                link_keys.add(tuple(sorted((term_id, other_id))))

    nodes = sorted(
        (
            {
                "id": term["id"],
                "name": term["name"],
                "description": term.get("description"),
                "synonyms": term.get("synonyms") or [],
                "zones": term.get("zones") or [],
                "relationship_count": degree.get(term["id"], 0),
                "column_attributes_count": column_counts.get(term["id"], 0),
                "sql_attributes_count": sql_counts.get(term["id"], 0),
            }
            for term in terms
        ),
        key=lambda node: node["name"],
    )[:limit]
    kept = {node["id"] for node in nodes}
    return {
        "nodes": nodes,
        "links": [
            {"source": source, "target": target}
            for source, target in sorted(link_keys)
            if source in kept and target in kept
        ],
    }


#: Spelled out because ``fetch_table_zones_map`` is **re-exported** from
#: ``zones`` rather than defined here, and the surface freeze resolves a
#: module's contract by ``__module__`` — which for a re-export points at the
#: definition, not at this file.
__all__ = [
    "MAX_EXPLORATION_GRAPH_NODES",
    "fetch_data_exploration_edges",
    "fetch_data_exploration_graph",
    "fetch_exploration_related_nodes",
    "fetch_semantic_exploration_graph",
    "fetch_table_exploration_details",
    "fetch_table_zones_map",
]
