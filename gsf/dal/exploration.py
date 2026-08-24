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

from sqlalchemy import and_, func, literal, select

from gsf.catalog.constants import Edges, Labels
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
    find_term_link_path,
    term_is_in_scope,
)
from gsf.dal.users import resolve_accessible_catalog_ids
from gsf.semantic.constants import (
    LABEL_COLUMN_ATTRIBUTE,
    LABEL_TERM,
    REL_HAS_ATTRIBUTE,
    REL_REPRESENTS,
    REL_SEMANTIC_FK,
    SEMANTIC_SOURCE,
)

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


def _table_catalog_join():
    """Table joined up to its schema and database."""
    return s.catalog_table.join(
        s.catalog_schema, s.catalog_schema.c.id == s.catalog_table.c.schema_id
    ).join(
        s.catalog_database, s.catalog_database.c.id == s.catalog_schema.c.database_id
    )


def _column_catalog_join():
    """Column joined up to its table, schema and database."""
    return (
        s.catalog_column.join(
            s.catalog_table, s.catalog_table.c.id == s.catalog_column.c.table_id
        )
        .join(s.catalog_schema, s.catalog_schema.c.id == s.catalog_table.c.schema_id)
        .join(
            s.catalog_database,
            s.catalog_database.c.id == s.catalog_schema.c.database_id,
        )
    )


#: A column, with the catalog path a client needs to expand it any further.
_COLUMN_REF_COLUMNS = (
    s.catalog_column.c.id,
    s.catalog_column.c.name,
    s.catalog_column.c.description,
    s.catalog_column.c.data_type,
    s.catalog_table.c.id.label("table_id"),
    s.catalog_table.c.name.label("table_name"),
    s.catalog_database.c.id.label("database_id"),
    s.catalog_database.c.name.label("database_name"),
    s.catalog_schema.c.id.label("schema_id"),
    s.catalog_schema.c.name.label("schema_name"),
)

#: A table, in the same shape the graph's own table nodes use.
_TABLE_REF_COLUMNS = (
    s.catalog_table.c.id,
    s.catalog_table.c.name,
    s.catalog_table.c.table_type,
    s.catalog_database.c.id.label("database_id"),
    s.catalog_database.c.name.label("database_name"),
    s.catalog_schema.c.id.label("schema_id"),
    s.catalog_schema.c.name.label("schema_name"),
)


#: Cap on the referencing columns one Column expansion returns. A shared
#: lookup-style column can be referenced by very many source columns, and this
#: field is capped rather than paged like every other field on that payload.
_MAX_REFERENCING_COLUMNS = 50


def _visible_table(table_filter: list[str] | None):
    """``catalog_table`` is inside the accessible set, or ``True`` when unscoped."""
    if table_filter is None:
        return literal(True)
    return s.catalog_table.c.id.in_(table_filter)


def _sql_in_scope(table_filter: list[str] | None):
    """A statement is visible only when **every** table it references is.

    The same all-or-nothing rule ``_sql_attr_zone_filter`` applies in
    ``gsf.dal.sql_attributes``: a statement half of whose tables are out of zone
    would otherwise leak the other half's names through its own text.
    """
    if table_filter is None:
        return literal(True)
    return ~(
        select(literal(1))
        .select_from(s.sql_query_table)
        .where(
            s.sql_query_table.c.sql_query_id == s.sql_query.c.id,
            s.sql_query_table.c.table_id.notin_(table_filter),
        )
        .exists()
    )


def _attribute_link():
    """``(column_id, attribute_id, rel_type)`` for both Column→ColumnAttribute links.

    HAS_ATTRIBUTE and SEMANTIC_FK live in separate tables and are unioned here
    with the type carried along, because every reader that walks a column to its
    attribute has to walk both -- a foreign-key-shaped column (a ``user_id``) is
    linked by SEMANTIC_FK rather than HAS_ATTRIBUTE -- while still being able to
    say which one it actually took.
    """
    return (
        select(
            s.column_has_attribute.c.column_id,
            s.column_has_attribute.c.attribute_id,
            literal(REL_HAS_ATTRIBUTE).label("rel_type"),
        )
        .union(
            select(
                s.column_semantic_fk.c.column_id,
                s.column_semantic_fk.c.attribute_id,
                literal(REL_SEMANTIC_FK).label("rel_type"),
            )
        )
        .subquery("attribute_link")
    )


def _term_table_links():
    """``(term_id, table_id, rel_type)`` for every term↔table link.

    The same three paths :func:`gsf.dal.terms.fetch_term_table_pairs` calls a
    link, with the relationship type that made each one. Read from both ends --
    :func:`fetch_table_exploration_details` walks it table→terms and
    :func:`fetch_term_exploration_details` term→tables -- so an expansion edge
    is labelled identically whichever end the user double-clicked.
    """
    represents = (
        select(
            s.table_term.c.term_id,
            s.table_term.c.table_id,
            literal(REL_REPRESENTS).label("rel_type"),
        )
        .select_from(s.table_term.join(s.term, s.term.c.id == s.table_term.c.term_id))
        .where(s.term.c.source == SEMANTIC_SOURCE)
    )
    link = _attribute_link()
    via_attribute = (
        select(
            s.column_attribute_term.c.term_id,
            s.catalog_column.c.table_id,
            link.c.rel_type,
        )
        .select_from(
            s.catalog_column.join(link, link.c.column_id == s.catalog_column.c.id)
            .join(
                s.column_attribute,
                and_(
                    s.column_attribute.c.id == link.c.attribute_id,
                    s.column_attribute.c.source == SEMANTIC_SOURCE,
                ),
            )
            .join(
                s.column_attribute_term,
                s.column_attribute_term.c.attribute_id == s.column_attribute.c.id,
            )
            .join(s.term, s.term.c.id == s.column_attribute_term.c.term_id)
        )
        .where(s.term.c.source == SEMANTIC_SOURCE)
    )
    return represents.union(via_attribute).subquery("term_table_link")


def _relationship_types(raw: Any) -> list[str]:
    """The aggregated ``rel_type`` array as a sorted list of names."""
    return sorted({value for value in (raw or []) if value})


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
                "relationship_types": [Edges.SQL],
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
                "relationship_types": [Edges.FOREIGN_KEY],
            }
            edges[key] = edge
        else:
            edge["via_foreign_key"] = True
            if Edges.FOREIGN_KEY not in edge["relationship_types"]:
                edge["relationship_types"].append(Edges.FOREIGN_KEY)
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

    Each term carries ``relationship_types`` -- the link type(s) actually
    reaching it from this table -- so a client can label the connection the way
    it reads in the graph. A table's own ColumnAttributes are deliberately left
    out: they are two hops away, one further than a table's own expansion should
    reveal in a single double-click, and
    :func:`fetch_column_exploration_details` is what surfaces them instead.

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

    # Every Term route the table has, from the one definition of a term-to-table
    # link -- REPRESENTS, or a column's attribute by either link.
    links = _term_table_links()
    total = store().query_read(
        select(func.count(func.distinct(links.c.term_id)).label("total")).where(
            links.c.table_id == table_id
        )
    )
    page = (
        select(
            s.term.c.id,
            s.term.c.name,
            s.term.c.description,
            func.array_agg(links.c.rel_type).label("rel_types"),
        )
        .select_from(links.join(s.term, s.term.c.id == links.c.term_id))
        .where(links.c.table_id == table_id)
        .group_by(s.term.c.id, s.term.c.name, s.term.c.description)
        .order_by(s.term.c.name, s.term.c.id)
        .offset(skip)
    )
    if limit is not None:
        page = page.limit(limit)

    return {
        "queries": [
            {"id": row["id"] or "", "sql": row["sql"] or ""}
            for row in queries
            if row["sql"]
        ],
        "terms": [
            {
                **{
                    key: value for key, value in dict(row).items() if key != "rel_types"
                },
                "relationship_types": _relationship_types(row["rel_types"]),
            }
            for row in store().query_read(page)
            if row["id"]
        ],
        "terms_total": int(total[0]["total"]) if total else 0,
    }


def fetch_term_exploration_details(
    term_id: str,
    zone_ids: list[str] | None = None,
    *,
    skip: int = 0,
    limit: int | None = None,
) -> dict[str, Any]:
    """One ordered page of the Tables linked to a visible Term.

    The reverse of :func:`fetch_table_exploration_details`'s Term list, down to
    the ``relationship_types`` each row carries — REPRESENTS directly, or the
    HAS_ATTRIBUTE / SEMANTIC_FK link off one of the table's columns — so a
    client labels the edge the same way from either end.
    """
    resolved = resolve_accessible_catalog_ids(zone_ids)
    if not term_is_in_scope(term_id, zone_ids, resolved):
        return {"tables": [], "tables_total": 0}

    links = _term_table_links()
    conditions: list[Any] = [links.c.term_id == term_id]
    if resolved is not None:
        conditions.append(links.c.table_id.in_(list(resolved["table_ids"])))

    total_rows = store().query_read(
        select(func.count(func.distinct(links.c.table_id)).label("total")).where(
            *conditions
        )
    )
    total = int(total_rows[0]["total"]) if total_rows else 0
    if not total:
        return {"tables": [], "tables_total": 0}

    page = (
        select(*_TABLE_REF_COLUMNS, func.array_agg(links.c.rel_type).label("rel_types"))
        .select_from(
            _table_catalog_join().join(links, links.c.table_id == s.catalog_table.c.id)
        )
        .where(*conditions)
        .group_by(*_TABLE_REF_COLUMNS)
        .order_by(s.catalog_table.c.name, s.catalog_table.c.id)
        .offset(skip)
    )
    if limit is not None:
        page = page.limit(limit)

    return {
        "tables": [
            {
                **{
                    key: value for key, value in dict(row).items() if key != "rel_types"
                },
                "relationship_types": _relationship_types(row["rel_types"]),
            }
            for row in store().query_read(page)
        ],
        "tables_total": total,
    }


def fetch_column_exploration_details(
    column_id: str,
    zone_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Everything one hop off a visible Column.

    Closes the ``Table -> Column -> ColumnAttribute -> Term`` chain from the
    Column's own end: :func:`fetch_table_exploration_details` links a table to
    its terms and :func:`fetch_term_exploration_details` the reverse, and this
    is the same idea one hop in from either. Four independent things come back,
    and a column can have any combination of them:

    ``column_attribute``
        The attribute this column carries, by HAS_ATTRIBUTE *or* SEMANTIC_FK —
        ``relationship_type`` names which, since a foreign-key-shaped column is
        linked the second way. ``None`` for the majority of columns.
    ``foreign_key_column``
        The column its own physical foreign key points at, with that column's
        full catalog path so the client can graft it on as an expandable node
        rather than a bare label.
    ``referencing_columns``
        The same edge read backwards — every visible column whose foreign key
        points *at* this one, which is what makes expanding a primary key show
        anything at all. Capped rather than paged, like every field here.
    ``sql_queries``
        Every visible statement that referenced this column directly.
    """
    resolved = resolve_accessible_catalog_ids(zone_ids)
    table_filter = None if resolved is None else list(resolved["table_ids"])

    # Aliased, and not correlated with whatever `catalog_column` the query
    # around it is already using for the *other* end of the edge -- the plain
    # table here would silently collapse the two into one.
    owner = s.catalog_column.alias("owner_column")
    owned = (
        select(literal(1))
        .select_from(owner)
        .where(
            owner.c.id == column_id,
            literal(True)
            if table_filter is None
            else owner.c.table_id.in_(table_filter),
        )
        .exists()
    )

    link = _attribute_link()
    attr_rows = store().query_read(
        select(
            s.column_attribute.c.id,
            s.column_attribute.c.name,
            s.column_attribute.c.description,
            link.c.rel_type.label("relationship_type"),
        )
        .select_from(
            link.join(
                s.column_attribute,
                and_(
                    s.column_attribute.c.id == link.c.attribute_id,
                    s.column_attribute.c.source == SEMANTIC_SOURCE,
                ),
            )
        )
        .where(link.c.column_id == column_id, owned)
        .limit(1)
    )

    fk_rows = store().query_read(
        select(*_COLUMN_REF_COLUMNS)
        .select_from(
            _column_catalog_join().join(
                s.column_foreign_key,
                s.column_foreign_key.c.target_column_id == s.catalog_column.c.id,
            )
        )
        .where(
            s.column_foreign_key.c.source_column_id == column_id,
            owned,
            _visible_table(table_filter),
        )
        .limit(1)
    )
    referencing_rows = store().query_read(
        select(*_COLUMN_REF_COLUMNS)
        .select_from(
            _column_catalog_join().join(
                s.column_foreign_key,
                s.column_foreign_key.c.source_column_id == s.catalog_column.c.id,
            )
        )
        .where(
            s.column_foreign_key.c.target_column_id == column_id,
            owned,
            _visible_table(table_filter),
        )
        .distinct()
        .order_by(
            s.catalog_table.c.name, s.catalog_column.c.name, s.catalog_column.c.id
        )
        .limit(_MAX_REFERENCING_COLUMNS)
    )

    sql_rows = store().query_read(
        select(s.sql_query.c.id, s.sql_query.c.sql_full_query.label("sql"))
        .select_from(
            s.sql_query_column.join(
                s.sql_query, s.sql_query.c.id == s.sql_query_column.c.sql_query_id
            )
        )
        .where(
            s.sql_query_column.c.column_id == column_id,
            owned,
            _sql_in_scope(table_filter),
        )
        .distinct()
        .order_by(s.sql_query.c.id)
    )

    return {
        "column_attribute": dict(attr_rows[0]) if attr_rows else None,
        "foreign_key_column": dict(fk_rows[0]) if fk_rows else None,
        "referencing_columns": [dict(row) for row in referencing_rows if row["id"]],
        "sql_queries": [dict(row) for row in sql_rows if row["id"] and row["sql"]],
    }


def fetch_column_attribute_exploration_details(
    attr_id: str,
    zone_ids: list[str] | None = None,
    *,
    skip: int = 0,
    limit: int | None = None,
) -> dict[str, Any]:
    """The Term owning a ColumnAttribute, and one page of every Column carrying it.

    Expanding an attribute node should reconnect it to its owning term *and* to
    every column that shares it — not just the one column whichever expansion
    grafted the attribute on already knew about. A common attribute (a
    ``user_id``-shaped one) is typically linked from many columns across many
    tables at once.

    ``term`` is ``None`` when the attribute has no owning Term, or that term
    falls outside *zone_ids*.
    """
    resolved = resolve_accessible_catalog_ids(zone_ids)
    term_rows = store().query_read(
        select(s.term.c.id, s.term.c.name, s.term.c.description)
        .select_from(
            s.column_attribute_term.join(
                s.term, s.term.c.id == s.column_attribute_term.c.term_id
            )
        )
        .where(s.column_attribute_term.c.attribute_id == attr_id)
        .limit(1)
    )
    term = (
        dict(term_rows[0])
        if term_rows and term_is_in_scope(term_rows[0]["id"], zone_ids, resolved)
        else None
    )

    link = _attribute_link()
    conditions: list[Any] = [link.c.attribute_id == attr_id]
    if resolved is not None:
        conditions.append(s.catalog_column.c.table_id.in_(list(resolved["table_ids"])))

    linked_columns = _column_catalog_join().join(
        link, link.c.column_id == s.catalog_column.c.id
    )
    total_rows = store().query_read(
        select(func.count(func.distinct(s.catalog_column.c.id)).label("total"))
        .select_from(linked_columns)
        .where(*conditions)
    )
    total = int(total_rows[0]["total"]) if total_rows else 0
    if not total:
        return {"term": term, "columns": [], "columns_total": 0}

    page = (
        select(*_COLUMN_REF_COLUMNS, func.array_agg(link.c.rel_type).label("rel_types"))
        .select_from(linked_columns)
        .where(*conditions)
        .group_by(*_COLUMN_REF_COLUMNS)
        .order_by(
            s.catalog_table.c.name, s.catalog_column.c.name, s.catalog_column.c.id
        )
        .offset(skip)
    )
    if limit is not None:
        page = page.limit(limit)

    return {
        "term": term,
        "columns": [
            {
                **{
                    key: value for key, value in dict(row).items() if key != "rel_types"
                },
                "relationship_types": _relationship_types(row["rel_types"]),
            }
            for row in store().query_read(page)
        ],
        "columns_total": total,
    }


def fetch_sql_attribute_exploration_details(
    attr_id: str,
    zone_ids: list[str] | None = None,
) -> dict[str, Any]:
    """The statement and owning Term behind a SqlAttribute.

    A term's own expansion already grafts its SqlAttributes on; this is what
    that attribute in turn connects to, so double-clicking it can graft both
    ends. ``sql`` is ``None`` when any table the statement references falls
    outside *zone_ids*, even where the attribute and its term are both visible —
    the same rule ``_sql_attr_zone_filter`` applies in ``gsf.dal.sql_attributes``.
    """
    resolved = resolve_accessible_catalog_ids(zone_ids)
    table_filter = None if resolved is None else list(resolved["table_ids"])

    sql_rows = store().query_read(
        select(s.sql_query.c.id, s.sql_query.c.sql_full_query.label("sql"))
        .select_from(
            s.sql_attribute_sql.join(
                s.sql_query, s.sql_query.c.id == s.sql_attribute_sql.c.sql_query_id
            )
        )
        .where(
            s.sql_attribute_sql.c.attribute_id == attr_id,
            _sql_in_scope(table_filter),
        )
        .order_by(s.sql_query.c.id)
        .limit(1)
    )
    term_rows = store().query_read(
        select(s.term.c.id, s.term.c.name, s.term.c.description)
        .select_from(
            s.sql_attribute_term.join(
                s.term, s.term.c.id == s.sql_attribute_term.c.term_id
            )
        )
        .where(s.sql_attribute_term.c.attribute_id == attr_id)
        .limit(1)
    )
    return {
        "sql": dict(sql_rows[0]) if sql_rows else None,
        "term": dict(term_rows[0]) if term_rows else None,
    }


def fetch_sql_exploration_details(
    sql_id: str,
    zone_ids: list[str] | None = None,
) -> dict[str, Any]:
    """The custom analyses, columns, tables and SqlAttributes hanging off a statement.

    A SqlAttribute's statement is identified by its **text**, so a custom
    analysis saved from that same text shares the row rather than getting its
    own — which is why most statements have no custom analysis to show here.
    The columns and tables are the statement's own references, enriched with
    their catalog path so a client can graft either on as a fully expandable
    node. The SqlAttributes are the reverse of
    :func:`fetch_sql_attribute_exploration_details`'s own lookup; one whose
    owning term falls outside *zone_ids* is dropped.
    """
    resolved = resolve_accessible_catalog_ids(zone_ids)
    table_filter = None if resolved is None else list(resolved["table_ids"])
    visible = _visible_table(table_filter)

    analyses = store().query_read(
        select(
            s.custom_analysis.c.id,
            s.custom_analysis.c.name,
            s.custom_analysis.c.description,
        )
        .select_from(
            s.custom_analysis_sql.join(
                s.custom_analysis,
                s.custom_analysis.c.id == s.custom_analysis_sql.c.analysis_id,
            ).join(
                s.sql_query, s.sql_query.c.id == s.custom_analysis_sql.c.sql_query_id
            )
        )
        .where(
            s.custom_analysis_sql.c.sql_query_id == sql_id,
            _sql_in_scope(table_filter),
        )
        .distinct()
        .order_by(s.custom_analysis.c.name)
    )

    columns = store().query_read(
        select(*_COLUMN_REF_COLUMNS)
        .select_from(
            _column_catalog_join().join(
                s.sql_query_column,
                s.sql_query_column.c.column_id == s.catalog_column.c.id,
            )
        )
        .where(s.sql_query_column.c.sql_query_id == sql_id, visible)
        .distinct()
        .order_by(
            s.catalog_table.c.name, s.catalog_column.c.name, s.catalog_column.c.id
        )
    )

    tables = store().query_read(
        select(*_TABLE_REF_COLUMNS)
        .select_from(
            _table_catalog_join().join(
                s.sql_query_table,
                s.sql_query_table.c.table_id == s.catalog_table.c.id,
            )
        )
        .where(s.sql_query_table.c.sql_query_id == sql_id, visible)
        .distinct()
        .order_by(s.catalog_table.c.name, s.catalog_table.c.id)
    )

    term = s.term.alias("attr_term")
    attributes = store().query_read(
        select(
            s.sql_attribute.c.id,
            s.sql_attribute.c.name,
            s.sql_attribute.c.description,
            term.c.id.label("term_id"),
            term.c.name.label("term_name"),
        )
        .select_from(
            s.sql_attribute_sql.join(
                s.sql_attribute,
                s.sql_attribute.c.id == s.sql_attribute_sql.c.attribute_id,
            )
            .outerjoin(
                s.sql_attribute_term,
                s.sql_attribute_term.c.attribute_id == s.sql_attribute.c.id,
            )
            .outerjoin(term, term.c.id == s.sql_attribute_term.c.term_id)
        )
        .where(s.sql_attribute_sql.c.sql_query_id == sql_id)
        .distinct()
        .order_by(s.sql_attribute.c.name)
    )

    return {
        "custom_analyses": [dict(row) for row in analyses if row["id"]],
        "columns": [dict(row) for row in columns if row["id"]],
        "tables": [dict(row) for row in tables if row["id"]],
        "sql_attributes": [
            dict(row)
            for row in attributes
            if row["id"]
            and (
                row["term_id"] is None
                or term_is_in_scope(row["term_id"], zone_ids, resolved)
            )
        ],
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

    Each link carries ``relationship_types`` -- the link type(s) by which either
    term reaches a table they share -- so a client can label a connection the
    way it reads in the graph.

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

    pairs = fetch_term_table_pairs(zone_ids, data_ids_by_zone=resolved)
    term_tables, table_terms = build_term_table_maps(pairs)
    # Which link type connected each (term, table) pair, kept per pair rather
    # than folded into the maps above, so a term-to-term edge can report *why*
    # each of its two ends reaches the table they share.
    paths_by_pair: dict[tuple[str, str], set[str]] = {}
    for row in pairs:
        term_id, table_id, path = (
            row.get("term_id"),
            row.get("table_id"),
            row.get("path"),
        )
        if term_id and table_id and path:
            paths_by_pair.setdefault((term_id, table_id), set()).add(path)

    visible = {term["id"] for term in terms}
    degree: dict[str, int] = {}
    link_keys: set[tuple[str, str]] = set()
    link_types: dict[tuple[str, str], set[str]] = {}
    for term_id, tables in term_tables.items():
        related: set[str] = set()
        for table_id in tables:
            related.update(table_terms.get(table_id, set()))
        related.discard(term_id)
        degree[term_id] = len(related)
        if term_id not in visible:
            continue
        for other_id in related:
            if other_id not in visible:
                continue
            key = tuple(sorted((term_id, other_id)))
            link_keys.add(key)
            types = link_types.setdefault(key, set())
            for table_id in tables & term_tables.get(other_id, set()):
                types.update(paths_by_pair.get((term_id, table_id), ()))
                types.update(paths_by_pair.get((other_id, table_id), ()))

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
            {
                "source": source,
                "target": target,
                "relationship_types": sorted(link_types.get((source, target), set())),
            }
            for source, target in sorted(link_keys)
            if source in kept and target in kept
        ],
    }


#: A path node's label → the lowerCamelCase ``type`` the client's own node kinds
#: use (``NodeType`` in ``GraphCanvas.tsx``), so a grafted path node styles and
#: expands like any other node of that type without a lookup of its own.
_LINK_PATH_NODE_TYPE = {
    LABEL_TERM: "term",
    Labels.TABLE: "table",
    Labels.COLUMN: "column",
    LABEL_COLUMN_ATTRIBUTE: "columnAttribute",
}


def _link_path_node(node: dict[str, Any]) -> dict[str, Any]:
    """One :func:`find_term_link_path` node, in the API's node-ref shape."""
    label = node.get("label")
    result = {
        "id": node.get("id"),
        "name": node.get("name"),
        "type": _LINK_PATH_NODE_TYPE.get(label, (label or "").lower()),
    }
    if label in (Labels.TABLE, Labels.COLUMN):
        # Only Table/Column nodes carry a catalog path — it is what lets a
        # client expand either one further instead of the node being a dead end
        # just because it arrived as part of a path.
        result["database_id"] = node.get("database_id")
        result["database_name"] = node.get("database_name")
        result["schema_id"] = node.get("schema_id")
        result["schema_name"] = node.get("schema_name")
    if label == Labels.COLUMN:
        result["table_id"] = node.get("table_id")
        result["table_name"] = node.get("table_name")
    return result


def fetch_semantic_link_path(
    source_term_id: str,
    target_term_id: str,
    zone_ids: list[str] | None = None,
) -> dict[str, Any]:
    """The real hop chain behind one term↔term edge of the semantic graph.

    :func:`fetch_semantic_exploration_graph` collapses every shared table, and
    every path reaching it from either term, into one ``relationship_types``
    label — accurate, but unable to show *which* table, column or attribute
    actually connects the two. This returns
    :func:`gsf.dal.terms.find_term_link_path`'s ordered node chain instead, so
    the client can graft those nodes on and highlight the real path.

    Every Table and Column the path passes through has to fall inside
    *zone_ids*, or the whole path is dropped: a chain that silently skipped an
    out-of-scope hop would misrepresent how the two terms connect. Columns are
    checked as well as tables because the traversal walks SEMANTIC_FK and
    HAS_ATTRIBUTE undirected — a path can reach a column from an attribute on
    both sides without ever stepping through that column's own table.
    """
    hops = find_term_link_path(source_term_id, target_term_id)
    if not hops:
        return {"hops": []}

    resolved = resolve_accessible_catalog_ids(zone_ids)
    if resolved is not None:
        table_ids = resolved["table_ids"]
        for hop in hops:
            for side in (hop["source"], hop["target"]):
                label = side.get("label")
                if label == Labels.TABLE and side.get("id") not in table_ids:
                    return {"hops": []}
                # `table_id` is already on every Column node, put there by
                # `find_term_link_path`'s own catalog enrichment — no extra
                # round trip needed to check it.
                if label == Labels.COLUMN and side.get("table_id") not in table_ids:
                    return {"hops": []}

    return {
        "hops": [
            {
                "relationship": hop["relationship"],
                "source": _link_path_node(hop["source"]),
                "target": _link_path_node(hop["target"]),
            }
            for hop in hops
        ]
    }


#: Spelled out because ``fetch_table_zones_map`` is **re-exported** from
#: ``zones`` rather than defined here, and the surface freeze resolves a
#: module's contract by ``__module__`` — which for a re-export points at the
#: definition, not at this file.
__all__ = [
    "MAX_EXPLORATION_GRAPH_NODES",
    "fetch_column_attribute_exploration_details",
    "fetch_column_exploration_details",
    "fetch_data_exploration_edges",
    "fetch_data_exploration_graph",
    "fetch_exploration_related_nodes",
    "fetch_semantic_exploration_graph",
    "fetch_semantic_link_path",
    "fetch_sql_attribute_exploration_details",
    "fetch_sql_exploration_details",
    "fetch_table_exploration_details",
    "fetch_table_zones_map",
    "fetch_term_exploration_details",
]
