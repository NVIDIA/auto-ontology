# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Catalog reads and writes: databases, schemas, tables, columns.

All 26 functions. Seven of them reach the semantic tier and landed a phase later
than the rest ([DECISION-009]) — nothing wrote terms or attributes until Phase 7,
and against no data a wrong join is indistinguishable from a correct one. They
are grouped at the bottom of the file, because that is where the joins get
interesting, not because they are still pending.

Re-check that split with ``python -m dev_tools.classify_dal_dependencies``
rather than by eye; three hand-analyses got it wrong.

Two shapes recur and are easy to lose in translation:

* **A database with no schemas, or a schema with no tables, does not appear.**
  The Cypher's ``MATCH (db)-[:CONTAINS]->(s)`` is an inner join, so an empty
  database is invisible rather than present-with-zero. Reproduced with inner
  joins for the same reason.
* **Zone scoping filters the counted rows, not just the returned ones**, so a
  scoped user sees a schema count covering only the tables they can see.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pandas as pd
from sqlalchemy import ColumnElement, and_, case, func, literal, select, update

from gsf.dal import schema as s
from gsf.dal.session import store
from gsf.dal.sql_fragments import column_description_expr, table_description_expr
from gsf.dal.users import resolve_accessible_catalog_ids
from gsf.semantic.constants import SQL_ATTR_SOURCE_BRIDGE
from gsf.utils.sample_values import parse_sample_values

logger = logging.getLogger(__name__)

#: Labels ``fetch_node_properties_by_id`` will look up, and their tables.
_NODE_TABLES = {
    "Database": s.catalog_database,
    "Schema": s.catalog_schema,
    "Table": s.catalog_table,
    "Column": s.catalog_column,
}


def _catalog_join():
    """Column → Table → Schema → Database."""
    return (
        s.catalog_column.join(
            s.catalog_table, s.catalog_column.c.table_id == s.catalog_table.c.id
        )
        .join(s.catalog_schema, s.catalog_table.c.schema_id == s.catalog_schema.c.id)
        .join(
            s.catalog_database,
            s.catalog_schema.c.database_id == s.catalog_database.c.id,
        )
    )


def _table_join():
    """Table → Schema → Database."""
    return s.catalog_table.join(
        s.catalog_schema, s.catalog_table.c.schema_id == s.catalog_schema.c.id
    ).join(
        s.catalog_database, s.catalog_schema.c.database_id == s.catalog_database.c.id
    )


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------


def fetch_databases(zone_ids: list[str] | None = None) -> list[dict[str, Any]]:
    """Databases with a schema count. ``schemas`` is empty — the tree is lazy."""
    scoped = resolve_accessible_catalog_ids(zone_ids)

    statement = (
        select(
            s.catalog_database.c.id,
            s.catalog_database.c.name,
            func.count(s.catalog_schema.c.id).label("schema_count"),
        )
        .select_from(
            s.catalog_database.join(
                s.catalog_schema,
                s.catalog_schema.c.database_id == s.catalog_database.c.id,
            )
        )
        .group_by(s.catalog_database.c.id, s.catalog_database.c.name)
        .order_by(s.catalog_database.c.name)
    )
    if scoped is not None:
        statement = statement.where(
            and_(
                s.catalog_database.c.id.in_(list(scoped["db_ids"])),
                s.catalog_schema.c.id.in_(list(scoped["schema_ids"])),
            )
        )

    return [
        {
            "id": r["id"],
            "name": r["name"],
            # The Cypher read `db.description`, a property nothing writes, so
            # this was always None. Kept in the shape because callers read the
            # key; there is no column to read it from.
            "description": None,
            "num_of_schemas": int(r["schema_count"]),
            "schemas": [],
        }
        for r in store().query_read(statement)
    ]


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def fetch_schemas_for_database(
    db_id: str,
    zone_ids: list[str] | None = None,
) -> dict[str, Any] | None:
    """``{schemas_count, schemas}`` for a database, or ``None`` if it has none.

    ``None`` rather than an empty result is the Cypher's behaviour: its match
    required at least one table, so a database with no tables produced no rows
    at all and the caller was expected to treat that as absent.
    """
    scoped = resolve_accessible_catalog_ids(zone_ids)

    statement = (
        select(
            s.catalog_schema.c.id,
            s.catalog_schema.c.name.label("schema_name"),
            func.count(s.catalog_table.c.id).label("tables_count"),
        )
        .select_from(
            s.catalog_schema.join(
                s.catalog_table, s.catalog_table.c.schema_id == s.catalog_schema.c.id
            )
        )
        .where(s.catalog_schema.c.database_id == db_id)
        .group_by(s.catalog_schema.c.id, s.catalog_schema.c.name)
        .order_by(s.catalog_schema.c.name)
    )
    if scoped is not None:
        statement = statement.where(
            and_(
                s.catalog_schema.c.id.in_(list(scoped["schema_ids"])),
                s.catalog_table.c.id.in_(list(scoped["table_ids"])),
            )
        )

    rows = store().query_read(statement)
    if not rows:
        return None
    return {
        "schemas_count": len(rows),
        "schemas": [
            {
                "id": r["id"],
                "schema_name": r["schema_name"],
                "description": None,
                "tables_count": int(r["tables_count"]),
            }
            for r in rows
        ],
    }


def fetch_all_schema_ids() -> list[str]:
    return [r["id"] for r in store().query_read(select(s.catalog_schema.c.id))]


def fetch_schema_ids_for_database(database_name: str) -> list[str]:
    """Scopes a catalog build to one database.

    Without it, schema-name collisions — several SQLite databases all calling
    theirs ``main`` — overwrite each other in the assembled schema map.
    """
    return [
        r["id"]
        for r in store().query_read(
            select(s.catalog_schema.c.id)
            .select_from(
                s.catalog_schema.join(
                    s.catalog_database,
                    s.catalog_schema.c.database_id == s.catalog_database.c.id,
                )
            )
            .where(s.catalog_database.c.name == database_name)
        )
    ]


def fetch_schemas_by_ids(
    relevant_schemas_ids: list | None = None,
) -> list[dict[str, str]]:
    """Flat column rows for the catalog map SQL validation is built from.

    An empty or absent id list means *every* schema, not none — the Cypher's
    ``WHERE size($schema_ids) = 0 OR ...``. Reading that as "none" would leave
    every query unresolvable rather than raising.
    """
    schema_ids = relevant_schemas_ids or []

    statement = select(
        s.catalog_column.c.name.label("column_name"),
        s.catalog_column.c.id.label("column_id"),
        s.catalog_table.c.name.label("table_name"),
        s.catalog_table.c.id.label("table_id"),
        s.catalog_database.c.name.label("database_name"),
        s.catalog_schema.c.name.label("table_schema"),
        s.catalog_column.c.data_type,
    ).select_from(_catalog_join())

    if schema_ids:
        statement = statement.where(s.catalog_schema.c.id.in_(list(schema_ids)))

    return [dict(r) for r in store().query_read(statement)]


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------


def _table_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "name": row["name"],
        "schema_name": row["schema_name"],
        "description": row["description"],
        "pk": row["pk"],
    }


def _table_select():
    return select(
        s.catalog_table.c.id,
        s.catalog_table.c.name,
        s.catalog_schema.c.name.label("schema_name"),
        s.catalog_table.c.description,
        s.catalog_table.c.pk,
    ).select_from(
        s.catalog_table.join(
            s.catalog_schema, s.catalog_table.c.schema_id == s.catalog_schema.c.id
        )
    )


def fetch_sorted_tables() -> list[dict[str, Any]]:
    """Every table, busiest first.

    ``name`` breaks ties. The Cypher ordered by ``query_count DESC`` alone, and
    nearly every table has a count of zero — so a function whose name promises
    an order returned rows in whatever order the store felt like, differing
    between calls. Adding a tiebreaker cannot break a caller that was already
    receiving an arbitrary order, and makes the result reproducible.
    """
    query_count = (
        select(func.count(s.sql_query_table.c.sql_query_id))
        .where(s.sql_query_table.c.table_id == s.catalog_table.c.id)
        .scalar_subquery()
        .label("query_count")
    )
    rows = store().query_read(
        select(
            s.catalog_table.c.id,
            s.catalog_table.c.name,
            s.catalog_schema.c.name.label("schema_name"),
            s.catalog_table.c.description,
            s.catalog_table.c.pk,
            query_count,
        )
        .select_from(
            s.catalog_table.join(
                s.catalog_schema, s.catalog_table.c.schema_id == s.catalog_schema.c.id
            )
        )
        .order_by(query_count.desc(), s.catalog_table.c.name)
    )
    return [
        {
            "id": r["id"],
            "name": r["name"],
            "schema_name": r["schema_name"],
            "description": r["description"] or "",
            "query_count": int(r["query_count"] or 0),
            "pk": r["pk"] or [],
        }
        for r in rows
    ]


def fetch_table_by_id(table_id: str) -> dict[str, Any] | None:
    rows = store().query_read(_table_select().where(s.catalog_table.c.id == table_id))
    return _table_row(rows[0]) if rows else None


def fetch_table_by_name(name: str) -> dict[str, Any] | None:
    """The first table of that name, in *any* schema or database.

    Ambiguous by construction — the Cypher took ``LIMIT 1`` with no ordering.
    Ordered by id here so repeated calls agree with each other; which row wins
    is still arbitrary, but it is at least the same arbitrary row.
    """
    rows = store().query_read(
        _table_select()
        .where(s.catalog_table.c.name == name)
        .order_by(s.catalog_table.c.id)
        .limit(1)
    )
    return _table_row(rows[0]) if rows else None


def fetch_join_neighbors(table_id: str) -> list[dict[str, Any]]:
    """Tables joined to this one, in either direction.

    The Cypher pattern was undirected (``-[:JOIN]-``), so both ends of the
    relationship count; here that is a union of the two columns.
    """
    other = s.catalog_table.alias("other")
    outgoing = (
        select(other.c.id, other.c.name, other.c.description)
        .select_from(
            s.table_join.join(other, other.c.id == s.table_join.c.target_table_id)
        )
        .where(s.table_join.c.source_table_id == table_id)
    )
    incoming = (
        select(other.c.id, other.c.name, other.c.description)
        .select_from(
            s.table_join.join(other, other.c.id == s.table_join.c.source_table_id)
        )
        .where(s.table_join.c.target_table_id == table_id)
    )
    return [dict(r) for r in store().query_read(outgoing.union(incoming))]


def fetch_join_edges() -> list[dict[str, Any]]:
    source = s.catalog_table.alias("source_table")
    target = s.catalog_table.alias("target_table")
    return [
        dict(r)
        for r in store().query_read(
            select(
                source.c.name.label("source_table"),
                source.c.id.label("source_table_id"),
                target.c.name.label("target_table"),
                target.c.id.label("target_table_id"),
                s.table_join.c.join_columns,
            ).select_from(
                s.table_join.join(
                    source, source.c.id == s.table_join.c.source_table_id
                ).join(target, target.c.id == s.table_join.c.target_table_id)
            )
        )
    ]


# ---------------------------------------------------------------------------
# Column
# ---------------------------------------------------------------------------


def count_columns_for_table(table_id: str) -> int:
    rows = store().query_read(
        select(func.count(s.catalog_column.c.id).label("total")).where(
            s.catalog_column.c.table_id == table_id
        )
    )
    return int(rows[0]["total"]) if rows else 0


def fetch_parent_table_id_for_column(column_id: str) -> str | None:
    rows = store().query_read(
        select(s.catalog_column.c.table_id).where(s.catalog_column.c.id == column_id)
    )
    return rows[0]["table_id"] if rows else None


def fetch_col_table_contexts(col_ids: list[str]) -> dict[str, dict[str, str]]:
    """Column id → its database/schema/table names.

    Returns ``{}`` on failure rather than raising, matching the Cypher: callers
    use this to decorate results, and losing the decoration is better than
    losing the result.
    """
    if not col_ids:
        return {}
    try:
        rows = store().query_read(
            select(
                s.catalog_column.c.id.label("col_id"),
                s.catalog_table.c.name.label("table_name"),
                s.catalog_schema.c.name.label("schema_name"),
                s.catalog_database.c.name.label("database_name"),
            )
            .select_from(_catalog_join())
            .where(s.catalog_column.c.id.in_(list(col_ids)))
        )
    except Exception:
        logger.warning("fetch_col_table_contexts: query failed", exc_info=True)
        return {}

    return {
        r["col_id"]: {
            "table_name": r.get("table_name") or "",
            "schema_name": r.get("schema_name") or "",
            "database_name": r.get("database_name") or "",
        }
        for r in rows
        if r.get("col_id")
    }


def _set_column_property(table_id: str, values: dict[str, Any], column: str) -> None:
    """Set one property across several of a table's columns, in one statement.

    ``CASE name WHEN 'a' THEN … END`` rather than a statement per column: the
    Cypher did it in one write, and a table can have hundreds of columns.
    """
    if not values:
        return

    target = s.catalog_column.c[column]
    store().query_write(
        update(s.catalog_column)
        .where(
            and_(
                s.catalog_column.c.table_id == table_id,
                s.catalog_column.c.name.in_(list(values)),
            )
        )
        .values(
            **{
                column: case(
                    {
                        name: literal(value, target.type)
                        for name, value in values.items()
                    },
                    value=s.catalog_column.c.name,
                    else_=target,
                )
            }
        )
    )


def store_column_sample_values(table_id: str, samples: dict[str, list]) -> None:
    """Write sample values, JSON-encoded, onto a table's columns."""
    if not samples:
        return
    _set_column_property(
        table_id,
        {name: json.dumps(values) for name, values in samples.items()},
        "sample_values",
    )


def store_column_uniqueness(table_id: str, uniqueness: dict[str, bool]) -> None:
    """Write ``is_unique`` flags onto a table's columns."""
    if not uniqueness:
        return
    _set_column_property(
        table_id,
        {name: bool(flag) for name, flag in uniqueness.items()},
        "is_unique",
    )


# ---------------------------------------------------------------------------
# Metadata writes
# ---------------------------------------------------------------------------


def apply_metadata_batch(
    database_name: str,
    table_rows: list[dict],
    column_rows: list[dict],
) -> None:
    """Write descriptions and sample values, **without overwriting existing ones**.

    ``coalesce(new, existing)`` in the Cypher, and the direction matters: a
    curated description survives a batch that has nothing to say about it.
    """
    for row in table_rows or []:
        store().query_write(
            update(s.catalog_table)
            .where(
                and_(
                    s.catalog_table.c.name == row["table_name"],
                    s.catalog_table.c.schema_id.in_(
                        select(s.catalog_schema.c.id)
                        .select_from(
                            s.catalog_schema.join(
                                s.catalog_database,
                                s.catalog_schema.c.database_id
                                == s.catalog_database.c.id,
                            )
                        )
                        .where(s.catalog_database.c.name == database_name)
                    ),
                )
            )
            .values(
                description=func.coalesce(
                    row.get("description"), s.catalog_table.c.description
                )
            )
        )

    for row in column_rows or []:
        store().query_write(
            update(s.catalog_column)
            .where(
                and_(
                    s.catalog_column.c.name == row["column_name"],
                    s.catalog_column.c.table_id.in_(
                        select(s.catalog_table.c.id)
                        .select_from(_table_join())
                        .where(
                            and_(
                                s.catalog_database.c.name == database_name,
                                s.catalog_table.c.name == row["table_name"],
                            )
                        )
                    ),
                )
            )
            .values(
                description=func.coalesce(
                    row.get("description"), s.catalog_column.c.description
                ),
                sample_values=func.coalesce(
                    row.get("sample_values"), s.catalog_column.c.sample_values
                ),
            )
        )


# ---------------------------------------------------------------------------
# Any catalog node
# ---------------------------------------------------------------------------


def patch_catalog_node(
    node_id: str,
    properties: dict[str, Any],
) -> dict[str, Any] | None:
    """Write properties onto whichever catalog row carries *node_id*.

    The Cypher matched four labels at once and let the id pick the node. Each is
    a separate table here, so they are tried in turn. Properties with no column
    are dropped, as the graph accepted anything and refusing would fail writes
    that work today.
    """
    for label, table in _NODE_TABLES.items():
        columns = {c.name for c in table.columns}
        values = {k: v for k, v in properties.items() if k in columns and k != "id"}
        rows = store().query_read(select(table.c.id).where(table.c.id == node_id))
        if not rows:
            continue
        if values:
            store().query_write(
                update(table).where(table.c.id == node_id).values(**values)
            )
        props = dict(store().query_read(select(table).where(table.c.id == node_id))[0])
        return {"id": node_id, "label": label, "props": props}
    return None


def fetch_node_properties_by_id(id: str, label: str | list[str]) -> dict | None:
    """All of a node's properties plus its label, or ``None``.

    Unknown labels are rejected with a warning rather than an exception, which
    is what the Cypher did — the label often arrives from a URL.
    """
    labels = label if isinstance(label, list) else [label]
    for candidate in labels:
        if candidate not in _NODE_TABLES:
            logger.warning(
                "Rejecting unknown label %r in fetch_node_properties_by_id", candidate
            )
            return None

    for candidate in labels:
        table = _NODE_TABLES[candidate]
        rows = store().query_read(select(table).where(table.c.id == id))
        if rows:
            props = dict(rows[0])
            props["label"] = candidate
            return props
    return None


def fetch_item_by_id(item_id: str, label: str | list[str]) -> dict | None:
    """As :func:`fetch_node_properties_by_id`, but logs when nothing matches."""
    result = fetch_node_properties_by_id(item_id, label)
    if result is None:
        logger.error("Required item with id %r not found.", item_id)
    return result


# ---------------------------------------------------------------------------
# The seven that reach the semantic tier (DECISION-009)
# ---------------------------------------------------------------------------


def _terms_count(table_id: ColumnElement) -> ColumnElement:
    """How many distinct Terms a table is associated with.

    Two routes, and the Cypher counted the deduplicated union of both:

    * **directly** — ``REPRESENTS``, the table *is* that business concept;
    * **through its columns** — a column carries a ColumnAttribute (by
      ``HAS_ATTRIBUTE`` or ``SEMANTIC_FK``) and that attribute is a property of
      a Term.

    Counted as "Terms reachable by any route" rather than as a ``UNION`` of the
    three id lists. The union reads more naturally and does not work: wrapping
    it in ``.subquery()`` to count it puts two levels between the leg predicates
    and ``catalog_table``, and SQLAlchemy stops correlating.

    The ``.correlate()`` calls below are load-bearing for the same reason.
    SQLAlchemy auto-correlates a table only against the *immediately* enclosing
    SELECT, and that one selects from ``term`` alone — so left to itself it adds
    a second, unconstrained ``catalog_table`` to each ``EXISTS`` and the count
    stops depending on which table is being counted. Every row then reports the
    same total, which on a fixture where the numbers happen to agree looks
    entirely correct. That is why the test asserts a table with *no* terms
    alongside one with two.
    """
    owner = table_id.table

    def via(link_table):
        return (
            select(literal(1))
            .select_from(
                s.catalog_column.join(
                    link_table, link_table.c.column_id == s.catalog_column.c.id
                ).join(
                    s.column_attribute_term,
                    s.column_attribute_term.c.attribute_id == link_table.c.attribute_id,
                )
            )
            .where(
                s.catalog_column.c.table_id == table_id,
                s.column_attribute_term.c.term_id == s.term.c.id,
            )
            .correlate(owner, s.term)
            .exists()
        )

    direct = (
        select(literal(1))
        .where(
            s.table_term.c.table_id == table_id,
            s.table_term.c.term_id == s.term.c.id,
        )
        .correlate(owner, s.term)
        .exists()
    )
    return (
        select(func.count())
        .select_from(s.term)
        .where(direct | via(s.column_has_attribute) | via(s.column_semantic_fk))
        .correlate(owner)
        .scalar_subquery()
    )


def _count_of(table, predicate) -> ColumnElement:
    return select(func.count()).select_from(table).where(predicate).scalar_subquery()


def fetch_tables_for_schema(
    schema_id: str,
    *,
    database_name: str | None = None,
    zone_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Tables in a schema, with column, SQL, and Term counts.

    *database_name* is accepted and ignored — schema ids are globally unique, so
    it can only ever agree with *schema_id* or contradict it. It stays in the
    signature because callers pass it.
    """
    scoped = resolve_accessible_catalog_ids(zone_ids)

    statement = (
        select(
            s.catalog_table.c.id,
            s.catalog_table.c.name,
            s.catalog_table.c.table_type,
            s.catalog_database.c.name.label("database_name"),
            s.catalog_schema.c.name.label("schema_name"),
            table_description_expr().label("description"),
            s.catalog_table.c.description_certified,
            _count_of(
                s.catalog_column, s.catalog_column.c.table_id == s.catalog_table.c.id
            ).label("columns_count"),
            _count_of(
                s.sql_query_table, s.sql_query_table.c.table_id == s.catalog_table.c.id
            ).label("sql_count"),
            _terms_count(s.catalog_table.c.id).label("terms_count"),
        )
        .select_from(_table_join())
        .where(s.catalog_table.c.schema_id == schema_id)
        .order_by(s.catalog_table.c.name)
    )
    if scoped is not None:
        statement = statement.where(s.catalog_table.c.id.in_(list(scoped["table_ids"])))

    return [dict(r) for r in store().query_read(statement)]


def fetch_all_tables_without_term(
    database_name: str | None = None,
) -> list[dict[str, Any]]:
    """Tables not yet assigned a Term — the work list for term compilation.

    Scoped to one database when *database_name* is given. Several databases can
    share a store (the BIRD benchmark puts dozens in one), and each compile pass
    tags its embeddings with a database name, so an unscoped pass would attribute
    one database's tables to another.

    Note this reads the table's *own* description, not the fallback: a table
    with no Term has no Term description to fall back to.
    """
    statement = (
        select(
            s.catalog_table.c.id,
            s.catalog_table.c.name,
            s.catalog_table.c.description,
            s.catalog_schema.c.name.label("schema_name"),
        )
        .select_from(_table_join())
        .where(
            ~select(s.table_term.c.term_id)
            .where(s.table_term.c.table_id == s.catalog_table.c.id)
            .exists()
        )
        .order_by(s.catalog_table.c.name)
    )
    if database_name is not None:
        statement = statement.where(s.catalog_database.c.name == database_name)

    return [dict(r) for r in store().query_read(statement)]


def fetch_columns_for_table(
    table_id: str,
    *,
    skip: int = 0,
    limit: int | None = None,
) -> dict[str, Any] | None:
    """A table with its columns nested, or ``None`` if the table is missing.

    Columns come back in ordinal order, so *skip* and *limit* read one page of
    it; pair them with ``count_columns_for_table`` for the total, which no paged
    read can report. Omit *limit* for every column, which is what the catalog
    tree and the text-to-SQL context want.

    **``None`` means the table is missing, never that the page is empty.** The
    Cypher guaranteed that by paging inside a subquery scoped to the table; two
    queries do it here, and the header query is the one that decides. A single
    join with ``OFFSET`` would lose the table's own fields as soon as *skip*
    ran past the last column, turning "page 3 of a 2-page table" into "no such
    table".
    """
    header = store().query_read(
        select(
            s.catalog_table.c.name.label("table_name"),
            s.catalog_table.c.table_type,
            s.catalog_schema.c.name.label("schema_name"),
            s.catalog_database.c.name.label("database_name"),
        )
        .select_from(_table_join())
        .where(s.catalog_table.c.id == table_id)
        .limit(1)
    )
    if not header:
        return None

    page = (
        select(
            s.catalog_column.c.id,
            s.catalog_column.c.ordinal_position,
            s.catalog_column.c.name.label("column_name"),
            s.catalog_column.c.data_type,
            column_description_expr().label("description"),
            s.catalog_column.c.description_certified,
            s.catalog_column.c.sample_values,
        )
        .where(s.catalog_column.c.table_id == table_id)
        # `ordinal_position` is nullable and not unique, so on its own it is not
        # a stable page order -- two columns sharing a position could swap
        # between page 1 and page 2, showing one twice and the other never. `id`
        # breaks the tie.
        .order_by(s.catalog_column.c.ordinal_position, s.catalog_column.c.id)
        .offset(skip)
    )
    if limit is not None:
        page = page.limit(limit)

    table = dict(header[0])
    table["columns"] = [
        {**dict(r), "sample_values": parse_sample_values(r["sample_values"])}
        for r in store().query_read(page)
    ]
    return table


def fetch_tables_by_ids(table_ids: list[str]) -> list[dict[str, Any]]:
    """Tables with a name/type/description summary of each column.

    Returns ``[]`` rather than raising if the query fails, which is what the
    Cypher did: this decorates retrieval results, and losing the decoration
    beats losing the results.

    A table with no columns does not appear at all — the Cypher's second
    ``MATCH`` was an inner join. Preserved rather than fixed; a column-less
    table in the catalog is a symptom worth seeing elsewhere, not something to
    paper over here.
    """
    if not table_ids:
        return []
    try:
        rows = store().query_read(
            select(
                s.catalog_table.c.id,
                s.catalog_table.c.name,
                s.catalog_table.c.description,
                s.catalog_database.c.name.label("database_name"),
                s.catalog_schema.c.name.label("schema_name"),
                s.catalog_column.c.name.label("column_name"),
                s.catalog_column.c.data_type,
                column_description_expr().label("column_description"),
            )
            .select_from(
                _table_join().join(
                    s.catalog_column,
                    s.catalog_column.c.table_id == s.catalog_table.c.id,
                )
            )
            .where(s.catalog_table.c.id.in_(list(table_ids)))
            .order_by(s.catalog_table.c.id, s.catalog_column.c.ordinal_position)
        )
    except Exception:
        logger.warning("fetch_tables_by_ids: query failed", exc_info=True)
        return []

    tables: dict[str, dict[str, Any]] = {}
    for row in rows:
        table = tables.setdefault(
            row["id"],
            {
                "id": row["id"],
                "name": row["name"] or "",
                "description": row["description"] or "",
                "database_name": row["database_name"] or "",
                "schema_name": row["schema_name"] or "",
                "label": "Table",
                "columns": [],
            },
        )
        # The Cypher dropped unnamed columns from the nested list after
        # collecting them. `name` is NOT NULL here, so this cannot fire -- kept
        # so the shape stays identical if that ever changes.
        if row["column_name"]:
            table["columns"].append(
                {
                    "name": row["column_name"],
                    "data_type": row["data_type"],
                    "description": row["column_description"],
                }
            )
    return list(tables.values())


def fetch_table_context(table_id: str) -> dict[str, Any]:
    """``{columns, fks}`` for one table — what the SQL generator is handed."""
    columns = [
        {
            "id": r["id"],
            "name": r["name"],
            "data_type": r["data_type"],
            "description": r["description"],
            "ordinal_position": r["ordinal_position"],
            "sample_values": r["sample_values"],
        }
        for r in store().query_read(
            select(
                s.catalog_column.c.id,
                s.catalog_column.c.name,
                s.catalog_column.c.data_type,
                column_description_expr().label("description"),
                s.catalog_column.c.ordinal_position,
                s.catalog_column.c.sample_values,
            )
            .where(s.catalog_column.c.table_id == table_id)
            .order_by(s.catalog_column.c.ordinal_position, s.catalog_column.c.id)
        )
    ]

    source = s.catalog_column.alias("source_column")
    target = s.catalog_column.alias("target_column")
    target_table = s.catalog_table.alias("target_table")
    fks = [
        dict(r)
        for r in store().query_read(
            select(
                source.c.name.label("source_column"),
                target.c.name.label("target_column"),
                target_table.c.name.label("target_table"),
                target_table.c.id.label("target_table_id"),
            )
            .select_from(
                source.join(
                    s.column_foreign_key,
                    s.column_foreign_key.c.source_column_id == source.c.id,
                )
                .join(target, target.c.id == s.column_foreign_key.c.target_column_id)
                .join(target_table, target_table.c.id == target.c.table_id)
            )
            .where(source.c.table_id == table_id)
        )
    ]
    return {"columns": columns, "fks": fks}


def fetch_tables_and_columns_by_node_ids(
    node_ids: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Table and column frames for ``TabularFetchEmbeddingsOp``.

    *node_ids* mixes table and column ids freely. A table id pulls in all of its
    columns; a column id pulls in only itself. Both frames carry
    ``database_name``, and the third return value is the first one found — the
    caller assumes a single database and there is nothing here that enforces it.
    """
    conn = store()
    ids = list(node_ids)

    columns_df = pd.DataFrame(
        [
            dict(r)
            for r in conn.query_read(
                select(
                    s.catalog_column.c.id,
                    s.catalog_table.c.name.label("table_name"),
                    s.catalog_schema.c.name.label("table_schema"),
                    s.catalog_column.c.name.label("column_name"),
                    s.catalog_column.c.data_type,
                    column_description_expr().label("description"),
                    s.catalog_column.c.sample_values,
                    s.catalog_database.c.name.label("database_name"),
                )
                .select_from(_catalog_join())
                .where(s.catalog_table.c.id.in_(ids) | s.catalog_column.c.id.in_(ids))
                .distinct()
            )
        ]
    )
    tables_df = pd.DataFrame(
        [
            dict(r)
            for r in conn.query_read(
                select(
                    s.catalog_table.c.id,
                    s.catalog_table.c.name.label("table_name"),
                    s.catalog_schema.c.name.label("table_schema"),
                    s.catalog_table.c.table_type,
                    # The table's own description, not the fallback: the Cypher
                    # read `t.description` here even though it used the fallback
                    # for columns three lines earlier. Asymmetric, and preserved
                    # -- these frames feed embeddings, and quietly widening what
                    # gets embedded would change retrieval results without any
                    # test noticing.
                    s.catalog_table.c.description,
                    s.catalog_database.c.name.label("database_name"),
                )
                .select_from(_table_join())
                .where(s.catalog_table.c.id.in_(ids))
            )
        ]
    )

    database_name = ""
    for frame in (tables_df, columns_df):
        if not frame.empty:
            database_name = str(frame.iloc[0].get("database_name") or "")
            break
    return tables_df, columns_df, database_name


def fetch_bridge_table_candidates(database_name: str) -> list[dict[str, Any]]:
    """Pure-FK junction tables eligible for a bridge SqlAttribute.

    A table qualifies when **every** column is a foreign key — by a real
    ``FOREIGN_KEY`` or a ``SEMANTIC_FK`` standing in for one — it has at least
    two of them, none of its columns already carries a ColumnAttribute, and no
    bridge SqlAttribute references it yet. Self-referential bridges count:
    ``also_buy(product_id, also_buy_product_id)`` targets one table twice.

    The Cypher expressed "every column resolves" twice over — once as
    ``ALL(c IN cols WHERE ...)`` and again as ``size(fk_pairs) = size(cols)``.
    The two are not redundant: the first checks each column has an outgoing
    edge, the second that each edge actually lands on a column inside a table
    (an FK pointing at a column whose table was never ingested passes the first
    and fails the second). Both are kept.
    """
    fk_target = s.catalog_column.alias("fk_target")
    fk_table = s.catalog_table.alias("fk_table")
    fk_schema = s.catalog_schema.alias("fk_schema")
    sem_target = s.catalog_column.alias("sem_target")
    sem_table = s.catalog_table.alias("sem_table")
    sem_schema = s.catalog_schema.alias("sem_schema")

    # Column -> the table/schema/column its FK lands on, by either route.
    # `resolved_count` below counts these rows, so a column with no resolvable
    # target contributes nothing and the size comparison fails -- which is the
    # Cypher's second check, the one that catches an FK pointing at a column
    # whose table was never ingested.
    #
    # The SEMANTIC_FK route is Column -> ColumnAttribute <- Column: the
    # attribute this column references is *owned* by some other column, and that
    # other column is the join target. Reversing it would make every bridge
    # point back at itself.
    sem_owner = s.column_has_attribute.alias("sem_owner")
    resolved = (
        select(
            s.catalog_column.c.id.label("column_id"),
            s.catalog_column.c.table_id.label("owner_table_id"),
            s.catalog_column.c.name.label("source_column"),
            func.coalesce(fk_table.c.name, sem_table.c.name).label("target_table"),
            func.coalesce(fk_schema.c.name, sem_schema.c.name).label("target_schema"),
            func.coalesce(fk_target.c.name, sem_target.c.name).label("target_column"),
            func.coalesce(fk_table.c.id, sem_table.c.id).label("target_table_id"),
        )
        .select_from(
            s.catalog_column.outerjoin(
                s.column_foreign_key,
                s.column_foreign_key.c.source_column_id == s.catalog_column.c.id,
            )
            .outerjoin(
                fk_target, fk_target.c.id == s.column_foreign_key.c.target_column_id
            )
            .outerjoin(fk_table, fk_table.c.id == fk_target.c.table_id)
            .outerjoin(fk_schema, fk_schema.c.id == fk_table.c.schema_id)
            .outerjoin(
                s.column_semantic_fk,
                s.column_semantic_fk.c.column_id == s.catalog_column.c.id,
            )
            .outerjoin(
                sem_owner,
                sem_owner.c.attribute_id == s.column_semantic_fk.c.attribute_id,
            )
            .outerjoin(sem_target, sem_target.c.id == sem_owner.c.column_id)
            .outerjoin(sem_table, sem_table.c.id == sem_target.c.table_id)
            .outerjoin(sem_schema, sem_schema.c.id == sem_table.c.schema_id)
        )
        .where(
            func.coalesce(fk_table.c.id, sem_table.c.id).isnot(None),
            func.coalesce(fk_target.c.id, sem_target.c.id).isnot(None),
        )
        .distinct()
        .subquery("resolved")
    )

    columns_count = _count_of(
        s.catalog_column, s.catalog_column.c.table_id == s.catalog_table.c.id
    )
    resolved_count = _count_of(
        resolved, resolved.c.owner_table_id == s.catalog_table.c.id
    )
    has_attribute_anywhere = (
        select(literal(1))
        .select_from(
            s.catalog_column.join(
                s.column_has_attribute,
                s.column_has_attribute.c.column_id == s.catalog_column.c.id,
            )
        )
        .where(s.catalog_column.c.table_id == s.catalog_table.c.id)
        .exists()
    )
    already_bridged = (
        select(literal(1))
        .select_from(
            s.sql_attribute.join(
                s.sql_attribute_sql,
                s.sql_attribute_sql.c.attribute_id == s.sql_attribute.c.id,
            ).join(
                s.sql_query_table,
                s.sql_query_table.c.sql_query_id == s.sql_attribute_sql.c.sql_query_id,
            )
        )
        .where(
            s.sql_attribute.c.source == SQL_ATTR_SOURCE_BRIDGE,
            s.sql_query_table.c.table_id == s.catalog_table.c.id,
        )
        .exists()
    )

    candidates = store().query_read(
        select(
            s.catalog_table.c.id.label("table_id"),
            s.catalog_table.c.name.label("table_name"),
            s.catalog_schema.c.name.label("schema_name"),
            s.catalog_table.c.description,
        )
        .select_from(_table_join())
        .where(
            s.catalog_database.c.name == database_name,
            columns_count >= 2,
            ~has_attribute_anywhere,
            ~already_bridged,
            resolved_count == columns_count,
        )
        .order_by(s.catalog_table.c.name)
    )
    if not candidates:
        return []

    pairs: dict[str, list[dict[str, Any]]] = {}
    for row in store().query_read(
        select(
            resolved.c.owner_table_id,
            resolved.c.source_column,
            resolved.c.target_table,
            resolved.c.target_schema,
            resolved.c.target_column,
            resolved.c.target_table_id,
        ).where(resolved.c.owner_table_id.in_([c["table_id"] for c in candidates]))
    ):
        pairs.setdefault(row["owner_table_id"], []).append(
            {
                "source_column": row["source_column"],
                "target_table": row["target_table"],
                "target_schema": row["target_schema"],
                "target_column": row["target_column"],
                "target_table_id": row["target_table_id"],
            }
        )

    return [{**dict(c), "fk_pairs": pairs.get(c["table_id"], [])} for c in candidates]
