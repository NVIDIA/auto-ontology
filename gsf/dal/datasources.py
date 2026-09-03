# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Catalog reads and writes: databases, schemas, tables, columns.

The functions at the bottom of the file reach the semantic tier — terms and
attributes — and their joins are where the interesting cases live.

Two shapes recur and are easy to break:

* **A database with no schemas, or a schema with no tables, does not appear.**
  These are inner joins, so an empty database is invisible rather than
  present-with-zero. Callers treat absence as "nothing here".
* **Zone scoping filters the counted rows, not just the returned ones**, so a
  scoped user sees a schema count covering only the tables they can see.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pandas as pd
from sqlalchemy import (
    ColumnElement,
    Text,
    and_,
    case,
    column,
    distinct,
    func,
    literal,
    select,
    update,
)
from sqlalchemy import values as sa_values

from gsf.dal import schema as s
from gsf.dal.session import store, write_transaction
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
            # Nothing writes a database description, so this is always None.
            # Kept in the shape because callers read the key.
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

    ``None`` rather than an empty result: the join requires at least one table,
    so a database with no tables produces no rows at all and the caller treats
    that as absent.
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

    An empty or absent id list means *every* schema, not none. Reading it the
    other way would leave every query unresolvable rather than raising.
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

    ``name`` breaks ties. Nearly every table has a query count of zero, so
    ordering by count alone would return rows in whatever order the store felt
    like, differing between calls — a function whose name promises an order has
    to impose a total one.
    """
    query_count = (
        select(func.count(s.sql_query__table.c.sql_query_id))
        .where(s.sql_query__table.c.table_id == s.catalog_table.c.id)
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

    Ambiguous by construction: a name is not unique across schemas or
    databases. Ordered by id so repeated calls agree with each other — which row
    wins is still arbitrary, but it is the same arbitrary row each time.
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

    A join is undirected, so both ends count — hence the union of the two
    columns rather than a single direction.
    """
    other = s.catalog_table.alias("other")
    outgoing = (
        select(other.c.id, other.c.name, other.c.description)
        .select_from(
            s.table__join.join(other, other.c.id == s.table__join.c.target_table_id)
        )
        .where(s.table__join.c.source_table_id == table_id)
    )
    incoming = (
        select(other.c.id, other.c.name, other.c.description)
        .select_from(
            s.table__join.join(other, other.c.id == s.table__join.c.source_table_id)
        )
        .where(s.table__join.c.target_table_id == table_id)
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
                s.table__join.c.join_columns,
            ).select_from(
                s.table__join.join(
                    source, source.c.id == s.table__join.c.source_table_id
                ).join(target, target.c.id == s.table__join.c.target_table_id)
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
    """Column id → its database/schema/table identity (ids + names).

    Ids come back alongside names so a caller that needs to let a client expand
    a Table/Column further (e.g. :func:`gsf.dal.terms.find_term_link_path`'s
    catalog enrichment) can reuse this instead of re-running the same
    database→schema→table→column join itself; a caller that only wants display
    text (e.g. :func:`gsf.dal.attributes.find_join_path`) ignores the extra keys.

    Returns ``{}`` on failure rather than raising: callers use this to decorate
    results, and losing the decoration beats losing the result.
    """
    if not col_ids:
        return {}
    try:
        rows = store().query_read(
            select(
                s.catalog_column.c.id.label("col_id"),
                s.catalog_table.c.id.label("table_id"),
                s.catalog_table.c.name.label("table_name"),
                s.catalog_schema.c.id.label("schema_id"),
                s.catalog_schema.c.name.label("schema_name"),
                s.catalog_database.c.id.label("database_id"),
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
            "table_id": r.get("table_id") or "",
            "table_name": r.get("table_name") or "",
            "schema_id": r.get("schema_id") or "",
            "schema_name": r.get("schema_name") or "",
            "database_id": r.get("database_id") or "",
            "database_name": r.get("database_name") or "",
        }
        for r in rows
        if r.get("col_id")
    }


def _set_column_property(table_id: str, values: dict[str, Any], column: str) -> None:
    """Set one property across several of a table's columns, in one statement.

    ``CASE name WHEN 'a' THEN … END`` rather than a statement per column — a
    table can have hundreds of them.
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


def store_column_date_formats(table_id: str, date_formats: dict[str, str]) -> None:
    """Write inferred value notations onto Column ``format``.

    ``format`` is generic storage notation (how values are written), not a
    date-specific property. Today only date inference fills it (``YYMMDD``,
    ``YYYY-MM-DD``, …); an address or id profiler would write the same field.
    The column's type/name/description say *what* the values are.

    A column whose notation could not be inferred is left alone rather than
    written as NULL: the inference declines to guess on a mixed column, and
    that is not a reason to discard what a previous run established.
    """
    _set_column_property(
        table_id,
        {name: str(fmt) for name, fmt in date_formats.items() if fmt},
        "format",
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

    ``coalesce(new, existing)``, and the direction matters: a curated
    description survives a batch that has nothing to say about it.

    Two statements, both ``UPDATE ... FROM (VALUES ...)``, in one transaction.
    A row-at-a-time loop was one autocommitted round trip per table *and* per
    column -- thousands for a warehouse-sized profiling run, each paying the
    network latency in full, and each leaving a partial batch behind for good
    if the run died halfway. The set-based form also evaluates the
    database-scoping subquery once instead of once per row.

    Note this makes a duplicated key within one batch indeterminate rather than
    last-wins: Postgres updates a target row at most once per statement, so of
    two rows naming the same column, one is applied and the other dropped.
    Callers build these from a catalog read, where the names are already unique.
    """
    with write_transaction():
        if table_rows:
            incoming = sa_values(
                column("table_name", Text),
                column("description", Text),
                name="incoming",
            ).data([(row["table_name"], row.get("description")) for row in table_rows])
            store().query_write(
                update(s.catalog_table)
                .where(
                    and_(
                        s.catalog_table.c.name == incoming.c.table_name,
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
                        incoming.c.description, s.catalog_table.c.description
                    )
                )
            )

        if column_rows:
            incoming = sa_values(
                column("table_name", Text),
                column("column_name", Text),
                column("description", Text),
                column("sample_values", Text),
                name="incoming",
            ).data(
                [
                    (
                        row["table_name"],
                        row["column_name"],
                        row.get("description"),
                        row.get("sample_values"),
                    )
                    for row in column_rows
                ]
            )
            store().query_write(
                update(s.catalog_column)
                .where(
                    and_(
                        s.catalog_column.c.name == incoming.c.column_name,
                        s.catalog_column.c.table_id.in_(
                            select(s.catalog_table.c.id)
                            .select_from(_table_join())
                            .where(
                                and_(
                                    s.catalog_database.c.name == database_name,
                                    s.catalog_table.c.name == incoming.c.table_name,
                                )
                            )
                        ),
                    )
                )
                .values(
                    description=func.coalesce(
                        incoming.c.description, s.catalog_column.c.description
                    ),
                    sample_values=func.coalesce(
                        incoming.c.sample_values, s.catalog_column.c.sample_values
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

    An id can name a database, schema, table or column, so the four tables are
    tried in turn. Properties with no matching column are dropped rather than
    rejected: callers set properties opportunistically, and refusing would fail
    writes that work today.
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

    Unknown labels are rejected with a warning rather than an exception — the
    label often arrives straight from a URL.
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
# Reads that reach the semantic tier
# ---------------------------------------------------------------------------


def _terms_count(table_id: ColumnElement) -> ColumnElement:
    """How many distinct Terms a table is associated with.

    Two routes, and the count is the deduplicated union of both:

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
                    s.column_attribute__term,
                    s.column_attribute__term.c.attribute_id
                    == link_table.c.attribute_id,
                )
            )
            .where(
                s.catalog_column.c.table_id == table_id,
                s.column_attribute__term.c.term_id == s.term.c.id,
            )
            .correlate(owner, s.term)
            .exists()
        )

    direct = (
        select(literal(1))
        .where(
            s.table__term.c.table_id == table_id,
            s.table__term.c.term_id == s.term.c.id,
        )
        .correlate(owner, s.term)
        .exists()
    )
    return (
        select(func.count())
        .select_from(s.term)
        .where(direct | via(s.column__has_attribute) | via(s.column__semantic_fk))
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
                s.sql_query__table,
                s.sql_query__table.c.table_id == s.catalog_table.c.id,
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
            ~select(s.table__term.c.term_id)
            .where(s.table__term.c.table_id == s.catalog_table.c.id)
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

    **``None`` means the table is missing, never that the page is empty.** That
    is why this runs two queries: the header decides existence, the page decides
    contents. A single join with ``OFFSET`` would lose the table's own fields as
    soon as *skip* ran past the last column, turning "page 3 of a 2-page table"
    into "no such table".
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

    Returns ``[]`` rather than raising if the query fails: this decorates
    retrieval results, and losing the decoration beats losing the results.

    A table with no columns does not appear at all — the column join is inner.
    Left that way deliberately: a column-less table in the catalog is a symptom
    worth seeing where it originates, not something to paper over here.
    """
    if not table_ids:
        return []
    try:
        rows = store().query_read(
            select(
                s.catalog_table.c.id,
                s.catalog_table.c.name,
                s.catalog_table.c.description,
                s.catalog_table.c.pk,
                s.catalog_database.c.name.label("database_name"),
                s.catalog_schema.c.name.label("schema_name"),
                s.catalog_column.c.name.label("column_name"),
                s.catalog_column.c.data_type,
                column_description_expr().label("column_description"),
                s.catalog_column.c.format,
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
                # The prediction graph keys its entities on this: a table
                # that arrives without a pk reaches KumoRFM with no identity,
                # which costs it every edge and makes it unusable in
                # `FOR EACH`. It has to survive every path to relevant_tables.
                "pk": row.get("pk") or [],
                "columns": [],
            },
        )
        # `name` is NOT NULL, so this cannot fire today -- kept so the shape
        # stays the same if that ever changes.
        if row["column_name"]:
            table["columns"].append(
                {
                    "name": row["column_name"],
                    "data_type": row["data_type"],
                    "description": row["column_description"],
                    "format": row["format"],
                }
            )
    return list(tables.values())


def fetch_table_context(table_id: str) -> dict[str, Any]:
    """``{columns, fks}`` for one table — what the SQL generator is handed.

    ``is_foreign_key_target`` marks a column some *other* column already points
    at; ``gsf.semantic.fk_suggester`` drops those from the columns it offers the
    LLM, so it cannot propose an FK that inverts one the catalog already knows.
    """
    is_fk_target = (
        select(literal(1))
        .select_from(s.column__foreign_key)
        .where(s.column__foreign_key.c.target_column_id == s.catalog_column.c.id)
        .exists()
    )
    columns = [
        {
            "id": r["id"],
            "name": r["name"],
            "data_type": r["data_type"],
            "description": r["description"],
            "ordinal_position": r["ordinal_position"],
            "sample_values": r["sample_values"],
            "format": r["format"],
            "is_foreign_key_target": bool(r["is_foreign_key_target"]),
        }
        for r in store().query_read(
            select(
                s.catalog_column.c.id,
                s.catalog_column.c.name,
                s.catalog_column.c.data_type,
                column_description_expr().label("description"),
                s.catalog_column.c.ordinal_position,
                s.catalog_column.c.sample_values,
                s.catalog_column.c.format,
                is_fk_target.label("is_foreign_key_target"),
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
                    s.column__foreign_key,
                    s.column__foreign_key.c.source_column_id == source.c.id,
                )
                .join(target, target.c.id == s.column__foreign_key.c.target_column_id)
                .join(target_table, target_table.c.id == target.c.table_id)
            )
            .where(source.c.table_id == table_id)
        )
    ]
    return {"columns": columns, "fks": fks}


def fetch_tables_and_columns_by_node_ids(
    node_ids: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Table and column frames for ``CatalogEmbeddingRowsOp``.

    *node_ids* mixes table and column ids freely. A table id pulls in all of its
    columns; a column id pulls in only itself. Both frames carry
    ``database_name``, and the third return value is the first one found — the
    caller assumes a single database and there is nothing here that enforces it.
    """
    conn = store()
    ids = list(node_ids)

    columns_df = pd.DataFrame(
        [
            # ``sample_values`` is stored as a JSON string. It has to be parsed
            # here: the operator downstream slices it to five, and on a string
            # that takes five *characters* rather than five values.
            {**dict(r), "sample_values": parse_sample_values(r["sample_values"])}
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
                    # The table's own description, not the fallback -- unlike
                    # the columns three lines earlier. Asymmetric on purpose:
                    # these frames feed embeddings, and widening what gets
                    # embedded would shift retrieval results with no test
                    # noticing.
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

    "Every column resolves" is checked twice, and the two are not redundant:
    one asks whether each column has an outgoing key at all, the other whether
    each key actually lands on a column inside a table. A key pointing at a
    column whose table was never ingested passes the first and fails the
    second.
    """
    fk_target = s.catalog_column.alias("fk_target")
    fk_table = s.catalog_table.alias("fk_table")
    fk_schema = s.catalog_schema.alias("fk_schema")
    sem_target = s.catalog_column.alias("sem_target")
    sem_table = s.catalog_table.alias("sem_table")
    sem_schema = s.catalog_schema.alias("sem_schema")

    # Column -> the table/schema/column its FK lands on, by either route.
    # `resolved_count` below counts these rows, so a column with no resolvable
    # target contributes nothing and the size comparison fails -- the second of
    # the two checks described above.
    #
    # The SEMANTIC_FK route is Column -> ColumnAttribute <- Column: the
    # attribute this column references is *owned* by some other column, and that
    # other column is the join target. Reversing it would make every bridge
    # point back at itself.
    sem_owner = s.column__has_attribute.alias("sem_owner")
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
                s.column__foreign_key,
                s.column__foreign_key.c.source_column_id == s.catalog_column.c.id,
            )
            .outerjoin(
                fk_target, fk_target.c.id == s.column__foreign_key.c.target_column_id
            )
            .outerjoin(fk_table, fk_table.c.id == fk_target.c.table_id)
            .outerjoin(fk_schema, fk_schema.c.id == fk_table.c.schema_id)
            .outerjoin(
                s.column__semantic_fk,
                s.column__semantic_fk.c.column_id == s.catalog_column.c.id,
            )
            .outerjoin(
                sem_owner,
                sem_owner.c.attribute_id == s.column__semantic_fk.c.attribute_id,
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
    # DISTINCT on the *column*, not a row count. `resolved` carries one row per
    # resolvable target, and a column may have several -- two foreign keys, or a
    # SEMANTIC_FK reaching an attribute owned by more than one column. Counting
    # rows lets a table where one column resolves twice and another not at all
    # match `columns_count`, which is exactly the "every column is a key" claim
    # this is supposed to enforce. That is the first of the two checks the
    # docstring describes; the subquery's own WHERE is the second.
    resolved_count = (
        select(func.count(distinct(resolved.c.column_id)))
        .select_from(resolved)
        .where(resolved.c.owner_table_id == s.catalog_table.c.id)
        .scalar_subquery()
    )
    resolved_rows = _count_of(
        resolved, resolved.c.owner_table_id == s.catalog_table.c.id
    )
    has_attribute_anywhere = (
        select(literal(1))
        .select_from(
            s.catalog_column.join(
                s.column__has_attribute,
                s.column__has_attribute.c.column_id == s.catalog_column.c.id,
            )
        )
        .where(s.catalog_column.c.table_id == s.catalog_table.c.id)
        .exists()
    )
    already_bridged = (
        select(literal(1))
        .select_from(
            s.sql_attribute.join(
                s.sql_attribute__sql,
                s.sql_attribute__sql.c.attribute_id == s.sql_attribute.c.id,
            ).join(
                s.sql_query__table,
                s.sql_query__table.c.sql_query_id
                == s.sql_attribute__sql.c.sql_query_id,
            )
        )
        .where(
            s.sql_attribute.c.source == SQL_ATTR_SOURCE_BRIDGE,
            s.sql_query__table.c.table_id == s.catalog_table.c.id,
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
            # Both bounds. The line above says every column resolves; this one
            # says none resolves twice. Without it a 2-column junction whose
            # first column carries two foreign keys still qualifies and hands
            # `_generate_bridge_sql_attribute` three pairs, so the bridge joins
            # a table the junction does not connect.
            resolved_rows == columns_count,
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


def find_column_id_by_table_and_name(
    table_name: str,
    column_name: str,
    database_name: str | None = None,
) -> str | None:
    """Resolve a ``table.column`` reference from generated SQL to its Column id.

    Case-insensitive on both table and column name, since the SQL came from an
    LLM and may not match the catalog's stored casing exactly. When
    *database_name* is given, scopes the match to that database only — the
    same table/column name can exist in multiple co-resident databases
    (see :func:`gsf.dal.attributes.find_unlinked_fk_columns`), and an unscoped
    match could silently resolve to the wrong database's column. Returns
    ``None`` (not an exception) on no match or an ambiguous multi-database
    match without *database_name*, so callers can treat "can't verify" the
    same as "no known edge" rather than crash.
    """
    if not table_name or not column_name:
        return None
    join = s.catalog_column.join(
        s.catalog_table, s.catalog_table.c.id == s.catalog_column.c.table_id
    )
    where = [
        func.lower(s.catalog_table.c.name) == table_name.lower(),
        func.lower(s.catalog_column.c.name) == column_name.lower(),
    ]
    if database_name:
        join = join.join(
            s.catalog_schema, s.catalog_schema.c.id == s.catalog_table.c.schema_id
        ).join(
            s.catalog_database,
            s.catalog_database.c.id == s.catalog_schema.c.database_id,
        )
        where.append(s.catalog_database.c.name == database_name)
        limit = 1
    else:
        limit = 2
    rows = store().query_read(
        select(s.catalog_column.c.id).select_from(join).where(*where).limit(limit)
    )
    if not database_name and len(rows) > 1:
        logger.info(
            "find_column_id_by_table_and_name: ambiguous match for %s.%s "
            "with no database_name given — treating as unresolved",
            table_name,
            column_name,
        )
        return None
    return rows[0]["id"] if rows else None


def find_table_id_by_name(
    table_name: str, database_name: str | None = None
) -> str | None:
    """Resolve a bare table name (from a join-path hop) to its Table id.

    Same database-scoping rationale as :func:`find_column_id_by_table_and_name`
    — an unscoped lookup (e.g. :func:`fetch_table_by_name`) risks matching a
    same-named table in a different co-resident database.
    """
    if not table_name:
        return None
    join = s.catalog_table
    where = [func.lower(s.catalog_table.c.name) == table_name.lower()]
    if database_name:
        join = s.catalog_table.join(
            s.catalog_schema, s.catalog_schema.c.id == s.catalog_table.c.schema_id
        ).join(
            s.catalog_database,
            s.catalog_database.c.id == s.catalog_schema.c.database_id,
        )
        where.append(s.catalog_database.c.name == database_name)
        limit = 1
    else:
        limit = 2
    rows = store().query_read(
        select(s.catalog_table.c.id).select_from(join).where(*where).limit(limit)
    )
    if not database_name and len(rows) > 1:
        logger.info(
            "find_table_id_by_name: ambiguous match for %s with no "
            "database_name given — treating as unresolved",
            table_name,
        )
        return None
    return rows[0]["id"] if rows else None


def find_table_key_columns(
    table_name: str, database_name: str | None = None
) -> dict[str, list[str]]:
    """Return ``{"pk": [...], "unique": [...]}`` column names (lowercased) for
    a bare table name, used to detect a vacuous ``GROUP BY``/``PARTITION BY``
    (grouping by a column already unique per row makes the aggregate a no-op).

    Same database-scoping rationale as :func:`find_table_id_by_name` — scope
    to *database_name* when given, since an unscoped lookup risks matching a
    same-named table in a different co-resident database. ``pk`` comes
    from ``catalog_table.pk`` (set at ingestion from the DDL); ``unique`` comes
    from ``catalog_column.is_unique`` (set from observed-data profiling — see
    :func:`store_column_uniqueness`), so it also catches a unique-in-practice
    column with no declared constraint. Returns ``{"pk": [], "unique": []}``
    (not ``None``) when no match is found.
    """
    empty: dict[str, list[str]] = {"pk": [], "unique": []}
    if not table_name:
        return empty
    join = s.catalog_table
    where = [func.lower(s.catalog_table.c.name) == table_name.lower()]
    if database_name:
        join = s.catalog_table.join(
            s.catalog_schema, s.catalog_schema.c.id == s.catalog_table.c.schema_id
        ).join(
            s.catalog_database,
            s.catalog_database.c.id == s.catalog_schema.c.database_id,
        )
        where.append(s.catalog_database.c.name == database_name)
    rows = store().query_read(
        select(s.catalog_table.c.id, s.catalog_table.c.pk)
        .select_from(join)
        .where(*where)
        .limit(1)
    )
    if not rows:
        return empty
    table_id, pk = rows[0]["id"], rows[0]["pk"]
    pk_cols = [str(c).lower() for c in (pk or [])]
    unique_rows = store().query_read(
        select(s.catalog_column.c.name).where(
            s.catalog_column.c.table_id == table_id,
            s.catalog_column.c.is_unique.is_(True),
        )
    )
    unique_cols = list(
        dict.fromkeys(r["name"].lower() for r in unique_rows if r["name"])
    )
    return {"pk": pk_cols, "unique": unique_cols}
