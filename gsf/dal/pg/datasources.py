# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Catalog reads and writes: databases, schemas, tables, columns.

**19 of the module's 26 functions.** The other seven reach the semantic tier —
five of them only through the description fallback — and moved to Phase 7 by
[DECISION-009], because nothing writes terms or attributes until then and
against no data a wrong join is indistinguishable from a correct one. They raise
here rather than returning something plausible.

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
from sqlalchemy import and_, case, func, literal, select, update

from gsf.dal.pg import schema as s
from gsf.dal.pg.session import store
from gsf.dal.pg.users import resolve_accessible_catalog_ids

logger = logging.getLogger(__name__)

#: Labels ``fetch_node_properties_by_id`` will look up, and their tables.
_NODE_TABLES = {
    "Database": s.catalog_database,
    "Schema": s.catalog_schema,
    "Table": s.catalog_table,
    "Column": s.catalog_column,
}

_PHASE_7 = (
    "{name} reads the semantic tier and moves to Phase 7 with the terms and "
    "attributes it joins to (DECISION-009). Until then it cannot be verified: "
    "against no semantic data a wrong join looks exactly like a correct one."
)


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
# Phase 7 — these read the semantic tier
# ---------------------------------------------------------------------------


def fetch_tables_for_schema(
    schema_id: str,
    *,
    database_name: str | None = None,
    zone_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    raise NotImplementedError(_PHASE_7.format(name="fetch_tables_for_schema"))


def fetch_all_tables_without_term(
    database_name: str | None = None,
) -> list[dict[str, Any]]:
    raise NotImplementedError(_PHASE_7.format(name="fetch_all_tables_without_term"))


def fetch_columns_for_table(
    table_id: str,
    *,
    skip: int = 0,
    limit: int | None = None,
) -> dict[str, Any] | None:
    raise NotImplementedError(_PHASE_7.format(name="fetch_columns_for_table"))


def fetch_tables_and_columns_by_node_ids(
    node_ids: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    raise NotImplementedError(
        _PHASE_7.format(name="fetch_tables_and_columns_by_node_ids")
    )


def fetch_bridge_table_candidates(database_name: str) -> list[dict[str, Any]]:
    raise NotImplementedError(_PHASE_7.format(name="fetch_bridge_table_candidates"))


def fetch_tables_by_ids(table_ids: list[str]) -> list[dict[str, Any]]:
    raise NotImplementedError(_PHASE_7.format(name="fetch_tables_by_ids"))


def fetch_table_context(table_id: str) -> dict[str, Any]:
    raise NotImplementedError(_PHASE_7.format(name="fetch_table_context"))
