# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Storing parsed SQL statements and what they reference.

``get_sql_counters`` and ``get_candidate_sql_ids`` are pure and live in
:mod:`gsf.catalog.query_stats`, re-exported here.

Two things this deliberately does not do:

**Per-month counters are not written.** A ``count_{month}_{year}`` value used
to be stored per month a statement was seen. No reader has ever been able to
parse those names — every one matches
``count_monthly_YYYY_MM`` — so they hold no information anything can use, and
reproducing them would mean reviving dynamic property names in a relational
store to store data nothing reads. ``total_counter`` is kept and is what a fix
would use — see ``sql_attribute_suggester._usage_score``.

**The ``deleted`` filter is dropped.** Both reads used to guard with
``coalesce(node.deleted, false) = false``, but nothing in GSF writes ``deleted``
— verified across the codebase. The guard is always true, so omitting it
preserves behaviour exactly, and carrying a column no writer sets would leave a
permanent puzzle.
"""

from __future__ import annotations

import logging

import pandas as pd
from sqlalchemy import distinct, func, select, update
from sqlalchemy.dialects.postgresql import insert

from gsf.catalog.constants import Edges, Props
from gsf.catalog.model.node import CatalogNode

# Pure, so it lives outside this module rather than being reimplemented.
from gsf.catalog.query_stats import (  # noqa: F401
    get_candidate_sql_ids,
    get_sql_counters,
)
from gsf.catalog.store import registry
from gsf.catalog.store.rows import upsert_row
from gsf.dal import schema as s
from gsf.dal.session import store

logger = logging.getLogger(__name__)

_EMPTY_SQLS_COLUMNS = [
    "sql_id",
    "tbls",
    "cols",
    "nodes_count",
    "join_count",
    "union_count",
    "sql_full_query",
]


# ---------------------------------------------------------------------------
# Writing a statement and what it references
# ---------------------------------------------------------------------------
#
# ``sql_parse`` hands over pairs of endpoints plus the properties of the
# relationship between them, four kinds in all: a statement to a table, a
# statement to a column, and column-to-column ``JOIN`` / ``UNION``. Each pair
# becomes two upserted rows and one association row.


#: Link properties that **accumulate** on conflict instead of overwriting.
#:
#: Each statement joining two columns adds its own reference, so the link
#: records *how often and where* the join was observed. Overwriting would
#: leave only the most recent statement and make the history meaningless.
_ACCUMULATING = frozenset({"refs"})


def _link_kind(properties: dict) -> str:
    """Which relationship a set of link properties describes.

    The kind is inferred from the properties rather than passed alongside
    them, because that is the shape ``sql_parse`` produces.
    """
    if Props.JOIN in properties:
        return Edges.JOIN
    if Props.UNION in properties:
        return Edges.UNION
    if Props.SQL_ID in properties:
        return Edges.SQL
    if Props.ANALYSIS_ID in properties:
        return Edges.HAS_SQL
    return next(iter(properties))


def _reject_nested(properties: dict) -> None:
    """Reject nested property values.

    These become scalar columns. Postgres would take nested JSON in a jsonb
    column, but not here — failing with the offending key named beats the
    insert's own message.
    """
    for key, value in properties.items():
        nested = isinstance(value, dict) or (
            isinstance(value, list)
            and any(isinstance(item, (list, dict)) for item in value)
        )
        if nested:
            raise ValueError(
                f"Invalid property name: {key}\nThe property value: {value}"
            )


def add_query(edges) -> None:
    """Write a parsed statement, the rows it references, and the links."""
    for source, target, properties in edges:
        _write_link(source, target, properties)


def _write_link(source: CatalogNode, target: CatalogNode, properties: dict) -> None:
    """Upsert both endpoint rows and the association between them."""
    for endpoint in (source, target):
        _reject_nested(endpoint.get_properties())
        if endpoint.get_override_existing_props():
            _reject_nested(endpoint.get_override_existing_props())
    _reject_nested(properties)

    spec = registry.link_spec(
        _link_kind(properties), source.get_label(), target.get_label()
    )

    source_id = upsert_row(
        source.get_label(),
        source.get_match_props(),
        source.get_properties(),
        on_match=source.get_override_existing_props() or {},
    )
    # For a foreign key link the child's row *carries* the relationship, so the
    # parent id goes in with the child rather than as a separate write.
    target_id = upsert_row(
        target.get_label(),
        target.get_match_props(),
        target.get_properties(),
        parent_id=source_id if spec.is_parent_link else None,
        on_match=target.get_override_existing_props() or {},
    )

    if spec.is_parent_link:
        return

    values = {spec.source_column: source_id, spec.target_column: target_id}
    payload = {k: v for k, v in properties.items() if k in spec.property_columns}
    values.update(payload)

    statement = insert(spec.table).values(**values)
    conflict = [spec.source_column, spec.target_column]
    if not payload:
        statement = statement.on_conflict_do_nothing(index_elements=conflict)
    else:
        updates = {}
        for key, value in payload.items():
            column = spec.table.c[key]
            if key in _ACCUMULATING:
                # Append, then de-duplicate: re-ingesting the same statement
                # must not grow the array without bound.
                updates[key] = _dedupe(column + getattr(statement.excluded, key))
            else:
                updates[key] = getattr(statement.excluded, key)
        statement = statement.on_conflict_do_update(
            index_elements=conflict, set_=updates
        )
    store().query_write(statement)


def _dedupe(array):
    """Distinct elements of a text[], order not preserved.

    The graph did not de-duplicate and would append the same reference on every
    re-ingest of the same statement; that is a leak rather than a behaviour
    worth reproducing, and the readers treat the array as a set.

    ``column_valued`` is what makes this legal: ``unnest`` is a set-returning
    function and has to sit in ``FROM``, not in the select list. Calling it
    inline instead makes SQLAlchemy infer the FROM from the array expression
    and emit ``FROM column_union, column_union AS excluded``, which Postgres
    rejects.
    """
    return select(
        func.array_agg(distinct(func.unnest(array).column_valued("ref")))
    ).scalar_subquery()


def get_sql_by_full_query(sql_full_query: str):
    rows = store().query_read(
        select(s.sql_query.c.id).where(s.sql_query.c.sql_full_query == sql_full_query)
    )
    return rows[0]["id"] if rows else None


def update_counters_and_timestamps_for_query_and_affected_data(
    identical_sql_id: str,
    sql_node: CatalogNode,
    update_data_last_query_timestamp: bool = True,
):
    """Add this sighting's count to the stored statement, and stamp the time.

    The per-month counters ``get_sql_counters`` also returns are ignored — see
    the module docstring.
    """
    latest_timestamp = sql_node.get_properties()["last_query_timestamp"]
    total_counter, _unreadable_month_counters = get_sql_counters(sql_node)

    store().query_write(
        update(s.sql_query)
        .where(s.sql_query.c.id == identical_sql_id)
        .values(
            last_query_timestamp=latest_timestamp,
            total_counter=s.sql_query.c.total_counter + total_counter,
        )
    )

    if not update_data_last_query_timestamp:
        return

    # A statement references both tables and columns, through two association
    # tables -- so two updates.
    for association, target, key in (
        (s.sql_query_table, s.catalog_table, "table_id"),
        (s.sql_query_column, s.catalog_column, "column_id"),
    ):
        if "last_query_timestamp" not in target.c:
            # Only statements carry a last-query timestamp; the catalog tier
            # has no column for one, and nothing reads it there.
            continue
        store().query_write(
            update(target)
            .where(
                target.c.id.in_(
                    select(association.c[key]).where(
                        association.c.sql_query_id == identical_sql_id
                    )
                )
            )
            .values(last_query_timestamp=latest_timestamp)
        )


def load_sqls_to_tables() -> pd.DataFrame:
    """Every stored statement with the table and column ids it references."""
    rows = store().query_read(
        select(
            s.sql_query.c.id.label("sql_id"),
            s.sql_query.c.nodes_count,
            s.sql_query.c.join_count,
            s.sql_query.c.union_count,
            s.sql_query.c.sql_full_query,
        )
    )
    if not rows:
        return pd.DataFrame(columns=_EMPTY_SQLS_COLUMNS)

    tables: dict[str, list[str]] = {}
    for row in store().query_read(select(s.sql_query_table)):
        tables.setdefault(row["sql_query_id"], []).append(row["table_id"])

    columns: dict[str, list[str]] = {}
    for row in store().query_read(select(s.sql_query_column)):
        columns.setdefault(row["sql_query_id"], []).append(row["column_id"])

    return pd.DataFrame(
        [
            {
                "sql_id": row["sql_id"],
                "tbls": tables.get(row["sql_id"], []),
                "cols": columns.get(row["sql_id"], []),
                "nodes_count": row["nodes_count"],
                "join_count": row["join_count"] or 0,
                "union_count": row["union_count"] or 0,
                "sql_full_query": row["sql_full_query"],
            }
            for row in rows
        ]
    )
