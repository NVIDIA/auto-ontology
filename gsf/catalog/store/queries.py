# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Storing parsed SQL statements and what they reference.

Public surface matches ``gsf.catalog.store.neo4j.queries`` function for
function. ``get_sql_counters`` and ``get_candidate_sql_ids`` are pure and live
in :mod:`gsf.catalog.query_stats`, re-exported here.

Two things the Cypher did that this deliberately does not:

**Per-month counters are not written.** The graph carried a
``count_{month}_{year}`` property per month a statement was seen. No reader has
ever been able to parse those names — every one matches
``count_monthly_YYYY_MM`` — so they hold no information anything can use, and
reproducing them would mean reviving dynamic property names in a relational
store to store data nothing reads. ``total_counter`` is kept and is what a fix
would use. See DECISIONS.md record 007.

**The ``deleted`` filter is dropped.** Both Cypher reads guard with
``coalesce(node.deleted, false) = false``, but nothing in GSF writes ``deleted``
— verified across the codebase. The guard is always true, so omitting it
preserves behaviour exactly, and carrying a column no writer sets would leave a
permanent puzzle.
"""

from __future__ import annotations

import logging

import pandas as pd
from sqlalchemy import select, update

from gsf.catalog.model.node import CatalogNode
from gsf.catalog.normalize import chunks

# Pure, so shared with the Neo4j backend rather than reimplemented.
from gsf.catalog.query_stats import (  # noqa: F401
    get_candidate_sql_ids,
    get_sql_counters,
)
from gsf.catalog.store.edges import add_edges, prepare_edge
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


def add_query(edges):
    """Write a parsed query's nodes and edges."""
    edges_data = [prepare_edge(edge) for edge in edges]
    for chunk in chunks(edges_data, 10):
        add_edges(chunk)


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

    # The Cypher walked SQL> from the statement to every Table and Column it
    # touches. Those are two association tables here, so it is two updates.
    for association, target, key in (
        (s.sql_query_table, s.catalog_table, "table_id"),
        (s.sql_query_column, s.catalog_column, "column_id"),
    ):
        if "last_query_timestamp" not in target.c:
            # Only Sql carries this today; the catalog tier does not. The Cypher
            # set it on Table and Column nodes regardless, because a property
            # graph needs no column to exist first. Nothing reads it there.
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
