# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Near-copies of a table, for the ``retrieve`` tool to name.

Warehouses keep several tables with nearly the same columns — a current table
and its history, a full list and a filtered one, the same feed loaded twice.
A search returns whichever scored highest, and a query on the wrong copy runs
and returns plausible rows that are not the answer. So every table ``retrieve``
shows is compared by column names with the rest of the catalog, and each close
match is named with its row count and the columns only one side has: enough
for the model to check which copy fits before it commits to one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from sqlglot import exp

from auto_ontology.dal.datasources import fetch_table_column_names
from auto_ontology.retrieval.text_to_sql.react.context import ToolContext
from auto_ontology.retrieval.text_to_sql.react.run_sql import connector_for

logger = logging.getLogger(__name__)

#: Shared columns below this never make two tables copies: ids and audit
#: columns alone are shared by unrelated tables.
_MIN_SHARED = 3
#: Shared over all distinct columns of the two.
_MIN_JACCARD = 0.5
#: Shared over the narrower table's columns: one table is mostly the other.
_MIN_CONTAINMENT = 0.8
#: Containment alone counts only between tables of similar width: a lookup
#: table whose few columns were copied into a wide fact table is not a copy.
_MAX_WIDTH_RATIO = 2.0
_SIBLINGS_PER_TABLE = 3
_MAX_LISTED_COLUMNS = 6


@dataclass(frozen=True)
class Sibling:
    """A table that shares most of its columns with another."""

    table: dict[str, Any]
    shared: int
    only_here: list[str]
    only_there: list[str]
    jaccard: float


def _qualified(table: dict) -> str:
    return ".".join(p for p in (table.get("schema_name"), table.get("name")) if p)


def _column_names(table: dict) -> dict[str, str]:
    """Lowercased name to name; columns may be names or column dicts."""
    names = [
        str(c.get("name") if isinstance(c, dict) else c)
        for c in table.get("columns") or []
    ]
    return {n.lower(): n for n in names}


def find_siblings(
    table: dict[str, Any], catalog: list[dict[str, Any]]
) -> list[Sibling]:
    """Tables in *catalog* close enough to *table* by column names, closest first."""
    mine = _column_names(table)
    found: list[Sibling] = []
    for other in catalog:
        if str(other.get("id")) == str(table.get("id")):
            continue
        if other.get("database_name") != table.get("database_name"):
            continue
        theirs = _column_names(other)
        shared = mine.keys() & theirs.keys()
        if len(shared) < _MIN_SHARED:
            continue
        jaccard = len(shared) / len(mine.keys() | theirs.keys())
        narrow, wide = sorted((len(mine), len(theirs)))
        contained = (
            len(shared) / narrow >= _MIN_CONTAINMENT
            and wide <= _MAX_WIDTH_RATIO * narrow
        )
        if jaccard < _MIN_JACCARD and not contained:
            continue
        found.append(
            Sibling(
                table=other,
                shared=len(shared),
                only_here=[mine[k] for k in mine if k not in shared],
                only_there=[theirs[k] for k in theirs if k not in shared],
                jaccard=jaccard,
            )
        )
    found.sort(key=lambda s: (-s.jaccard, _qualified(s.table)))
    return found[:_SIBLINGS_PER_TABLE]


def _catalog(ctx: ToolContext) -> list[dict[str, Any]]:
    if ctx.catalog_tables is None:
        ctx.catalog_tables = fetch_table_column_names(ctx.database_name)
    return ctx.catalog_tables


def _row_count(ctx: ToolContext, table: dict) -> int | None:
    """Exact rows in *table*, counted once per episode; ``None`` if it fails."""
    key = str(table.get("id"))
    if key not in ctx.row_counts:
        ctx.row_counts[key] = None
        connector = connector_for(ctx)
        if connector is not None:
            sql = (
                exp.select(exp.Count(this=exp.Star()))
                .from_(exp.table_(table["name"], db=table.get("schema_name") or None))
                .sql(dialect=getattr(connector, "dialect", None) or None)
            )
            try:
                ctx.row_counts[key] = int(connector.execute(sql).iat[0, 0])
            except Exception:  # noqa: BLE001 — a missing count only drops a detail
                logger.info("ReAct sibling row count failed: %s", sql, exc_info=True)
    return ctx.row_counts[key]


def _with_rows(ctx: ToolContext, table: dict) -> str:
    count = _row_count(ctx, table)
    return _qualified(table) + (f" ({count} rows)" if count is not None else "")


def _only(table: dict, columns: list[str]) -> str:
    if not columns:
        return f"{_qualified(table)} has no columns of its own"
    listed = ", ".join(columns[:_MAX_LISTED_COLUMNS])
    if len(columns) > _MAX_LISTED_COLUMNS:
        listed += f" and {len(columns) - _MAX_LISTED_COLUMNS} more"
    return f"only {_qualified(table)} has {listed}"


def sibling_lines(ctx: ToolContext, new_tables: list[dict]) -> list[str]:
    """A line for each near-copy of each new table, once per pair per episode."""
    lines: list[str] = []
    catalog = _catalog(ctx) if new_tables else []
    for table in new_tables:
        for sibling in find_siblings(table, catalog):
            pair = frozenset((str(table["id"]), str(sibling.table["id"])))
            if pair in ctx.sibling_pairs:
                continue
            ctx.sibling_pairs.add(pair)
            lines.append(
                f"SIMILAR {_with_rows(ctx, table)} ~ {_with_rows(ctx, sibling.table)}: "
                f"{sibling.shared} shared columns; "
                f"{_only(table, sibling.only_here)}; "
                f"{_only(sibling.table, sibling.only_there)}"
            )
    return lines
