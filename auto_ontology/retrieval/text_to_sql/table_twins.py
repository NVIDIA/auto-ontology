# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pair each table in scope with the tables that hold nearly the same columns.

Warehouses keep several tables of one thing side by side: a current table and
its history, a subset and the full list, two loads of one catalog. Their column
names are almost identical, so retrieval finds whichever one its scores favour,
and SQL generation picks between them without knowing what tells them apart.
What does is usually a handful of columns one of them has alone -- a period
column on the history table, status columns on the subset -- and those are the
columns a question needing that table mentions.

So every table in scope is compared with the rest of its schema by column
names. A *twin* shares at least ``_MIN_SHARED_SHARE`` of the smaller table's
columns. Each table in scope is annotated with its twins and the columns that
differ, and a twin not already in scope is added, since the prompt can only use
tables it lists. Legacy copies are left out: compilation already judged them
replaced.
"""

from __future__ import annotations

import logging

from auto_ontology.dal.datasources import (
    fetch_legacy_table_ids,
    fetch_tables_by_ids,
    fetch_tables_for_legacy_detection,
)

logger = logging.getLogger(__name__)

_MIN_SHARED_SHARE = 0.8
#: Below this many columns, overlapping names say little: most small tables
#: share an id and a name column with something.
_MIN_COLUMNS = 4
_MAX_ADDED_TWINS = 6


def _names(table: dict) -> dict[str, str]:
    """Column names of *table*, keyed by their case-folded form."""
    return {
        str(column["name"]).casefold(): str(column["name"])
        for column in table.get("columns") or []
        if column.get("name")
    }


def _twin_note(own: dict[str, str], other: dict[str, str], name: str) -> dict:
    return {
        "name": name,
        "only_here": sorted(own[key] for key in own.keys() - other.keys()),
        "only_there": sorted(other[key] for key in other.keys() - own.keys()),
    }


def find_twins(
    table: dict, catalog: list[dict]
) -> list[tuple[dict, dict[str, str], dict[str, str]]]:
    """Catalog tables of *table*'s schema sharing most of its columns.

    Returns ``(twin, table_columns, twin_columns)`` triples, the column maps as
    :func:`_names` builds them, compared from the catalog so a table pruned to
    fewer columns for the prompt is still judged on all of them.
    """
    entry = next((t for t in catalog if str(t["id"]) == str(table.get("id"))), None)
    if entry is None:
        return []
    own = _names(entry)
    twins = []
    for other in catalog:
        if other["id"] == entry["id"] or other.get("schema_name") != entry.get(
            "schema_name"
        ):
            continue
        theirs = _names(other)
        smaller = min(len(own), len(theirs))
        if smaller < _MIN_COLUMNS:
            continue
        if len(own.keys() & theirs.keys()) / smaller >= _MIN_SHARED_SHARE:
            twins.append((other, own, theirs))
    return twins


def attach_twins(
    relevant_tables: list[dict], database_name: str | None
) -> tuple[list[dict], list[str]]:
    """Annotate *relevant_tables* with their twins and add the missing ones.

    Each table gains ``twins``: ``{"name", "only_here", "only_there"}`` per
    twin. An added twin carries ``twin_of``, the name of the table in scope it
    was found from, and is appended after the tables already in scope. Returns
    the tables and the names of those added.
    """
    if not database_name or not relevant_tables:
        return relevant_tables, []
    legacy = {str(i) for i in fetch_legacy_table_ids(database_name)}
    catalog = [
        t
        for t in fetch_tables_for_legacy_detection(database_name)
        if str(t["id"]) not in legacy
    ]
    in_scope = {str(t.get("id")) for t in relevant_tables}
    to_add: dict[str, tuple[str, dict]] = {}
    for table in relevant_tables:
        notes = []
        for twin, own, theirs in find_twins(table, catalog):
            notes.append(_twin_note(own, theirs, twin["name"]))
            twin_id = str(twin["id"])
            if twin_id not in in_scope and twin_id not in to_add:
                to_add[twin_id] = (
                    table.get("name", ""),
                    _twin_note(theirs, own, table.get("name", "")),
                )
        if notes:
            table["twins"] = notes

    added_ids = list(to_add)[:_MAX_ADDED_TWINS]
    added = fetch_tables_by_ids(added_ids) if added_ids else []
    for table in added:
        base_name, note = to_add[str(table["id"])]
        table["twin_of"] = base_name
        table["twins"] = [note]
    return relevant_tables + added, [t["name"] for t in added]


__all__ = ["attach_twins", "find_twins"]
