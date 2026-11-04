# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Table families: two copies of one schema that differ only by name prefix.

A warehouse migrated from one system to another often keeps both: ``OLD_ROOMS``
beside ``NEW_ROOMS``, ``OLD_BUILDING`` beside ``NEW_BUILDING``. Each prefix is
a *family*. A semantic FK from ``NEW_ROOMS`` into ``OLD_BUILDING`` joins one
copy's rows to the other's, so join paths mix the two.
:func:`keep_semantic_fks_in_family` moves such an edge onto the twin table in
the source's own family, or drops it.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from auto_ontology.connectors.base import SQLDatabase
from auto_ontology.dal.attributes import (
    delete_semantic_fk,
    fetch_semantic_fk_join_groups,
    find_column_attribute_by_column_id,
    merge_semantic_fk,
    set_semantic_fk_join_default,
)
from auto_ontology.dal.datasources import (
    fetch_table_column_names,
    find_column_id_by_table_and_name,
)
from auto_ontology.utils.sql_identifiers import quoted_identifier

logger = logging.getLogger(__name__)

_MIN_TWINS = 3
_MIN_JACCARD = 0.5
_HISTORY_RE = re.compile(r"_(HIST|HISTORY)(_\d+)?$", re.IGNORECASE)


@dataclass
class FamilyPair:
    """Two prefixes whose tables mirror each other."""

    prefixes: tuple[str, str]
    # base name -> {prefix: table}
    twins: dict[str, dict[str, dict[str, Any]]]


def prefix_of(name: str) -> str | None:
    """``OLD_`` for ``OLD_ROOMS``; ``None`` when the name has no prefix."""
    head, sep, tail = name.partition("_")
    return f"{head}_".upper() if sep and head and tail else None


def is_history(name: str) -> bool:
    return bool(_HISTORY_RE.search(name))


def strip_prefix(name: str, prefixes: tuple[str, ...]) -> str:
    """*name* upper-cased without whichever of *prefixes* it starts with."""
    upper = name.upper()
    for prefix in prefixes:
        if upper.startswith(prefix):
            return upper[len(prefix) :]
    return upper


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def _stripped_columns(table: dict[str, Any], prefixes: tuple[str, ...]) -> set[str]:
    return {strip_prefix(c, prefixes) for c in table.get("columns") or []}


def detect_family_pairs(tables: list[dict[str, Any]]) -> list[FamilyPair]:
    """Prefix pairs sharing at least :data:`_MIN_TWINS` twin tables.

    *tables* are ``{id, name, schema_name, columns}`` rows. Two tables are twins
    when their names match after the prefix and their columns, compared the
    same way, overlap by Jaccard :data:`_MIN_JACCARD` or more. History tables
    (``*_HIST``) are never twins.
    """
    by_prefix: dict[str, dict[str, dict[str, Any]]] = {}
    for table in tables:
        prefix = prefix_of(table["name"])
        if prefix is None or is_history(table["name"]):
            continue
        base = table["name"].upper()[len(prefix) :]
        by_prefix.setdefault(prefix, {})[base] = table

    pairs: list[FamilyPair] = []
    prefixes = sorted(by_prefix)
    for i, a in enumerate(prefixes):
        for b in prefixes[i + 1 :]:
            twins: dict[str, dict[str, dict[str, Any]]] = {}
            for base in by_prefix[a].keys() & by_prefix[b].keys():
                ta, tb = by_prefix[a][base], by_prefix[b][base]
                if (
                    _jaccard(
                        _stripped_columns(ta, (a, b)), _stripped_columns(tb, (a, b))
                    )
                    >= _MIN_JACCARD
                ):
                    twins[base] = {a: ta, b: tb}
            if len(twins) >= _MIN_TWINS:
                pairs.append(FamilyPair(prefixes=(a, b), twins=twins))
    return pairs


def keep_semantic_fks_in_family(
    database_name: str,
    pairs: list[FamilyPair],
    connector: SQLDatabase | None,
) -> int:
    """Move semantic FKs that cross into the other family onto the twin.

    An edge whose source and target sit in the two families of one pair is
    redirected to the same-named column of the target's twin in the source's
    family, when that column covers the source's values at least as well.
    Otherwise it is dropped. Returns the number of crossing edges handled.
    """
    family_of = {
        str(table["id"]): (index, prefix)
        for index, pair in enumerate(pairs)
        for twin in pair.twins.values()
        for prefix, table in twin.items()
    }
    prefix_index = {
        prefix: index for index, pair in enumerate(pairs) for prefix in pair.prefixes
    }
    handled = 0
    for row in fetch_semantic_fk_join_groups(database_name):
        source_prefix = prefix_of(row["source_table_name"])
        target = family_of.get(str(row["target_table_id"]))
        if target is None or source_prefix is None:
            continue
        index, target_prefix = target
        if prefix_index.get(source_prefix) != index or source_prefix == target_prefix:
            continue
        moved = _move_to_twin(
            database_name, row, pairs[index], source_prefix, connector
        )
        delete_semantic_fk(row["source_column_id"], row["target_attribute_id"])
        handled += 1
        logger.info(
            "semantic FK %s family: %s.%s -> %s.%s%s",
            "kept in" if moved else "dropped, crosses",
            row["source_table_name"],
            row["source_column_name"],
            row["target_table_name"],
            row["target_column_name"],
            f" now -> {moved}" if moved else "",
        )
    return handled


def _move_to_twin(
    database_name: str,
    row: dict[str, Any],
    pair: FamilyPair,
    source_prefix: str,
    connector: SQLDatabase | None,
) -> str | None:
    """Point *row*'s source column at the twin column; its name, or ``None``.

    A twin that is the source table itself is refused: semantic FK resolution
    never links a table to itself either.
    """
    target_name = row["target_table_name"]
    base = target_name.upper()[len(prefix_of(target_name) or "") :]
    twin_table = pair.twins.get(base, {}).get(source_prefix)
    if twin_table is None or str(twin_table["id"]) == str(row["source_table_id"]):
        return None
    wanted = strip_prefix(row["target_column_name"], pair.prefixes)
    twin_column = next(
        (
            c
            for c in twin_table.get("columns") or []
            if strip_prefix(c, pair.prefixes) == wanted
        ),
        None,
    )
    if twin_column is None:
        return None
    column_id = find_column_id_by_table_and_name(
        twin_table["name"], twin_column, database_name
    )
    attr_id = find_column_attribute_by_column_id(column_id) if column_id else None
    if not attr_id or connector is None:
        return None
    source = (
        row.get("source_schema_name"),
        row["source_table_name"],
        row["source_column_name"],
    )
    old = _coverage(
        connector,
        source,
        (row.get("target_schema_name"), target_name, row["target_column_name"]),
    )
    new = _coverage(
        connector,
        source,
        (twin_table.get("schema_name"), twin_table["name"], twin_column),
    )
    if not new or old is None or new < old:
        return None
    if not merge_semantic_fk(row["source_column_id"], attr_id):
        return None
    set_semantic_fk_join_default(
        row["source_table_id"], attr_id, row["source_column_id"]
    )
    return f"{twin_table['name']}.{twin_column}"


def _coverage(
    connector: SQLDatabase,
    source: tuple[str | None, str, str],
    target: tuple[str | None, str, str],
) -> int | None:
    """Distinct source values found in the target column; ``None`` on error."""
    dialect = getattr(connector, "dialect", None)
    sc = quoted_identifier(source[2], dialect)
    tc = quoted_identifier(target[2], dialect)
    sql = (
        f"SELECT COUNT(DISTINCT {sc}) AS n "
        f"FROM {connector.qualify(source[0], source[1])} "
        f"WHERE {sc} IN (SELECT {tc} FROM {connector.qualify(target[0], target[1])})"
    )
    try:
        return int(connector.execute(sql).iloc[0, 0])
    except Exception:
        logger.info(
            "coverage check skipped for %s.%s", source[1], source[2], exc_info=True
        )
        return None


def resolve_table_families(database_name: str, connector: SQLDatabase | None) -> int:
    """Keep semantic FKs inside their table family. Returns pairs found."""
    pairs = detect_family_pairs(fetch_table_column_names(database_name))
    for pair in pairs:
        logger.info("table family %s / %s: %d twins", *pair.prefixes, len(pair.twins))
    if pairs:
        keep_semantic_fks_in_family(database_name, pairs, connector)
    return len(pairs)
