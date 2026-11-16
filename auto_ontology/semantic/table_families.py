# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Table families: two copies of one schema that differ only by name prefix.

A warehouse migrated from one system to another often keeps both: ``OLD_ROOMS``
beside ``NEW_ROOMS``, ``OLD_BUILDING`` beside ``NEW_BUILDING``. Each prefix is
a *family*. A semantic FK from ``NEW_ROOMS`` into ``OLD_BUILDING`` joins one
copy's rows to the other's, so join paths mix the two.
:func:`keep_semantic_fks_in_family` moves such an edge onto the twin table in
the source's own family, or drops it when the twin is missing or worse.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

from auto_ontology.connectors.base import SQLDatabase, StatementTimeout
from auto_ontology.dal.attributes import (
    delete_semantic_fk,
    fetch_semantic_fk_join_groups,
    find_column_attribute_by_column_id,
    merge_semantic_fk,
    set_semantic_fk_join_default,
)
from auto_ontology.dal.datasources import (
    fetch_table_column_names,
    find_column_id_in_table,
)
from auto_ontology.utils.sql_identifiers import quoted_identifier

logger = logging.getLogger(__name__)

_MIN_TWINS = 3
_MIN_JACCARD = 0.5
# How long one coverage query may run, on connectors that support a cap. The
# same as the profiling cap in visit_enter.py. Each crossing FK costs two full
# scans and the rest of compilation waits on them; a check that does not finish
# keeps the edge as it is.
_COVERAGE_TIMEOUT_SECONDS = 120
_HISTORY_RE = re.compile(r"_(HIST|HISTORY)(_\d+)?$", re.IGNORECASE)


class _Outcome(Enum):
    MOVED = "moved"
    DROP = "drop"
    KEEP = "keep"


@dataclass
class FamilyPair:
    """Two prefixes whose tables mirror each other within one schema."""

    schema: str
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
    when they sit in the same schema, their names match after the prefix and
    their columns, compared the same way, overlap by Jaccard
    :data:`_MIN_JACCARD` or more. History tables (``*_HIST``) are never twins.

    Families never span schemas: ``live.NEW_X`` and ``archive.OLD_X`` belong to
    independent schemas, and ``NEW_X`` in two schemas are two tables, not one.
    """
    by_schema: dict[str, list[dict[str, Any]]] = {}
    for table in tables:
        by_schema.setdefault((table.get("schema_name") or "").lower(), []).append(table)
    return [
        pair
        for schema, schema_tables in sorted(by_schema.items())
        for pair in _pairs_in_schema(schema, schema_tables)
    ]


def _pairs_in_schema(schema: str, tables: list[dict[str, Any]]) -> list[FamilyPair]:
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
                pairs.append(FamilyPair(schema=schema, prefixes=(a, b), twins=twins))
    return pairs


def keep_semantic_fks_in_family(
    database_name: str,
    pairs: list[FamilyPair],
    connector: SQLDatabase | None,
) -> int:
    """Move semantic FKs that cross into the other family onto the twin.

    An edge whose source and target sit in the two families of one pair, in
    that pair's schema, is redirected to the same-named column of the target's twin in the source's
    family, when that column covers the source's values at least as well.
    It is dropped when the twin is missing or covers less, and kept as it is
    when the twin could not be checked. Returns the number moved or dropped.
    """
    family_of = {
        str(table["id"]): (index, prefix)
        for index, pair in enumerate(pairs)
        for twin in pair.twins.values()
        for prefix, table in twin.items()
    }
    handled = 0
    for row in fetch_semantic_fk_join_groups(database_name):
        source_prefix = prefix_of(row["source_table_name"])
        target = family_of.get(str(row["target_table_id"]))
        if target is None or source_prefix is None:
            continue
        index, target_prefix = target
        pair = pairs[index]
        if (
            source_prefix not in pair.prefixes
            or source_prefix == target_prefix
            or (row.get("source_schema_name") or "").lower() != pair.schema
        ):
            continue
        outcome, detail = _move_to_twin(row, pair, source_prefix, connector)
        edge = (
            f"{row['source_table_name']}.{row['source_column_name']} -> "
            f"{row['target_table_name']}.{row['target_column_name']}"
        )
        if outcome == _Outcome.KEEP:
            logger.warning(
                "semantic FK %s crosses families, kept unchecked: %s", edge, detail
            )
            continue
        delete_semantic_fk(row["source_column_id"], row["target_attribute_id"])
        handled += 1
        if outcome == _Outcome.MOVED:
            logger.info("semantic FK kept in family: %s now -> %s", edge, detail)
        else:
            logger.info("semantic FK dropped, crosses family: %s (%s)", edge, detail)
    return handled


def _move_to_twin(
    row: dict[str, Any],
    pair: FamilyPair,
    source_prefix: str,
    connector: SQLDatabase | None,
) -> tuple[_Outcome, str]:
    """Point *row*'s source column at the twin column, with why or where.

    ``DROP`` when the twin was looked for and is missing or covers less;
    ``KEEP`` when it could not be checked, so the edge is not lost to an error.
    A twin that is the source table itself is refused: semantic FK resolution
    never links a table to itself either.
    """
    target_name = row["target_table_name"]
    base = target_name.upper()[len(prefix_of(target_name) or "") :]
    twin_table = pair.twins.get(base, {}).get(source_prefix)
    if twin_table is None:
        return _Outcome.DROP, "no twin table"
    if str(twin_table["id"]) == str(row["source_table_id"]):
        return _Outcome.DROP, "the twin is the source table"
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
        return _Outcome.DROP, "no twin column"
    if connector is None:
        return _Outcome.KEEP, "no connector to compare coverage"
    column_id = find_column_id_in_table(str(twin_table["id"]), twin_column)
    attr_id = find_column_attribute_by_column_id(column_id) if column_id else None
    if not attr_id:
        return _Outcome.KEEP, "twin column has no attribute"
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
    if old is None or new is None:
        return _Outcome.KEEP, "coverage check failed"
    if not new or new < old:
        return _Outcome.DROP, f"twin covers {new} source values, the original {old}"
    if not merge_semantic_fk(row["source_column_id"], attr_id):
        return _Outcome.KEEP, "the moved edge could not be written"
    set_semantic_fk_join_default(
        row["source_table_id"], attr_id, row["source_column_id"]
    )
    return _Outcome.MOVED, f"{twin_table['name']}.{twin_column}"


def _coverage(
    connector: SQLDatabase,
    source: tuple[str | None, str, str],
    target: tuple[str | None, str, str],
) -> int | None:
    """Distinct source values found in the target column; ``None`` on error.

    Capped at :data:`_COVERAGE_TIMEOUT_SECONDS` where the connector supports
    it. ``is True`` because a test double answers every attribute with
    something truthy, and passing ``timeout_s`` to a connector that does not
    take it would be a TypeError.
    """
    dialect = getattr(connector, "dialect", None)
    sc = quoted_identifier(source[2], dialect)
    tc = quoted_identifier(target[2], dialect)
    sql = (
        f"SELECT COUNT(DISTINCT {sc}) AS n "
        f"FROM {connector.qualify(source[0], source[1])} "
        f"WHERE {sc} IN (SELECT {tc} FROM {connector.qualify(target[0], target[1])})"
    )
    kwargs: dict[str, Any] = {}
    if getattr(connector, "supports_statement_timeout", False) is True:
        kwargs["timeout_s"] = _COVERAGE_TIMEOUT_SECONDS
    try:
        return int(connector.execute(sql, **kwargs).iloc[0, 0])
    except StatementTimeout as exc:
        logger.warning(
            "coverage check for %s.%s timed out: %s", source[1], source[2], exc
        )
        return None
    except Exception:
        logger.info(
            "coverage check skipped for %s.%s", source[1], source[2], exc_info=True
        )
        return None


def resolve_table_families(database_name: str, connector: SQLDatabase | None) -> int:
    """Keep semantic FKs inside their table family. Returns pairs found."""
    pairs = detect_family_pairs(fetch_table_column_names(database_name))
    for pair in pairs:
        logger.info(
            "table family %s: %s / %s, %d twins",
            pair.schema,
            *pair.prefixes,
            len(pair.twins),
        )
    if pairs:
        keep_semantic_fks_in_family(database_name, pairs, connector)
    return len(pairs)
