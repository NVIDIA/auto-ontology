# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Detect link tables that store every relationship in both directions.

A table such as ``connected(atom_id, atom_id2, bond_id)`` can hold each link
twice, once as (A, B) and once as (B, A). Joining the target on one of the two
columns then already returns both members of a link, and a SQL generator that
joins the target twice returns every pair mirrored. Nothing in the schema says
which kind of table this is — a directed table (``follows``, ``reports_to``)
has the same shape — so the decision is made from the data: a pair is
symmetric only when at least 99% of at least 20 rows have their mirror.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Callable, Sequence

import pandas as pd

from auto_ontology.utils.sql_identifiers import quoted_identifier

if TYPE_CHECKING:
    from auto_ontology.connectors.base import SQLDatabase

logger = logging.getLogger(__name__)

MIN_PROBED_ROWS = 20
MIN_MIRRORED_RATIO = 0.99
# Rows checked for a mirror. Each lookup can scan the whole table when the
# pair is not indexed, so probing every row would cost rows squared.
PROBE_SAMPLE_ROWS = 1000


def candidate_column_pairs(
    table: dict[str, Any],
    columns: Sequence[dict[str, Any]],
    declared_fks: Sequence[dict[str, Any]],
) -> list[tuple[str, str]]:
    """Pairs of columns that are the only two referencing one column of another
    table, in column order.

    Only a declared FK knows its target, so only declared FKs are paired. Three
    or more columns on one target (``home_player_1`` .. ``home_player_11``) are
    slots rather than the two ends of a link, and are never probed.
    """
    order = {col.get("name"): i for i, col in enumerate(columns)}
    own = {str(table.get("id") or ""), str(table.get("name") or "")} - {""}
    groups: dict[tuple[str, str], set[str]] = defaultdict(set)
    for fk in declared_fks:
        source = fk.get("source_column")
        target_table = fk.get("target_table_id") or fk.get("target_table")
        if source not in order or not target_table or not fk.get("target_column"):
            continue
        targets = {
            str(fk.get("target_table_id") or ""),
            str(fk.get("target_table") or ""),
        }
        if targets & own:
            continue
        groups[(str(target_table), str(fk["target_column"]))].add(source)

    pairs: list[tuple[str, str]] = []
    for members in groups.values():
        if len(members) == 2:
            first, second = sorted(members, key=order.__getitem__)
            pairs.append((first, second))
    return sorted(pairs, key=lambda pair: (order[pair[0]], order[pair[1]]))


def _key_columns(
    pair: tuple[str, str],
    column_names: Sequence[str],
    pk: Sequence[str],
    fk_names: set[str],
) -> list[str]:
    """The table's other key columns, which a mirrored row must also match.

    A single-column primary key outside the pair is a row id, unique to each
    stored direction, so it can never match its mirror and is left out.
    """
    pk_keys = set(pk) if len(pk) > 1 or set(pk) & set(pair) else set()
    keys = (pk_keys | fk_names) - set(pair)
    return [name for name in column_names if name in keys]


def mirror_probe_sql(
    qualified_table: str,
    pair: tuple[str, str],
    key_columns: Sequence[str],
    dialect: str | None,
) -> str:
    """One read-only query over a sample of rows: how many were checked, and
    how many have their mirror anywhere in the table."""

    def q(name: str) -> str:
        return quoted_identifier(name, dialect)

    c1, c2 = (q(name) for name in pair)
    conditions = [f"r.{c1} = t.{c2}", f"r.{c2} = t.{c1}"]
    for key in key_columns:
        k = q(key)
        conditions.append(f"(r.{k} = t.{k} OR (r.{k} IS NULL AND t.{k} IS NULL))")
    return (
        "SELECT COUNT(*) AS total, "
        "SUM(CASE WHEN EXISTS ("
        f"SELECT 1 FROM {qualified_table} r WHERE {' AND '.join(conditions)}"
        ") THEN 1 ELSE 0 END) AS mirrored "
        f"FROM (SELECT * FROM {qualified_table} LIMIT {PROBE_SAMPLE_ROWS}) t"
    )


def _counts(df: pd.DataFrame | None) -> tuple[int, int]:
    if df is None or df.empty:
        return 0, 0
    total, mirrored = df.iloc[0, 0], df.iloc[0, 1]
    # SUM over zero rows is NULL.
    return (
        0 if pd.isna(total) else int(total),
        0 if pd.isna(mirrored) else int(mirrored),
    )


def detect_symmetric_column_pairs(
    table: dict[str, Any],
    columns: Sequence[dict[str, Any]],
    declared_fks: Sequence[dict[str, Any]],
    suggested_fk_names: set[str],
    connector: "SQLDatabase",
    execute: Callable[["SQLDatabase", str], pd.DataFrame],
) -> list[list[str]]:
    """The ``[c1, c2]`` pairs of *table* whose rows are stored in both directions.

    Returns ``[]`` when no pair qualifies, and also when a probe fails or times
    out: an unproven table is treated as directed.
    """
    table_name = table.get("name") or ""
    pairs = candidate_column_pairs(table, columns, declared_fks)
    if not pairs:
        return []
    dialect = getattr(connector, "dialect", None)
    qualified = connector.qualify(table.get("schema_name"), table_name)
    column_names = [str(col.get("name")) for col in columns if col.get("name")]
    fk_names = {
        str(fk["source_column"]) for fk in declared_fks if fk.get("source_column")
    } | set(suggested_fk_names)
    pk = [str(name) for name in (table.get("pk") or [])]

    symmetric: list[list[str]] = []
    for pair in pairs:
        keys = _key_columns(pair, column_names, pk, fk_names)
        sql = mirror_probe_sql(qualified, pair, keys, dialect)
        try:
            total, mirrored = _counts(execute(connector, sql))
        except Exception as exc:
            logger.info(
                "[%s] symmetric-link probe on (%s, %s) failed (%s) — skipping table",
                table_name,
                *pair,
                type(exc).__name__,
            )
            return []
        if total >= MIN_PROBED_ROWS and mirrored / total >= MIN_MIRRORED_RATIO:
            symmetric.append(list(pair))
        logger.info(
            "[%s] symmetric-link probe on (%s, %s): %d of %d rows mirrored",
            table_name,
            *pair,
            mirrored,
            total,
        )
    return symmetric
