# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Detect tables whose undirected edges are stored in both orientations."""

from __future__ import annotations

import logging
from collections import defaultdict
from itertools import combinations
from typing import Any

from sqlglot import exp

from auto_ontology.connectors import get_connectors
from auto_ontology.connectors.base import SQLDatabase
from auto_ontology.dal.attributes import fetch_semantic_fk_join_groups
from auto_ontology.dal.datasources import (
    fetch_table_headers_for_database,
    replace_relationship_semantics,
)
from auto_ontology.retrieval.text_to_sql.db_probe.executor import ProbeExecutor

logger = logging.getLogger(__name__)

RELATIONSHIP_TYPE = "reciprocal_undirected_edge"
_MIN_NON_SELF_ORIENTATIONS = 6


def detect_reciprocal_edges(database_name: str) -> int:
    """Verify and persist reciprocal-edge facts for one database."""
    connector = _resolve_connector(database_name)
    if connector is None:
        logger.warning(
            "Reciprocal-edge detection skipped: no connector for database %r",
            database_name,
        )
        return 0

    tables = {
        str(table["id"]): table
        for table in fetch_table_headers_for_database(database_name)
    }
    grouped_roles: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    roles_by_table: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for role in fetch_semantic_fk_join_groups(database_name):
        roles_by_table[str(role["source_table_id"])].append(role)
        grouped_roles[
            (
                str(role["source_table_id"]),
                str(role["target_table_id"]),
                str(role["target_column_id"]),
            )
        ].append(role)

    candidates: list[
        tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[str]]
    ] = []
    for (table_id, _, _), roles in grouped_roles.items():
        table = tables.get(table_id)
        if table is None:
            continue
        for left, right in combinations(roles, 2):
            endpoints = {
                str(left["source_column_name"]),
                str(right["source_column_name"]),
            }
            if len(endpoints) != 2:
                continue
            edge_keys = [
                str(column)
                for column in (table.get("pk") or [])
                if str(column) not in endpoints
            ]
            if not edge_keys:
                endpoint_target = (
                    str(left["target_table_id"]),
                    str(left["target_column_id"]),
                )
                edge_keys = list(
                    dict.fromkeys(
                        str(role["source_column_name"])
                        for role in roles_by_table[table_id]
                        if str(role["source_column_name"]) not in endpoints
                        and (
                            str(role["target_table_id"]),
                            str(role["target_column_id"]),
                        )
                        != endpoint_target
                    )
                )
            if not edge_keys:
                continue
            candidates.append((table, left, right, edge_keys))

    facts_by_table: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with ProbeExecutor(connector, max_calls=max(1, len(candidates))) as executor:
        for table, left, right, edge_keys in candidates:
            sql = _verification_sql(
                connector,
                str(table.get("schema_name") or "") or None,
                str(table["name"]),
                str(left["source_column_name"]),
                str(right["source_column_name"]),
                edge_keys,
            )
            result = executor.run(sql, purpose="reciprocal edge verification")
            if not result["ok"] or not result.get("rows"):
                continue
            stats = result["rows"][0]
            if not _is_verified(stats):
                continue
            facts_by_table[str(table["id"])].append(
                {
                    "relationship_type": RELATIONSHIP_TYPE,
                    "table": str(table["name"]),
                    "endpoint_columns": [
                        str(left["source_column_name"]),
                        str(right["source_column_name"]),
                    ],
                    "edge_key_columns": edge_keys,
                    "target_table": str(left["target_table_name"]),
                    "target_column": str(left["target_column_name"]),
                    "reciprocal_rows_verified": True,
                }
            )

    replace_relationship_semantics(database_name, dict(facts_by_table))
    count = sum(len(facts) for facts in facts_by_table.values())
    logger.info(
        "Verified %d reciprocal-edge relationship fact(s) in database %r",
        count,
        database_name,
    )
    return count


def _is_verified(stats: dict[str, Any]) -> bool:
    try:
        return (
            int(stats.get("orientation_count") or 0) >= _MIN_NON_SELF_ORIENTATIONS
            and int(stats.get("missing_reciprocal_count") or 0) == 0
            and int(stats.get("malformed_edge_count") or 0) == 0
        )
    except (TypeError, ValueError):
        return False


def _verification_sql(
    connector: SQLDatabase,
    schema_name: str | None,
    table_name: str,
    endpoint_a: str,
    endpoint_b: str,
    edge_keys: list[str],
) -> str:
    """Build a read-only full-scope symmetry check."""
    dialect = getattr(connector, "dialect", None)
    table = connector.qualify(schema_name, table_name)

    def ident(name: str) -> str:
        try:
            return exp.to_identifier(name, quoted=True).sql(dialect=dialect or None)
        except ValueError:
            return exp.to_identifier(name, quoted=True).sql()

    a = ident(endpoint_a)
    b = ident(endpoint_b)
    keys = [ident(key) for key in edge_keys]
    projected = ", ".join([f"{a} AS endpoint_a", f"{b} AS endpoint_b", *keys])
    non_null = " AND ".join(
        [
            f"{a} IS NOT NULL",
            f"{b} IS NOT NULL",
            *(f"{key} IS NOT NULL" for key in keys),
        ]
    )
    reverse_key_match = " AND ".join(f"r.{key} = e.{key}" for key in keys)
    grouped_keys = ", ".join(keys)

    return f"""\
WITH reciprocal_rows AS (
  SELECT {projected}
  FROM {table}
  WHERE {non_null} AND {a} <> {b}
),
edge_groups AS (
  SELECT {grouped_keys}, COUNT(*) AS orientation_count
  FROM reciprocal_rows
  GROUP BY {grouped_keys}
)
SELECT
  (SELECT COUNT(*) FROM reciprocal_rows) AS orientation_count,
  (
    SELECT COUNT(*)
    FROM reciprocal_rows e
    WHERE NOT EXISTS (
      SELECT 1
      FROM reciprocal_rows r
      WHERE r.endpoint_a = e.endpoint_b
        AND r.endpoint_b = e.endpoint_a
        AND {reverse_key_match}
    )
  ) AS missing_reciprocal_count,
  (
    SELECT COUNT(*)
    FROM edge_groups
    WHERE orientation_count <> 2
  ) AS malformed_edge_count
"""


def _resolve_connector(database_name: str) -> SQLDatabase | None:
    target = database_name.casefold()
    return next(
        (
            connector
            for connector in get_connectors()
            if str(getattr(connector, "database_name", "")).casefold() == target
        ),
        None,
    )


__all__ = ["RELATIONSHIP_TYPE", "detect_reciprocal_edges"]
