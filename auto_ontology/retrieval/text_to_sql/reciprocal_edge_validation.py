# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic SQL checks for verified reciprocal-row relationships."""

from __future__ import annotations

import re
from typing import Any

import sqlglot
from sqlglot import exp

_RELATIONSHIP_TYPE = "reciprocal_undirected_edge"
_COUNT_INTENT = re.compile(
    r"\b(relationship|relationships|edge|edges|connection|connections|"
    r"participating entities|participants|unique entities|distinct entities)\b",
    re.IGNORECASE,
)
_PHYSICAL_ROW_INTENT = re.compile(
    r"\b(physical rows?|stored rows?|records?|orientations?)\b", re.IGNORECASE
)


def find_reciprocal_edge_violations(
    sql: str,
    question: str,
    tables: list[dict[str, Any]],
    dialect: str | None = None,
    *,
    previously_rejected_sql: str | None = None,
) -> list[str]:
    """Return only structurally certain reciprocal-edge violations."""
    try:
        tree = sqlglot.parse_one(sql, read=dialect or None)
    except (sqlglot.errors.ParseError, ValueError):
        return []

    if previously_rejected_sql and _canonical_sql(
        previously_rejected_sql, dialect
    ) == _canonical_sql(sql, dialect):
        return [
            "Reconstruction returned the same SQL previously rejected for violating "
            "a verified reciprocal-row relationship representation."
        ]

    violations: list[str] = []
    for table, fact in _verified_facts(tables):
        aliases = _table_aliases(tree, table)
        if not aliases:
            continue
        endpoints = tuple(str(v).casefold() for v in fact["endpoint_columns"])
        edge_keys = {str(v).casefold() for v in fact.get("edge_key_columns") or []}
        if len(endpoints) != 2:
            continue

        explicit_roles = _explicitly_requests_both_roles(
            question, fact["endpoint_columns"]
        )
        projected_endpoints = _projected_endpoint_names(tree, aliases, endpoints)
        if (
            len(projected_endpoints) == 2
            and not explicit_roles
            and not _canonical_pair(tree, aliases, endpoints)
            and not _groups_by_edge_key(tree, aliases, edge_keys)
        ):
            violations.append(
                "Both reciprocal endpoint columns are projected as separate outputs, "
                "but the question does not explicitly request two endpoint roles. "
                "Project one endpoint for a flat entity list, or canonicalize an "
                "unordered pair."
            )

        duplicated_attribute = _duplicated_target_projection(tree, aliases, endpoints)
        if duplicated_attribute and not explicit_roles:
            violations.append(
                "The same target attribute is projected twice through aliases joined "
                "to opposite reciprocal endpoints. Reciprocal rows already enumerate "
                "both roles; project the attribute through one endpoint only."
            )

        if _has_unsafe_physical_count(tree, question, aliases, endpoints, edge_keys):
            violations.append(
                "The query counts physical reciprocal rows at a relationship or "
                "participating-entity grain, which doubles undirected edges. Count "
                "the verified edge key, a canonical unordered pair, or distinct "
                "entity IDs from one endpoint."
            )

    return list(dict.fromkeys(violations))


def _verified_facts(
    tables: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    return [
        (table, fact)
        for table in tables
        for fact in (table.get("relationship_semantics") or [])
        if fact.get("relationship_type") == _RELATIONSHIP_TYPE
        and fact.get("reciprocal_rows_verified") is True
        and len(fact.get("endpoint_columns") or []) == 2
    ]


def _table_aliases(tree: exp.Expression, table: dict[str, Any]) -> set[str]:
    name = str(table.get("name") or "").casefold()
    schema = str(table.get("schema_name") or "").casefold()
    aliases: set[str] = set()
    for node in tree.find_all(exp.Table):
        if str(node.name or "").casefold() != name:
            continue
        if schema and node.db and str(node.db).casefold() != schema:
            continue
        aliases.add(str(node.alias_or_name).casefold())
    return aliases


def _projection_expressions(tree: exp.Expression) -> list[exp.Expression]:
    select = tree.find(exp.Select)
    return list(select.expressions) if select is not None else []


def _projected_endpoint_names(
    tree: exp.Expression,
    aliases: set[str],
    endpoints: tuple[str, str],
) -> set[str]:
    found: set[str] = set()
    for projection in _projection_expressions(tree):
        for column in projection.find_all(exp.Column):
            if (
                _column_on_alias(column, aliases)
                and column.name.casefold() in endpoints
            ):
                found.add(column.name.casefold())
    return found


def _duplicated_target_projection(
    tree: exp.Expression,
    edge_aliases: set[str],
    endpoints: tuple[str, str],
) -> bool:
    target_alias_to_endpoint: dict[str, str] = {}
    for equality in tree.find_all(exp.EQ):
        left, right = equality.this, equality.expression
        if not isinstance(left, exp.Column) or not isinstance(right, exp.Column):
            continue
        for edge_col, target_col in ((left, right), (right, left)):
            if (
                _column_on_alias(edge_col, edge_aliases)
                and edge_col.name.casefold() in endpoints
                and target_col.table
            ):
                target_alias_to_endpoint[target_col.table.casefold()] = (
                    edge_col.name.casefold()
                )

    projected: dict[str, set[str]] = {}
    for projection in _projection_expressions(tree):
        columns = list(projection.find_all(exp.Column))
        if len(columns) != 1:
            continue
        column = columns[0]
        endpoint = target_alias_to_endpoint.get(column.table.casefold())
        if endpoint:
            projected.setdefault(column.name.casefold(), set()).add(endpoint)
    return any(len(endpoint_roles) == 2 for endpoint_roles in projected.values())


def _has_unsafe_physical_count(
    tree: exp.Expression,
    question: str,
    aliases: set[str],
    endpoints: tuple[str, str],
    edge_keys: set[str],
) -> bool:
    counts = list(tree.find_all(exp.Count))
    if not counts:
        return False
    if _PHYSICAL_ROW_INTENT.search(question):
        return False
    if not (
        _COUNT_INTENT.search(question)
        or _or_filters_both_endpoints(tree, aliases, endpoints)
    ):
        return False
    if _canonical_pair(tree, aliases, endpoints) or _groups_by_edge_key(
        tree, aliases, edge_keys
    ):
        return False
    for count in counts:
        if count.find(exp.Distinct):
            continue
        argument_columns = list(count.find_all(exp.Column))
        if not argument_columns or any(
            _column_on_alias(column, aliases) for column in argument_columns
        ):
            return True
    return False


def _or_filters_both_endpoints(
    tree: exp.Expression,
    aliases: set[str],
    endpoints: tuple[str, str],
) -> bool:
    for node in tree.find_all(exp.Or):
        names = {
            column.name.casefold()
            for column in node.find_all(exp.Column)
            if _column_on_alias(column, aliases)
        }
        if set(endpoints) <= names:
            return True
    return False


def _groups_by_edge_key(
    tree: exp.Expression, aliases: set[str], edge_keys: set[str]
) -> bool:
    group = tree.find(exp.Group)
    if group is None or not edge_keys:
        return False
    grouped = {
        column.name.casefold()
        for column in group.find_all(exp.Column)
        if _column_on_alias(column, aliases)
    }
    return edge_keys <= grouped


def _canonical_pair(
    tree: exp.Expression,
    aliases: set[str],
    endpoints: tuple[str, str],
) -> bool:
    has_least = any(
        {column.name.casefold() for column in node.find_all(exp.Column)}
        >= set(endpoints)
        for node in tree.find_all(exp.Least)
    )
    has_greatest = any(
        {column.name.casefold() for column in node.find_all(exp.Column)}
        >= set(endpoints)
        for node in tree.find_all(exp.Greatest)
    )
    if has_least and has_greatest:
        return True
    for kind in (exp.LT, exp.LTE, exp.GT, exp.GTE):
        for comparison in tree.find_all(kind):
            names = {
                column.name.casefold()
                for column in comparison.find_all(exp.Column)
                if _column_on_alias(column, aliases)
            }
            if names == set(endpoints):
                return True
    return False


def _explicitly_requests_both_roles(question: str, endpoint_columns: list[Any]) -> bool:
    normalized = question.casefold()
    endpoint_names = [
        str(name).casefold().replace("_", " ") for name in endpoint_columns
    ]
    if all(name and name in normalized for name in endpoint_names):
        return True
    role_pairs = (
        ("source", "target"),
        ("parent", "child"),
        ("origin", "destination"),
        ("sender", "recipient"),
        ("from", "to"),
    )
    return any(
        re.search(rf"\b{left}\b", normalized) and re.search(rf"\b{right}\b", normalized)
        for left, right in role_pairs
    ) or bool(re.search(r"\bboth\s+(?:endpoint|role|side)s?\b", normalized))


def _column_on_alias(column: exp.Column, aliases: set[str]) -> bool:
    return bool(column.table) and column.table.casefold() in aliases


def _canonical_sql(sql: str, dialect: str | None) -> str:
    try:
        return sqlglot.parse_one(sql, read=dialect or None).sql(
            dialect=dialect or None, pretty=False
        )
    except (sqlglot.errors.ParseError, ValueError):
        return " ".join(sql.split())


__all__ = ["find_reciprocal_edge_violations"]
