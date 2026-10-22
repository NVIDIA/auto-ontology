# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded read-only tools for semantic-FK role analysis."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from auto_ontology.dal.datasources import fetch_table_by_id, fetch_table_context
from auto_ontology.dal.exploration import fetch_column_exploration_details
from auto_ontology.dal.terms import fetch_terms_and_attributes_for_table

MAX_COLUMNS = 40
MAX_NEIGHBOR_EDGES = 30
MAX_QUERIES = 8
MAX_SAMPLE_VALUES = 5


class SemanticFkRoleToolName(StrEnum):
    """Tools exposed to the semantic-FK role agent."""

    GET_TABLE_CONTEXT = "get_table_context"
    GET_COLUMN_USAGE = "get_column_usage"
    GET_RELATIONSHIP_NEIGHBORHOOD = "get_relationship_neighborhood"


def _error(code: str, message: str) -> dict[str, Any]:
    return {"ok": False, "error": {"code": code, "message": message}}


def _trim_column(column: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in {
            "id": column.get("id"),
            "name": column.get("name"),
            "data_type": column.get("data_type"),
            "description": column.get("description"),
            "is_nullable": column.get("is_nullable"),
            "is_unique": column.get("is_unique"),
            "sample_values": list(column.get("sample_values") or [])[
                :MAX_SAMPLE_VALUES
            ],
        }.items()
        if value not in (None, "", [])
    }


def get_table_context(table_id: str) -> dict[str, Any]:
    """Return bounded catalog and semantic context for one table."""
    context = fetch_table_context(table_id)
    table = fetch_table_by_id(table_id)
    if not table:
        return _error("not_found", f"Unknown table id: {table_id}")
    terms, attributes = fetch_terms_and_attributes_for_table(table_id)
    columns = [
        _trim_column(column)
        for column in list(context.get("columns") or [])[:MAX_COLUMNS]
        if isinstance(column, dict)
    ]
    return {
        "ok": True,
        "result": {
            "table": {
                key: table.get(key)
                for key in (
                    "id",
                    "name",
                    "description",
                    "database_name",
                    "schema_name",
                    "pk",
                    "is_junction_table",
                )
                if table.get(key) not in (None, "")
            },
            "columns": columns,
            "columns_truncated": len(context.get("columns") or []) > MAX_COLUMNS,
            "terms": terms[:10],
            "column_attributes": attributes[:MAX_COLUMNS],
            "declared_foreign_keys": list(context.get("fks") or [])[:20],
        },
    }


def get_column_usage(column_id: str) -> dict[str, Any]:
    """Return bounded observed-query and relationship context for one column."""
    details = fetch_column_exploration_details(column_id)
    if not details:
        return _error("not_found", f"Unknown column id: {column_id}")
    result = dict(details)
    result["sql_queries"] = list(result.get("sql_queries") or [])[:MAX_QUERIES]
    result["referencing_columns"] = list(result.get("referencing_columns") or [])[
        :MAX_NEIGHBOR_EDGES
    ]
    return {"ok": True, "result": result}


def get_relationship_neighborhood(
    table_id: str,
    all_edges: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return semantic-FK edges touching one table from the current database."""
    edges = [
        edge
        for edge in all_edges
        if table_id in {edge.get("source_table_id"), edge.get("target_table_id")}
    ]
    if not edges:
        return _error("not_found", f"No semantic-FK edges for table id: {table_id}")
    return {
        "ok": True,
        "result": {
            "edges": edges[:MAX_NEIGHBOR_EDGES],
            "truncated": len(edges) > MAX_NEIGHBOR_EDGES,
        },
    }


def run_semantic_fk_role_tool(
    tool: SemanticFkRoleToolName | str,
    arguments: dict[str, Any],
    *,
    all_edges: list[dict[str, Any]],
) -> dict[str, Any]:
    """Dispatch one validated read-only role-analysis tool call."""
    try:
        tool_name = SemanticFkRoleToolName(tool)
    except ValueError:
        return _error("unknown_tool", f"Unknown tool: {tool}")

    if tool_name == SemanticFkRoleToolName.GET_TABLE_CONTEXT:
        table_id = str(arguments.get("table_id") or "")
        if not table_id:
            return _error("invalid_arguments", "table_id is required")
        return get_table_context(table_id)
    if tool_name == SemanticFkRoleToolName.GET_COLUMN_USAGE:
        column_id = str(arguments.get("column_id") or "")
        if not column_id:
            return _error("invalid_arguments", "column_id is required")
        return get_column_usage(column_id)

    table_id = str(arguments.get("table_id") or "")
    if not table_id:
        return _error("invalid_arguments", "table_id is required")
    return get_relationship_neighborhood(table_id, all_edges)


__all__ = [
    "SemanticFkRoleToolName",
    "run_semantic_fk_role_tool",
]
