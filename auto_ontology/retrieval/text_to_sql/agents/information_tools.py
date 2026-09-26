# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded, read-only catalog tools used by the information metadata agent."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from auto_ontology.dal.datasources import (
    fetch_columns_for_table,
    fetch_databases,
    fetch_join_neighbors,
    fetch_parent_table_id_for_column,
    fetch_schemas_for_database,
    fetch_table_by_id,
    fetch_table_context,
    fetch_tables_for_schema,
    find_column_id_by_table_and_name,
    find_table_id_by_name,
)
from auto_ontology.dal.exploration import (
    fetch_column_attribute_exploration_details,
    fetch_column_exploration_details,
)
from auto_ontology.dal.terms import fetch_terms_and_attributes_for_table
from auto_ontology.semantic.constants import REL_SEMANTIC_FK

MAX_DATASETS = 25
MAX_SCHEMAS = 25
MAX_TABLES = 50
MAX_COLUMNS = 100
MAX_CONNECTIONS = 50
MAX_ATTRIBUTE_COLUMNS = 50


class InformationToolName(StrEnum):
    """Tools exposed to the information agent."""

    GET_DATASET = "get_dataset"
    GET_TABLE = "get_table"
    GET_COLUMN = "get_column"
    GET_TABLE_SEMANTIC_FKS = "get_table_semantic_fks"


def _safe(value: Any) -> Any:
    """Convert DAL values to JSON-serializable primitives."""
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_safe(item) for item in value]
    if isinstance(value, (UUID, date, datetime)):
        return str(value)
    return value


def _ok(data: Any, *, truncated: bool = False) -> dict[str, Any]:
    return {"ok": True, "data": _safe(data), "truncated": truncated}


def _error(code: str, message: str) -> dict[str, Any]:
    return {"ok": False, "error": {"code": code, "message": message}}


def _match_record(
    records: list[dict[str, Any]], identifier: str
) -> list[dict[str, Any]]:
    needle = identifier.strip().casefold()
    return [
        record
        for record in records
        if needle
        in {
            str(record.get("id") or "").casefold(),
            str(record.get("name") or "").casefold(),
        }
    ]


def _resolve_table(
    *,
    table_id: str | None = None,
    table_name: str | None = None,
    database_name: str | None = None,
) -> tuple[str | None, dict[str, Any] | None]:
    if table_id:
        table = fetch_table_by_id(table_id)
        if not table:
            return None, _error("not_found", f"Table id {table_id!r} was not found.")
        return table_id, None
    if not table_name:
        return None, _error("invalid_arguments", "Provide table_id or table_name.")
    resolved = find_table_id_by_name(table_name, database_name=database_name)
    if resolved:
        return resolved, None
    scope = f" in dataset {database_name!r}" if database_name else ""
    return None, _error(
        "not_found_or_ambiguous",
        f"Table {table_name!r}{scope} was not found or is ambiguous.",
    )


def get_dataset(
    *,
    dataset_id: str | None = None,
    dataset_name: str | None = None,
    **_: Any,
) -> dict[str, Any]:
    """Return one dataset and a bounded hierarchy of its schemas and tables."""
    identifier = dataset_id or dataset_name
    if not identifier:
        return _error("invalid_arguments", "Provide dataset_id or dataset_name.")

    databases = fetch_databases()
    matches = _match_record(databases, identifier)
    if not matches:
        return _error("not_found", f"Dataset {identifier!r} was not found.")
    if len(matches) > 1:
        return _error("ambiguous", f"Dataset {identifier!r} is ambiguous.")

    database = matches[0]
    schemas = fetch_schemas_for_database(str(database["id"]))
    schema_payload: list[dict[str, Any]] = []
    table_count = 0
    truncated = len(schemas) > MAX_SCHEMAS
    for schema in schemas[:MAX_SCHEMAS]:
        tables = fetch_tables_for_schema(
            str(schema["id"]), database_name=str(database.get("name") or "")
        )
        remaining = max(0, MAX_TABLES - table_count)
        shown_tables = tables[:remaining]
        table_count += len(shown_tables)
        schema_payload.append({**schema, "tables": shown_tables})
        if len(tables) > remaining:
            truncated = True
        if table_count >= MAX_TABLES:
            if len(schema_payload) < len(schemas):
                truncated = True
            break

    return _ok(
        {
            "dataset": database,
            "schemas": schema_payload,
            "limits": {"schemas": MAX_SCHEMAS, "tables": MAX_TABLES},
        },
        truncated=truncated,
    )


def get_table(
    *,
    table_id: str | None = None,
    table_name: str | None = None,
    database_name: str | None = None,
    **_: Any,
) -> dict[str, Any]:
    """Return catalog, semantic, column, and connection metadata for one table."""
    resolved, error = _resolve_table(
        table_id=table_id, table_name=table_name, database_name=database_name
    )
    if error:
        return error
    assert resolved is not None

    table = fetch_table_by_id(resolved)
    table_columns = fetch_columns_for_table(
        resolved, skip=0, limit=MAX_COLUMNS + 1
    ) or {"columns": []}
    columns = list(table_columns.get("columns") or [])
    neighbors = fetch_join_neighbors(resolved)
    terms, attributes = fetch_terms_and_attributes_for_table(resolved)
    truncated = (
        len(columns) > MAX_COLUMNS
        or len(neighbors) > MAX_CONNECTIONS
        or len(attributes) > MAX_COLUMNS
    )
    return _ok(
        {
            "table": table,
            "columns": columns[:MAX_COLUMNS],
            "connected_tables": neighbors[:MAX_CONNECTIONS],
            "terms": terms,
            "column_attributes": attributes[:MAX_COLUMNS],
            "limits": {
                "columns": MAX_COLUMNS,
                "connections": MAX_CONNECTIONS,
                "column_attributes": MAX_COLUMNS,
            },
        },
        truncated=truncated,
    )


def get_column(
    *,
    column_id: str | None = None,
    column_name: str | None = None,
    table_id: str | None = None,
    table_name: str | None = None,
    database_name: str | None = None,
    **_: Any,
) -> dict[str, Any]:
    """Return one column's catalog identity and semantic calculation metadata."""
    resolved_column_id = column_id
    resolved_table_id = table_id
    if resolved_column_id:
        resolved_table_id = fetch_parent_table_id_for_column(resolved_column_id)
        if not resolved_table_id:
            return _error(
                "not_found", f"Column id {resolved_column_id!r} was not found."
            )
    else:
        if not column_name:
            return _error("invalid_arguments", "Provide column_id or column_name.")
        if not table_name and table_id:
            table = fetch_table_by_id(table_id)
            table_name = str((table or {}).get("name") or "") or None
        if not table_name:
            return _error(
                "invalid_arguments",
                "column_name requires table_name or table_id.",
            )
        resolved_column_id = find_column_id_by_table_and_name(
            table_name,
            column_name,
            database_name=database_name,
        )
        if not resolved_column_id:
            return _error(
                "not_found_or_ambiguous",
                f"Column {table_name}.{column_name} was not found or is ambiguous.",
            )
        resolved_table_id = fetch_parent_table_id_for_column(resolved_column_id)

    assert resolved_column_id is not None
    assert resolved_table_id is not None
    table_columns = fetch_columns_for_table(
        resolved_table_id, skip=0, limit=MAX_COLUMNS + 1
    ) or {"columns": []}
    columns = list(table_columns.get("columns") or [])
    column = next(
        (item for item in columns if str(item.get("id")) == resolved_column_id),
        None,
    )
    if column is None:
        return _error("not_found", f"Column id {resolved_column_id!r} was not found.")

    details = fetch_column_exploration_details(resolved_column_id) or {}
    table = fetch_table_by_id(resolved_table_id)
    neighbors = fetch_join_neighbors(resolved_table_id)
    return _ok(
        {
            "column": column,
            "table": table,
            "semantic_details": details,
            "connected_tables": neighbors[:MAX_CONNECTIONS],
            "limits": {"connections": MAX_CONNECTIONS},
        },
        truncated=(len(columns) > MAX_COLUMNS or len(neighbors) > MAX_CONNECTIONS),
    )


def get_table_semantic_fks(
    *,
    table_id: str | None = None,
    table_name: str | None = None,
    database_name: str | None = None,
    **_: Any,
) -> dict[str, Any]:
    """Return semantic-FK attributes and the columns/tables they connect."""
    resolved, error = _resolve_table(
        table_id=table_id, table_name=table_name, database_name=database_name
    )
    if error:
        return error
    assert resolved is not None

    context = fetch_table_context(resolved)
    columns = list(context.get("columns") or [])
    connections: list[dict[str, Any]] = []
    truncated = len(columns) > MAX_COLUMNS
    for column in columns[:MAX_COLUMNS]:
        column_id = str(column.get("id") or "")
        if not column_id:
            continue
        details = fetch_column_exploration_details(column_id) or {}
        attribute = details.get("column_attribute")
        if not isinstance(attribute, dict):
            continue
        if attribute.get("relationship_type") != REL_SEMANTIC_FK:
            continue
        attribute_id = str(attribute.get("id") or "")
        linked = (
            fetch_column_attribute_exploration_details(attribute_id)
            if attribute_id
            else {}
        ) or {}
        linked_columns = list(linked.get("columns") or [])
        if len(linked_columns) > MAX_ATTRIBUTE_COLUMNS:
            truncated = True
        connections.append(
            {
                "source_column": column,
                "attribute": attribute,
                "connected_columns": linked_columns[:MAX_ATTRIBUTE_COLUMNS],
            }
        )
        if len(connections) >= MAX_CONNECTIONS:
            truncated = True
            break

    return _ok(
        {
            "table": fetch_table_by_id(resolved),
            "semantic_foreign_keys": connections,
            "limits": {
                "connections": MAX_CONNECTIONS,
                "columns_per_attribute": MAX_ATTRIBUTE_COLUMNS,
            },
        },
        truncated=truncated,
    )


INFORMATION_TOOLS: dict[InformationToolName, Callable[..., dict[str, Any]]] = {
    InformationToolName.GET_DATASET: get_dataset,
    InformationToolName.GET_TABLE: get_table,
    InformationToolName.GET_COLUMN: get_column,
    InformationToolName.GET_TABLE_SEMANTIC_FKS: get_table_semantic_fks,
}


def run_information_tool(
    tool_name: InformationToolName | str, arguments: dict[str, Any]
) -> dict[str, Any]:
    """Dispatch a named read-only metadata tool."""
    try:
        name = InformationToolName(tool_name)
    except ValueError:
        return _error("unknown_tool", f"Unknown information tool {tool_name!r}.")
    return INFORMATION_TOOLS[name](**arguments)


__all__ = [
    "INFORMATION_TOOLS",
    "InformationToolName",
    "get_column",
    "get_dataset",
    "get_table",
    "get_table_semantic_fks",
    "run_information_tool",
]
