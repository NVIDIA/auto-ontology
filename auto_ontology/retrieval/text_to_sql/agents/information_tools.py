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
    count_columns_for_table,
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
    fetch_sql_attribute_exploration_details,
)
from auto_ontology.dal.sql_attributes import (
    fetch_sql_attributes_by_term_id,
    find_attr_by_name,
    get_full_sql_attribute_by_id,
)
from auto_ontology.dal.terms import (
    count_terms,
    fetch_all_terms,
    fetch_column_attributes_by_term_id,
    fetch_terms_and_attributes_for_table,
    get_full_term_by_id,
)
from auto_ontology.retrieval.data_access.semantic_search import search_semantic_index
from auto_ontology.semantic.constants import (
    LABEL_COLUMN_ATTRIBUTE,
    LABEL_SQL_ATTRIBUTE,
    LABEL_TERM,
    REL_SEMANTIC_FK,
)
from auto_ontology.catalog.constants import Labels

MAX_DATASETS = 25
MAX_SCHEMAS = 25
MAX_TABLES = 50
MAX_COLUMNS = 100
MAX_CONNECTIONS = 50
MAX_ATTRIBUTE_COLUMNS = 50
MAX_TERMS = 50
MAX_TERM_ATTRIBUTES = 100
MAX_SEMANTIC_SEARCH_RESULTS = 10
SEMANTIC_SEARCH_LABELS = frozenset(
    {
        LABEL_TERM,
        LABEL_COLUMN_ATTRIBUTE,
        LABEL_SQL_ATTRIBUTE,
        Labels.CUSTOM_ANALYSIS,
    }
)


class InformationToolName(StrEnum):
    """Tools exposed to the information agent."""

    LIST_DATABASES = "list_databases"
    LIST_SCHEMAS = "list_schemas"
    LIST_TABLES = "list_tables"
    LIST_COLUMNS = "list_columns"
    GET_DATASET = "get_dataset"
    GET_TABLE = "get_table"
    GET_COLUMN = "get_column"
    GET_TABLE_SEMANTIC_FKS = "get_table_semantic_fks"
    GET_TERM = "get_term"
    GET_COLUMN_ATTRIBUTE = "get_column_attribute"
    GET_SQL_ATTRIBUTE = "get_sql_attribute"
    LIST_TERMS = "list_terms"
    SEARCH_SEMANTIC_LAYER = "search_semantic_layer"


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


def list_databases(**_: Any) -> dict[str, Any]:
    """Return the bounded root of the lazy catalog tree."""
    databases = fetch_databases()
    return _ok(
        {
            "databases": databases[:MAX_DATASETS],
            "total": len(databases),
            "limit": MAX_DATASETS,
        },
        truncated=len(databases) > MAX_DATASETS,
    )


def list_schemas(
    *,
    database_id: str | None = None,
    dataset_id: str | None = None,
    **_: Any,
) -> dict[str, Any]:
    """Return schemas directly beneath one database."""
    resolved_id = database_id or dataset_id
    if not resolved_id:
        return _error("invalid_arguments", "Provide database_id.")
    result = fetch_schemas_for_database(resolved_id)
    if result is None:
        return _error(
            "not_found",
            f"Database id {resolved_id!r} was not found or has no schemas.",
        )
    schemas = list(result.get("schemas") or [])
    return _ok(
        {
            "database_id": resolved_id,
            "schemas": schemas[:MAX_SCHEMAS],
            "total": int(result.get("schemas_count") or len(schemas)),
            "limit": MAX_SCHEMAS,
        },
        truncated=len(schemas) > MAX_SCHEMAS,
    )


def list_tables(
    *,
    schema_id: str | None = None,
    database_name: str | None = None,
    **_: Any,
) -> dict[str, Any]:
    """Return tables directly beneath one schema."""
    if not schema_id:
        return _error("invalid_arguments", "Provide schema_id.")
    tables = fetch_tables_for_schema(schema_id, database_name=database_name)
    return _ok(
        {
            "schema_id": schema_id,
            "tables": tables[:MAX_TABLES],
            "total": len(tables),
            "limit": MAX_TABLES,
        },
        truncated=len(tables) > MAX_TABLES,
    )


def list_columns(
    *,
    table_id: str | None = None,
    skip: int = 0,
    limit: int = MAX_COLUMNS,
    **_: Any,
) -> dict[str, Any]:
    """Return one bounded page of columns directly beneath a table."""
    if not table_id:
        return _error("invalid_arguments", "Provide table_id.")
    bounded_skip = max(int(skip), 0)
    bounded_limit = min(max(int(limit), 1), MAX_COLUMNS)
    result = fetch_columns_for_table(
        table_id,
        skip=bounded_skip,
        limit=bounded_limit,
    )
    if result is None:
        return _error("not_found", f"Table id {table_id!r} was not found.")
    total = count_columns_for_table(table_id)
    columns = list(result.get("columns") or [])
    return _ok(
        {
            **result,
            "total": total,
            "skip": bounded_skip,
            "limit": bounded_limit,
        },
        truncated=bounded_skip + len(columns) < total,
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
    schemas_result = fetch_schemas_for_database(str(database["id"])) or {}
    schemas = (
        list(schemas_result.get("schemas") or [])
        if isinstance(schemas_result, dict)
        else list(schemas_result)
    )
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
    column = None
    skip = 0
    while column is None:
        table_columns = fetch_columns_for_table(
            resolved_table_id,
            skip=skip,
            limit=MAX_COLUMNS,
        )
        columns = list((table_columns or {}).get("columns") or [])
        column = next(
            (item for item in columns if str(item.get("id")) == resolved_column_id),
            None,
        )
        if column is not None or len(columns) < MAX_COLUMNS:
            break
        skip += len(columns)
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
        truncated=len(neighbors) > MAX_CONNECTIONS,
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


def _resolve_term(
    *,
    term_id: str | None = None,
    term_name: str | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Resolve one Term by id or exact case-insensitive name."""
    if term_id:
        term = get_full_term_by_id(term_id)
        if term:
            return term, None
        return None, _error("not_found", f"Term id {term_id!r} was not found.")
    if not term_name:
        return None, _error("invalid_arguments", "Provide term_id or term_name.")
    matches = _match_record(
        fetch_all_terms(search=term_name, limit=MAX_TERMS + 1), term_name
    )
    if not matches:
        return None, _error("not_found", f"Term {term_name!r} was not found.")
    if len(matches) > 1:
        return None, _error("ambiguous", f"Term {term_name!r} is ambiguous.")
    term = get_full_term_by_id(str(matches[0]["id"]))
    if not term:
        return None, _error("not_found", f"Term {term_name!r} was not found.")
    return term, None


def list_terms(
    *,
    search: str | None = None,
    limit: int = 25,
    **_: Any,
) -> dict[str, Any]:
    """Return a bounded glossary list with descriptions and synonyms."""
    bounded_limit = min(max(int(limit), 1), MAX_TERMS)
    terms = fetch_all_terms(search=search, limit=bounded_limit + 1)
    total = count_terms(search=search)
    return _ok(
        {
            "terms": terms[:bounded_limit],
            "total": total,
            "limit": bounded_limit,
        },
        truncated=(len(terms) > bounded_limit or total > bounded_limit),
    )


def get_term(
    *,
    term_id: str | None = None,
    term_name: str | None = None,
    **_: Any,
) -> dict[str, Any]:
    """Return one Term plus bounded ColumnAttributes and SqlAttributes."""
    term, error = _resolve_term(term_id=term_id, term_name=term_name)
    if error:
        return error
    assert term is not None
    term = dict(term)
    term_tables = list(term.get("tables") or [])
    term["tables"] = term_tables[:MAX_TABLES]
    resolved_id = str(term["id"])
    column_attributes = fetch_column_attributes_by_term_id(
        resolved_id, limit=MAX_TERM_ATTRIBUTES + 1
    )
    sql_attributes = fetch_sql_attributes_by_term_id(
        resolved_id, limit=MAX_TERM_ATTRIBUTES + 1
    )
    truncated = (
        len(term_tables) > MAX_TABLES
        or len(column_attributes) > MAX_TERM_ATTRIBUTES
        or len(sql_attributes) > MAX_TERM_ATTRIBUTES
    )
    return _ok(
        {
            "term": term,
            "column_attributes": column_attributes[:MAX_TERM_ATTRIBUTES],
            "sql_attributes": sql_attributes[:MAX_TERM_ATTRIBUTES],
            "limits": {"attributes_per_kind": MAX_TERM_ATTRIBUTES},
        },
        truncated=truncated,
    )


def get_column_attribute(
    *,
    column_attribute_id: str | None = None,
    column_attribute_name: str | None = None,
    term_id: str | None = None,
    term_name: str | None = None,
    **_: Any,
) -> dict[str, Any]:
    """Return one ColumnAttribute, its owning Term, and linked columns."""
    resolved_term: dict[str, Any] | None = None
    if term_id or term_name:
        resolved_term, error = _resolve_term(term_id=term_id, term_name=term_name)
        if error:
            return error

    if column_attribute_id and resolved_term is None:
        exploration = (
            fetch_column_attribute_exploration_details(
                column_attribute_id, limit=MAX_ATTRIBUTE_COLUMNS + 1
            )
            or {}
        )
        term = exploration.get("term")
        if isinstance(term, dict) and term.get("id"):
            resolved_term = get_full_term_by_id(str(term["id"]))
    else:
        exploration = {}

    if resolved_term is None:
        if column_attribute_id:
            return _error(
                "not_found",
                f"ColumnAttribute id {column_attribute_id!r} was not found.",
            )
        return _error(
            "invalid_arguments",
            "Provide term_id or term_name when resolving a ColumnAttribute by name.",
        )

    attributes = fetch_column_attributes_by_term_id(
        str(resolved_term["id"]), limit=MAX_TERM_ATTRIBUTES + 1
    )
    identifier = column_attribute_id or column_attribute_name
    if not identifier:
        return _error(
            "invalid_arguments",
            "Provide column_attribute_id or column_attribute_name.",
        )
    matches = _match_record(attributes, identifier)
    if not matches:
        return _error(
            "not_found",
            f"ColumnAttribute {identifier!r} was not found for this Term.",
        )
    if len(matches) > 1:
        return _error("ambiguous", f"ColumnAttribute {identifier!r} is ambiguous.")

    attribute = matches[0]
    attribute_id = str(attribute["id"])
    if not exploration:
        exploration = (
            fetch_column_attribute_exploration_details(
                attribute_id, limit=MAX_ATTRIBUTE_COLUMNS + 1
            )
            or {}
        )
    return _ok(
        {
            "column_attribute": attribute,
            "term": resolved_term,
            "linked_columns": (exploration.get("columns") or [])[
                :MAX_ATTRIBUTE_COLUMNS
            ],
            "linked_columns_total": exploration.get("columns_total") or 0,
        },
        truncated=(
            len(attributes) > MAX_TERM_ATTRIBUTES
            or (exploration.get("columns_total") or 0) > MAX_ATTRIBUTE_COLUMNS
        ),
    )


def get_sql_attribute(
    *,
    sql_attribute_id: str | None = None,
    sql_attribute_name: str | None = None,
    **_: Any,
) -> dict[str, Any]:
    """Return one SqlAttribute with its Term, SQL, and linked statement."""
    resolved_id = sql_attribute_id
    if not resolved_id and sql_attribute_name:
        match = find_attr_by_name(sql_attribute_name, None)
        resolved_id = str(match.get("id")) if match and match.get("id") else None
    if not resolved_id:
        return _error(
            "invalid_arguments",
            "Provide a valid sql_attribute_id or sql_attribute_name.",
        )
    attribute = get_full_sql_attribute_by_id(resolved_id)
    if not attribute:
        return _error(
            "not_found",
            f"SqlAttribute {sql_attribute_id or sql_attribute_name!r} not found.",
        )
    exploration = fetch_sql_attribute_exploration_details(resolved_id) or {}
    return _ok(
        {
            "sql_attribute": attribute,
            "term": exploration.get("term"),
            "sql": exploration.get("sql"),
        }
    )


def search_semantic_layer(
    *,
    query: str | None = None,
    labels: list[str] | None = None,
    database_name: str | None = None,
    limit: int = 5,
    semantic_retriever: Any = None,
    **_: Any,
) -> dict[str, Any]:
    """Search bounded semantic/vector candidates for a metadata concept."""
    if not query or not query.strip():
        return _error("invalid_arguments", "Provide a non-empty query.")
    if semantic_retriever is None:
        return _error("unavailable", "The semantic retriever is not configured.")
    requested_labels = labels or [
        LABEL_TERM,
        LABEL_COLUMN_ATTRIBUTE,
        LABEL_SQL_ATTRIBUTE,
        Labels.CUSTOM_ANALYSIS,
    ]
    invalid = sorted(set(requested_labels) - SEMANTIC_SEARCH_LABELS)
    if invalid:
        return _error(
            "invalid_arguments",
            f"Unsupported semantic labels: {', '.join(invalid)}.",
        )
    bounded_limit = min(max(int(limit), 1), MAX_SEMANTIC_SEARCH_RESULTS)
    rows = search_semantic_index(
        semantic_retriever,
        query.strip(),
        label_filter=requested_labels,
        per_label_k=bounded_limit,
        database_name=database_name,
    )
    return _ok(
        {
            "matches": rows[:MAX_SEMANTIC_SEARCH_RESULTS],
            "allowed_labels": sorted(SEMANTIC_SEARCH_LABELS),
        },
        truncated=len(rows) > MAX_SEMANTIC_SEARCH_RESULTS,
    )


INFORMATION_TOOLS: dict[InformationToolName, Callable[..., dict[str, Any]]] = {
    InformationToolName.LIST_DATABASES: list_databases,
    InformationToolName.LIST_SCHEMAS: list_schemas,
    InformationToolName.LIST_TABLES: list_tables,
    InformationToolName.LIST_COLUMNS: list_columns,
    InformationToolName.GET_DATASET: get_dataset,
    InformationToolName.GET_TABLE: get_table,
    InformationToolName.GET_COLUMN: get_column,
    InformationToolName.GET_TABLE_SEMANTIC_FKS: get_table_semantic_fks,
    InformationToolName.GET_TERM: get_term,
    InformationToolName.GET_COLUMN_ATTRIBUTE: get_column_attribute,
    InformationToolName.GET_SQL_ATTRIBUTE: get_sql_attribute,
    InformationToolName.LIST_TERMS: list_terms,
    InformationToolName.SEARCH_SEMANTIC_LAYER: search_semantic_layer,
}


def run_information_tool(
    tool_name: InformationToolName | str,
    arguments: dict[str, Any],
    *,
    semantic_retriever: Any = None,
) -> dict[str, Any]:
    """Dispatch a named read-only metadata tool."""
    try:
        name = InformationToolName(tool_name)
    except ValueError:
        return _error("unknown_tool", f"Unknown information tool {tool_name!r}.")
    if name == InformationToolName.SEARCH_SEMANTIC_LAYER:
        return INFORMATION_TOOLS[name](
            **arguments, semantic_retriever=semantic_retriever
        )
    return INFORMATION_TOOLS[name](**arguments)


__all__ = [
    "INFORMATION_TOOLS",
    "InformationToolName",
    "get_column",
    "get_column_attribute",
    "get_dataset",
    "get_sql_attribute",
    "get_table",
    "get_table_semantic_fks",
    "get_term",
    "list_columns",
    "list_databases",
    "list_schemas",
    "list_tables",
    "list_terms",
    "run_information_tool",
    "search_semantic_layer",
]
