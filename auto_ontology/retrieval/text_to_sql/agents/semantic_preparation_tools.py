# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded read-only tools for semantic SQL-context preparation."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Callable

from sqlglot import exp

from auto_ontology.catalog.constants import Labels
from auto_ontology.dal.attributes import (
    fetch_attr_column_contexts,
    find_anchor_hub_siblings,
    find_connected_junction_tables,
    find_join_path,
    find_kept_table_bridges,
)
from auto_ontology.retrieval.data_access.relevant_tables import get_relevant_tables
from auto_ontology.retrieval.data_access.semantic_search import search_semantic_index
from auto_ontology.retrieval.text_to_sql.agents.information_tools import (
    get_column_attribute,
    get_table,
)
from auto_ontology.retrieval.text_to_sql.db_probe.executor import ProbeExecutor
from auto_ontology.semantic.constants import (
    LABEL_COLUMN_ATTRIBUTE,
    LABEL_SQL_ATTRIBUTE,
    LABEL_TERM,
)
from auto_ontology.utils.sql_identifiers import qualified_name, quoted_identifier

MAX_SEARCH_RESULTS = 10
MAX_VALUE_SUSPECTS = 12
MAX_CONNECTION_TABLES = 20
_SEMANTIC_LABELS = frozenset(
    {
        LABEL_TERM,
        LABEL_COLUMN_ATTRIBUTE,
        LABEL_SQL_ATTRIBUTE,
        Labels.CUSTOM_ANALYSIS,
    }
)
_TEXT_TYPE_MARKERS = (
    "char",
    "text",
    "string",
    "clob",
    "varchar",
    "nvarchar",
    "enum",
)


class SemanticPreparationToolName(StrEnum):
    """Tools available to the semantic-preparation reasoning loop."""

    SEARCH_SEMANTIC_LAYER = "search_semantic_layer"
    SEARCH_TABLES = "search_tables"
    GET_COLUMN_ATTRIBUTE = "get_column_attribute"
    GET_TABLE = "get_table"
    PROBE_VALUE = "probe_value"
    FIND_JOIN_PATH = "find_join_path"
    DISCOVER_CONNECTIONS = "discover_connections"


def _ok(data: Any, *, truncated: bool = False) -> dict[str, Any]:
    return {"ok": True, "data": data, "truncated": truncated}


def _error(code: str, message: str) -> dict[str, Any]:
    return {"ok": False, "error": {"code": code, "message": message}}


def search_semantic_layer(
    *,
    query: str | None = None,
    labels: list[str] | None = None,
    database_name: str | None = None,
    limit: int = 5,
    semantic_retriever: Any = None,
    **_: Any,
) -> dict[str, Any]:
    """Search the semantic index with explicit label and database bounds."""
    if semantic_retriever is None:
        return _error("unavailable", "The semantic retriever is not configured.")
    if not query or not query.strip():
        return _error("invalid_arguments", "Provide a non-empty query.")
    requested = labels or [
        LABEL_COLUMN_ATTRIBUTE,
        LABEL_SQL_ATTRIBUTE,
        Labels.CUSTOM_ANALYSIS,
        LABEL_TERM,
    ]
    invalid = sorted(set(requested) - _SEMANTIC_LABELS)
    if invalid:
        return _error(
            "invalid_arguments",
            f"Unsupported semantic labels: {', '.join(invalid)}.",
        )
    bounded = min(max(int(limit), 1), MAX_SEARCH_RESULTS)
    rows = search_semantic_index(
        semantic_retriever,
        query.strip(),
        label_filter=requested,
        per_label_k=bounded,
        database_name=database_name,
    )
    return _ok(
        {"matches": rows[:MAX_SEARCH_RESULTS]},
        truncated=len(rows) > MAX_SEARCH_RESULTS,
    )


def search_tables(
    *,
    query: str | None = None,
    database_name: str | None = None,
    limit: int = 5,
    data_retriever: Any = None,
    **_: Any,
) -> dict[str, Any]:
    """Search table embeddings without materializing the full catalog."""
    if data_retriever is None:
        return _error("unavailable", "The table retriever is not configured.")
    if not query or not query.strip():
        return _error("invalid_arguments", "Provide a non-empty query.")
    bounded = min(max(int(limit), 1), MAX_SEARCH_RESULTS)
    rows = get_relevant_tables(
        data_retriever,
        query.strip(),
        k=bounded,
        database_name=database_name,
    )
    return _ok({"tables": rows[:bounded]}, truncated=len(rows) > bounded)


def _connector_for_database(connectors: list[Any], database_name: str) -> Any | None:
    matches = [
        connector
        for connector in connectors
        if str(getattr(connector, "database_name", "")).casefold()
        == database_name.casefold()
    ]
    if len(matches) == 1:
        return matches[0]
    if not database_name and len(connectors) == 1:
        return connectors[0]
    return None


def _literal_sql(value: Any, dialect: str | None) -> str:
    if isinstance(value, bool):
        return exp.Boolean(this=value).sql(dialect=dialect or None)
    if isinstance(value, (int, float)):
        return exp.Literal.number(str(value)).sql(dialect=dialect or None)
    return exp.Literal.string(str(value)).sql(dialect=dialect or None)


def _is_text_type(datatype: Any) -> bool:
    normalized = str(datatype or "").casefold()
    return any(marker in normalized for marker in _TEXT_TYPE_MARKERS)


def _probe_sql(
    context: dict[str, Any],
    value: Any,
    *,
    contains: bool,
    dialect: str | None,
) -> str:
    column = quoted_identifier(str(context["col_name"]), dialect)
    table = qualified_name(
        context.get("schema_name"),
        context.get("table_name"),
        dialect=dialect,
    )
    alias = quoted_identifier("matched_value", dialect)
    text_column = f"CAST({column} AS TEXT)"
    if contains:
        predicate = (
            f"LOWER({text_column}) LIKE "
            f"LOWER({_literal_sql('%' + str(value) + '%', dialect)})"
        )
    else:
        predicate = (
            f"LOWER({text_column}) = "
            f"LOWER(CAST({_literal_sql(value, dialect)} AS TEXT))"
        )
    return f"SELECT {column} AS {alias} FROM {table} WHERE {predicate}"


def probe_value(
    *,
    value: Any = None,
    attribute_ids: list[str] | None = None,
    database_name: str | None = None,
    connectors: list[Any] | None = None,
    **_: Any,
) -> dict[str, Any]:
    """Find a literal in ordered suspects, globally preferring equality to LIKE."""
    if value is None or value == "":
        return _error("invalid_arguments", "Provide a non-empty value.")
    ordered_ids = list(dict.fromkeys(attribute_ids or []))[:MAX_VALUE_SUSPECTS]
    if not ordered_ids:
        return _error("invalid_arguments", "Provide suspected ColumnAttribute IDs.")
    contexts = fetch_attr_column_contexts(ordered_ids, database_name=database_name)
    ordered_contexts = [
        (attribute_id, contexts[attribute_id])
        for attribute_id in ordered_ids
        if attribute_id in contexts
    ]
    if not ordered_contexts:
        return _error("not_found", "No suspected attributes could be resolved.")

    resolved_database = database_name or str(
        ordered_contexts[0][1].get("database_name") or ""
    )
    connector = _connector_for_database(connectors or [], resolved_database)
    if connector is None:
        return _error(
            "connector_unavailable",
            f"No unique connector is available for {resolved_database!r}.",
        )
    dialect = getattr(connector, "dialect", None)

    with ProbeExecutor(
        connector,
        max_calls=min(MAX_VALUE_SUSPECTS * 2, 24),
        max_rows=2,
    ) as executor:
        phases = (
            ("equal", ordered_contexts),
            (
                "contains",
                [
                    item
                    for item in ordered_contexts
                    if _is_text_type(item[1].get("datatype"))
                ],
            ),
        )
        for match_type, suspects in phases:
            for attribute_id, context in suspects:
                result = executor.run(
                    _probe_sql(
                        context,
                        value,
                        contains=match_type == "contains",
                        dialect=dialect,
                    ),
                    purpose=f"{match_type} value grounding",
                )
                rows = result.get("rows") or []
                if result.get("ok") and rows:
                    stored_value = next(iter(rows[0].values()))
                    return _ok(
                        {
                            "input_value": value,
                            "stored_value": stored_value,
                            "match_type": match_type,
                            "column_attribute_id": attribute_id,
                            "column_id": context.get("col_id"),
                            "column_name": context.get("col_name"),
                            "table_id": context.get("table_id"),
                            "table_name": context.get("table_name"),
                            "schema_name": context.get("schema_name"),
                            "database_name": context.get("database_name"),
                            "probe_calls": executor.calls,
                        }
                    )
        return _ok(
            {
                "input_value": value,
                "match": None,
                "probe_calls": executor.calls,
            }
        )


def find_attribute_join_path(
    *,
    anchor_attribute_id: str | None = None,
    destination_attribute_id: str | None = None,
    database_name: str | None = None,
    **_: Any,
) -> dict[str, Any]:
    """Resolve two semantic attributes and return their shortest directed path."""
    if not anchor_attribute_id or not destination_attribute_id:
        return _error(
            "invalid_arguments",
            "Provide anchor_attribute_id and destination_attribute_id.",
        )
    contexts = fetch_attr_column_contexts(
        [anchor_attribute_id, destination_attribute_id],
        database_name=database_name,
    )
    if anchor_attribute_id not in contexts or destination_attribute_id not in contexts:
        return _error("not_found", "One or both attributes could not be resolved.")
    path = find_join_path(
        contexts[anchor_attribute_id]["col_id"],
        contexts[destination_attribute_id]["col_id"],
    )
    return _ok(
        {
            "anchor_attribute_id": anchor_attribute_id,
            "destination_attribute_id": destination_attribute_id,
            "path": path,
        }
    )


def discover_connections(
    *,
    table_ids: list[str] | None = None,
    anchor_table_id: str | None = None,
    **_: Any,
) -> dict[str, Any]:
    """Return bounded structural junction, bridge, and shared-parent context."""
    ids = list(dict.fromkeys(table_ids or []))[:MAX_CONNECTION_TABLES]
    junctions, junction_paths = find_connected_junction_tables(ids)
    bridges, bridge_paths, skipped_pairs = find_kept_table_bridges(ids, set(ids))
    siblings: list[dict[str, Any]] = []
    if anchor_table_id:
        siblings, _ = find_anchor_hub_siblings(
            anchor_table_id,
            max_siblings=MAX_CONNECTION_TABLES,
        )
    return _ok(
        {
            "junction_tables": junctions,
            "junction_paths": junction_paths,
            "bridge_tables": bridges,
            "bridge_paths": bridge_paths,
            "skipped_bridge_pairs": skipped_pairs,
            "anchor_hub_siblings": siblings,
        }
    )


SEMANTIC_PREPARATION_TOOLS: dict[
    SemanticPreparationToolName, Callable[..., dict[str, Any]]
] = {
    SemanticPreparationToolName.SEARCH_SEMANTIC_LAYER: search_semantic_layer,
    SemanticPreparationToolName.SEARCH_TABLES: search_tables,
    SemanticPreparationToolName.GET_COLUMN_ATTRIBUTE: get_column_attribute,
    SemanticPreparationToolName.GET_TABLE: get_table,
    SemanticPreparationToolName.PROBE_VALUE: probe_value,
    SemanticPreparationToolName.FIND_JOIN_PATH: find_attribute_join_path,
    SemanticPreparationToolName.DISCOVER_CONNECTIONS: discover_connections,
}


def run_semantic_preparation_tool(
    tool_name: SemanticPreparationToolName | str,
    arguments: dict[str, Any],
    *,
    semantic_retriever: Any = None,
    data_retriever: Any = None,
    connectors: list[Any] | None = None,
) -> dict[str, Any]:
    """Validate and dispatch one semantic-preparation tool call."""
    try:
        name = SemanticPreparationToolName(tool_name)
    except ValueError:
        return _error("unknown_tool", f"Unknown preparation tool {tool_name!r}.")
    injected: dict[str, Any] = {}
    if name == SemanticPreparationToolName.SEARCH_SEMANTIC_LAYER:
        injected["semantic_retriever"] = semantic_retriever
    elif name == SemanticPreparationToolName.SEARCH_TABLES:
        injected["data_retriever"] = data_retriever
    elif name == SemanticPreparationToolName.PROBE_VALUE:
        injected["connectors"] = connectors or []
    return SEMANTIC_PREPARATION_TOOLS[name](**arguments, **injected)


__all__ = [
    "MAX_SEARCH_RESULTS",
    "MAX_VALUE_SUSPECTS",
    "SEMANTIC_PREPARATION_TOOLS",
    "SemanticPreparationToolName",
    "discover_connections",
    "find_attribute_join_path",
    "probe_value",
    "run_semantic_preparation_tool",
    "search_semantic_layer",
    "search_tables",
]
