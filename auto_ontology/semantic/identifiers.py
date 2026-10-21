# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Choose one reusable default identity projection for each catalog table."""

from __future__ import annotations

import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from auto_ontology.dal.datasources import (
    fetch_tables_for_identifier_selection,
    set_default_table_identifier,
)
from auto_ontology.semantic.models import IdentifierSelectionResult
from auto_ontology.utils.llm_invoke import (
    get_llm_client,
    invoke_with_structured_output,
)

logger = logging.getLogger(__name__)

_SYSTEM = """\
Select the one column that should be returned by default when a user asks to identify
an entity represented by this table without explicitly requesting a field.

Prefer a stable business or surrogate identifier over a display name. A display name
is appropriate only when no reliable identifier exists. Do not select one component of
a composite primary key as though it uniquely identifies a row. Return null when no
single supplied column reliably identifies the entity. Return only a supplied column ID."""


def _normalized(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _find_named_column(
    columns: list[dict[str, Any]], names: set[str]
) -> dict[str, Any] | None:
    matches = [
        column
        for column in columns
        if _normalized(str(column.get("name") or "")) in names
        and column.get("is_unique") is not False
    ]
    return matches[0] if len(matches) == 1 else None


def _deterministic_identifier(table: dict[str, Any]) -> dict[str, Any] | None:
    columns = table["columns"]
    by_id = {column["id"]: column for column in columns}
    current = table.get("default_identifier_column_id")
    if current in by_id:
        return by_id[current]

    pk = table.get("pk") or []
    if isinstance(pk, str):
        pk = [pk] if pk else []
    if len(pk) == 1:
        pk_name = str(pk[0]).casefold()
        matches = [
            column
            for column in columns
            if str(column.get("name") or "").casefold() == pk_name
        ]
        if len(matches) == 1:
            return matches[0]

    # A composite key has no valid one-column default unless another candidate
    # is selected below or by the semantic LLM.
    composite_parts = {str(name).casefold() for name in pk} if len(pk) > 1 else set()
    single_column_candidates = [
        column
        for column in columns
        if str(column.get("name") or "").casefold() not in composite_parts
    ]
    exact = _find_named_column(single_column_candidates, {"id"})
    if exact is not None:
        return exact

    table_name = _normalized(str(table.get("name") or ""))
    entity_names = {table_name} if table_name else set()
    if len(table_name) > 3 and table_name.endswith("s"):
        entity_names.add(table_name[:-1])
    table_ids = {f"{name}id" for name in entity_names}
    exact = _find_named_column(single_column_candidates, table_ids | {"uuid"})
    if exact is not None:
        return exact

    display = _find_named_column(
        single_column_candidates,
        {"name", *(f"{name}name" for name in entity_names)},
    )
    if display is not None and len(columns) == 1:
        return display
    return None


def _llm_identifier(table: dict[str, Any]) -> str | None:
    columns = table["columns"]
    if not columns:
        return None
    lines = "\n".join(
        "- "
        f"id={column['id']!r}; name={column['name']!r}; "
        f"type={column.get('data_type')!r}; unique={column.get('is_unique')!r}; "
        f"description={column.get('description')!r}"
        for column in columns
    )
    prompt = (
        f"Table: {table.get('schema_name')}.{table.get('name')}\n"
        f"Description: {table.get('description') or ''}\n"
        f"Declared primary key: {table.get('pk') or '(none)'}\n"
        f"Columns:\n{lines}"
    )
    result = invoke_with_structured_output(
        get_llm_client(temperature=0.0),
        [SystemMessage(content=_SYSTEM), HumanMessage(content=prompt)],
        IdentifierSelectionResult,
    )
    if result is None or result.selected_column_id is None:
        return None
    pk = table.get("pk") or []
    if isinstance(pk, str):
        pk = [pk] if pk else []
    composite_parts = {str(name).casefold() for name in pk} if len(pk) > 1 else set()
    allowed = {
        column["id"]
        for column in columns
        if str(column.get("name") or "").casefold() not in composite_parts
    }
    if result.selected_column_id not in allowed:
        logger.warning(
            "Ignoring invalid default identifier %r for table %r",
            result.selected_column_id,
            table.get("id"),
        )
        return None
    return result.selected_column_id


def select_default_identifiers(database_name: str) -> int:
    """Populate missing table identifiers, preserving every valid existing choice."""
    selected = 0
    for table in fetch_tables_for_identifier_selection(database_name):
        deterministic = _deterministic_identifier(table)
        column_id = (
            deterministic["id"] if deterministic is not None else _llm_identifier(table)
        )
        if column_id is None:
            logger.info("No reliable default identifier for table %r", table["id"])
            continue
        if column_id == table.get("default_identifier_column_id"):
            continue
        if set_default_table_identifier(table["id"], column_id):
            selected += 1
    return selected
