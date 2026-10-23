# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Classify junction-table row grain after semantic FK resolution."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from auto_ontology.dal.attributes import fetch_semantic_fk_join_groups
from auto_ontology.dal.datasources import (
    fetch_table_context,
    fetch_table_headers_for_database,
    replace_junction_table_classifications,
)
from auto_ontology.semantic.models import JunctionTableDecisions
from auto_ontology.utils.llm_invoke import (
    get_llm_client,
    invoke_with_structured_output,
)

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
Classify relational tables by row grain.

A junction table is true only when each row represents an association between
at least two foreign-key roles. Distinct roles may reference the same entity
type. Non-key columns are allowed only when they describe the association,
including its role, ordering, quantity, status, or validity.

Return false for entity tables, lookup tables, child/detail extensions,
hierarchies with one relationship role, and event or transaction records with
an independent business identity or lifecycle. Multiple foreign keys alone do
not make a table a junction.

Use the complete resolved foreign-key roles supplied below. Return one decision
for every supplied table_id and do not invent table IDs."""


def _merge_fk_roles(
    table: dict[str, Any],
    context: dict[str, Any],
    semantic_rows: list[dict[str, Any]],
) -> list[dict[str, str]]:
    roles: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in context.get("fks", []):
        source = str(row.get("source_column") or "")
        target_table = str(row.get("target_table") or "")
        target_column = str(row.get("target_column") or "")
        if source and target_table:
            roles[(source, target_table, target_column)] = {
                "source_column": source,
                "target_table": target_table,
                "target_column": target_column,
            }
    for row in semantic_rows:
        source = str(row.get("source_column_name") or "")
        target_table = str(row.get("target_table_name") or "")
        target_column = str(row.get("target_column_name") or "")
        if source and target_table:
            roles[(source, target_table, target_column)] = {
                "source_column": source,
                "target_table": target_table,
                "target_column": target_column,
            }
    return sorted(
        roles.values(),
        key=lambda role: (
            role["source_column"].casefold(),
            role["target_table"].casefold(),
            role["target_column"].casefold(),
        ),
    )


def _format_candidate(
    table: dict[str, Any],
    context: dict[str, Any],
    roles: list[dict[str, str]],
) -> str:
    columns = "\n".join(
        "  - "
        f"{column.get('name')} ({column.get('data_type') or 'unknown'}): "
        f"{column.get('description') or '(no description)'}"
        for column in context.get("columns", [])
    )
    role_lines = "\n".join(
        f"  - {role['source_column']} -> {role['target_table']}.{role['target_column']}"
        for role in roles
    )
    qualified_name = ".".join(
        part for part in (table.get("schema_name"), table.get("name")) if part
    )
    return (
        f"table_id: {table['id']}\n"
        f"table: {qualified_name}\n"
        f"description: {table.get('description') or '(none)'}\n"
        f"primary_key: {table.get('pk') or []}\n"
        f"resolved_foreign_key_roles:\n{role_lines}\n"
        f"columns:\n{columns}"
    )


def classify_junction_tables(database_name: str) -> int:
    """Classify and persist every true junction table in one database."""
    tables = fetch_table_headers_for_database(database_name)
    semantic_by_table: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in fetch_semantic_fk_join_groups(database_name):
        semantic_by_table[str(row.get("source_table_id") or "")].append(row)

    deterministic: list[dict[str, str]] = []
    llm_candidates: list[
        tuple[dict[str, Any], dict[str, Any], list[dict[str, str]]]
    ] = []
    for table in tables:
        context = fetch_table_context(str(table["id"]))
        roles = _merge_fk_roles(
            table,
            context,
            semantic_by_table.get(str(table["id"]), []),
        )
        role_columns = {role["source_column"] for role in roles}
        if len(role_columns) < 2:
            continue

        columns = {
            str(column.get("name") or "")
            for column in context.get("columns", [])
            if column.get("name")
        }
        if columns and columns <= role_columns:
            deterministic.append(
                {
                    "table_id": str(table["id"]),
                    "kind": "bridge",
                    "rationale": (
                        "Every column is a resolved foreign-key role, so each row "
                        "represents an association."
                    ),
                }
            )
        else:
            llm_candidates.append((table, context, roles))

    classifications = list(deterministic)
    if llm_candidates:
        prompt = "\n\n---\n\n".join(
            _format_candidate(table, context, roles)
            for table, context, roles in llm_candidates
        )
        result = invoke_with_structured_output(
            get_llm_client(temperature=0.0),
            [
                SystemMessage(content=_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ],
            JunctionTableDecisions,
        )
        candidates_by_id = {str(table["id"]): table for table, _, _ in llm_candidates}
        decided_ids: set[str] = set()
        if result is not None:
            for decision in result.decisions:
                table_id = decision.table_id.strip()
                if table_id not in candidates_by_id or table_id in decided_ids:
                    continue
                decided_ids.add(table_id)
                if decision.is_junction_table:
                    classifications.append(
                        {
                            "table_id": table_id,
                            "kind": "association",
                            "rationale": decision.rationale.strip(),
                        }
                    )

        # A missing batch decision is uncertainty, not evidence that an
        # existing positive classification became false.
        for table, _, _ in llm_candidates:
            table_id = str(table["id"])
            if (
                table_id not in decided_ids
                and table.get("is_junction_table")
                and table.get("junction_table_kind")
            ):
                classifications.append(
                    {
                        "table_id": table_id,
                        "kind": str(table["junction_table_kind"]),
                        "rationale": str(table.get("junction_table_rationale") or ""),
                    }
                )

    replace_junction_table_classifications(database_name, classifications)
    logger.info(
        "Classified %d junction table(s) in database %r",
        len(classifications),
        database_name,
    )
    return len(classifications)


__all__ = ["classify_junction_tables"]
