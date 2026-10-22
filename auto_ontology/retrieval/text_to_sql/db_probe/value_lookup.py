# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Safe live lookup of one textual value in one catalog column."""

from __future__ import annotations

from typing import Any

from sqlglot import exp

from auto_ontology.connectors.base import SQLDatabase
from auto_ontology.retrieval.text_to_sql.db_probe.executor import ProbeExecutor
from auto_ontology.utils.sql_identifiers import quoted_identifier


def _first_value(result: dict[str, Any]) -> str | None:
    rows = result.get("rows") or []
    if not rows or not isinstance(rows[0], dict) or not rows[0]:
        return None
    value = next(iter(rows[0].values()))
    return None if value is None else str(value)


def _text_ref(column_ref: str, dialect: str | None) -> str:
    """Make Postgres ENUM values usable by LOWER/TRIM without harming text."""
    return f"{column_ref}::text" if dialect == "postgres" else column_ref


def _run_value_probe(
    executor: ProbeExecutor,
    *,
    table_ref: str,
    column_ref: str,
    predicate: str,
    purpose: str,
) -> str | None:
    result = executor.run(
        f"SELECT DISTINCT {column_ref} AS value FROM {table_ref} "
        f"WHERE {predicate} LIMIT 1",
        purpose=purpose,
    )
    if not result.get("ok"):
        return None
    return _first_value(result)


def lookup_text_value(
    executor: ProbeExecutor,
    connector: SQLDatabase,
    *,
    schema_name: str,
    table_name: str,
    column_name: str,
    value: str,
) -> tuple[str, str] | None:
    """Return ``(stored_value, kind)`` using exact, normalized, then contains."""
    dialect = getattr(connector, "dialect", None)
    table_ref = connector.qualify(schema_name or None, table_name)
    column_ref = quoted_identifier(column_name, dialect)
    literal = exp.Literal.string(value).sql(dialect=dialect or None)

    exact = _run_value_probe(
        executor,
        table_ref=table_ref,
        column_ref=column_ref,
        predicate=f"{column_ref} = {literal}",
        purpose="value_anchor_exact",
    )
    if exact is not None:
        return exact, "value"
    if executor.budget_left <= 0:
        return None

    text_ref = _text_ref(column_ref, dialect)
    normalized = _run_value_probe(
        executor,
        table_ref=table_ref,
        column_ref=column_ref,
        predicate=f"LOWER(TRIM({text_ref})) = LOWER(TRIM({literal}))",
        purpose="value_anchor_normalized",
    )
    if normalized is not None:
        return normalized, "value"
    if executor.budget_left <= 0:
        return None

    escaped = value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    pattern = exp.Literal.string(f"%{escaped}%").sql(dialect=dialect or None)
    contains = _run_value_probe(
        executor,
        table_ref=table_ref,
        column_ref=column_ref,
        predicate=f"LOWER({text_ref}) LIKE LOWER({pattern}) ESCAPE '\\\\'",
        purpose="value_anchor_contains",
    )
    if contains is not None:
        return contains, "contains"
    return None


__all__ = ["lookup_text_value"]
