# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Safe live lookup of one literal value in one catalog column."""

from __future__ import annotations

from typing import Any, Literal

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
    """Cast every physical type to comparable text in the active dialect."""
    target_type = "STRING" if dialect in {"bigquery", "databricks", "spark"} else "TEXT"
    return f"CAST({column_ref} AS {target_type})"


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
    match_kind: Literal["equal", "contains"],
) -> str | None:
    """Return the stored value from one case-insensitive equality or LIKE probe."""
    dialect = getattr(connector, "dialect", None)
    table_ref = connector.qualify(schema_name or None, table_name)
    column_ref = quoted_identifier(column_name, dialect)
    text_ref = _text_ref(column_ref, dialect)
    if match_kind == "equal":
        literal = exp.Literal.string(value).sql(dialect=dialect or None)
        predicate = f"LOWER(TRIM({text_ref})) = LOWER(TRIM({literal}))"
        purpose = "value_anchor_equal"
    else:
        escaped = value.replace("!", "!!").replace("%", "!%").replace("_", "!_")
        pattern = exp.Literal.string(f"%{escaped}%").sql(dialect=dialect or None)
        predicate = f"LOWER({text_ref}) LIKE LOWER({pattern}) ESCAPE '!'"
        purpose = "value_anchor_contains"

    return _run_value_probe(
        executor,
        table_ref=table_ref,
        column_ref=column_ref,
        predicate=predicate,
        purpose=purpose,
    )


__all__ = ["lookup_text_value"]
