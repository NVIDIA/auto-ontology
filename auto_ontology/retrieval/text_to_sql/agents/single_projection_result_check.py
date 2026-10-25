# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Post-execution integrity checks for one-column list results."""

from __future__ import annotations

import json
from typing import Any

import sqlglot
from sqlglot import exp

from auto_ontology.retrieval.text_to_sql.agents.empty_like_result_check import (
    _get_sql_code,
    _parse_db_rows,
    _set_sql_code,
)
from auto_ontology.retrieval.text_to_sql.base import BaseAgent
from auto_ontology.retrieval.text_to_sql.connector_routing import (
    resolve_connector_from_tables,
)
from auto_ontology.retrieval.text_to_sql.state import AgentState


def _single_projection_select(
    sql: str,
    dialect: str | None,
) -> tuple[exp.Select, exp.Expression] | None:
    """Return a simple outer SELECT and its sole projection."""
    try:
        tree = sqlglot.parse_one(sql, read=dialect or None)
    except Exception:
        return None
    if not isinstance(tree, exp.Select) or len(tree.expressions) != 1:
        return None
    return tree, tree.expressions[0]


def _single_result_values(db_result: Any) -> list[Any] | None:
    """Return one-column result values, or ``None`` for another result shape."""
    rows = _parse_db_rows(db_result)
    if not rows:
        return []
    if any(len(row) != 1 for row in rows):
        return None
    return [next(iter(row.values())) for row in rows]


def _has_repeated_values(values: list[Any]) -> bool:
    """Compare scalar or structured result values deterministically."""
    seen: set[str] = set()
    for value in values:
        canonical = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        if canonical in seen:
            return True
        seen.add(canonical)
    return False


def _bare_projected_column(projection: exp.Expression) -> exp.Column | None:
    source = projection.this if isinstance(projection, exp.Alias) else projection
    while isinstance(source, exp.Paren):
        source = source.this
    return source if isinstance(source, exp.Column) else None


def _is_unfiltered_single_table_column(
    select: exp.Select,
    projection: exp.Expression,
) -> bool:
    """Whether NULL removal has the narrow, unambiguous requested shape."""
    if len(list(select.find_all(exp.Select))) != 1:
        return False
    if len(list(select.find_all(exp.Table))) != 1:
        return False
    if select.args.get("joins"):
        return False
    if any(
        select.find(predicate_type) is not None
        for predicate_type in (exp.Where, exp.Having, exp.Qualify, exp.In)
    ):
        return False
    return _bare_projected_column(projection) is not None


def build_single_projection_null_error(column: exp.Column) -> str:
    """Mandatory reconstruction instruction for an unfiltered NULL list."""
    return (
        "SQL executed successfully, but its single projected field returned NULL "
        "values. This is an unfiltered single-table list query. Add "
        f"`WHERE {column.sql()} IS NOT NULL` and keep the projection, table, "
        "ordering, grouping, and limit unchanged."
    )


class SingleProjectionResultCheckAgent(BaseAgent):
    """Remove NULLs and repeated values from one-column list results."""

    def __init__(self) -> None:
        super().__init__("single_projection_result_check")

    def validate_input(self, state: AgentState) -> bool:
        path_state = state.get("path_state", {})
        if path_state.get("sql_response_from_db") is None:
            self.logger.warning("No SQL execution result found for projection check")
            return False
        return True

    def execute(self, state: AgentState) -> dict[str, Any]:
        path_state = dict(state.get("path_state", {}))
        sql_code = _get_sql_code(path_state)
        values = _single_result_values(path_state.get("sql_response_from_db"))
        if values is None or not values:
            return {"decision": "valid_sql", "path_state": path_state}

        connector = resolve_connector_from_tables(
            path_state.get("relevant_tables") or [],
            state.get("connectors") or [],
        )
        dialect = getattr(connector, "dialect", None)
        parsed = _single_projection_select(sql_code, dialect)
        if parsed is None:
            return {"decision": "valid_sql", "path_state": path_state}
        select, projection = parsed

        if any(
            value is None for value in values
        ) and _is_unfiltered_single_table_column(select, projection):
            if path_state.get("single_projection_null_retry_attempted"):
                self.logger.info(
                    "Single-projection NULL repair already attempted — passing through"
                )
                return {"decision": "valid_sql", "path_state": path_state}
            column = _bare_projected_column(projection)
            assert column is not None
            path_state["single_projection_null_retry_attempted"] = True
            path_state["error"] = build_single_projection_null_error(column)
            path_state["error_known_fixable"] = True
            self.logger.info(
                "Unfiltered one-column result contains NULL — routing to reconstruction"
            )
            return {"decision": "invalid_sql", "path_state": path_state}

        if not _has_repeated_values(values) or select.args.get("distinct"):
            return {"decision": "valid_sql", "path_state": path_state}

        rewritten = select.copy()
        rewritten.set("distinct", exp.Distinct())
        rewritten_sql = rewritten.sql(dialect=dialect or None)
        _set_sql_code(path_state, rewritten_sql)
        path_state["sql_response_from_db"] = None
        self.logger.info(
            "One-column result contains repeated values — adding DISTINCT and "
            "re-executing SQL"
        )
        return {"decision": "retry_sql", "path_state": path_state}


__all__ = [
    "SingleProjectionResultCheckAgent",
    "build_single_projection_null_error",
]
