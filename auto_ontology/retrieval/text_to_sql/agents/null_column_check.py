# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reject a result with an output column that is NULL in every row.

A requested output that comes back NULL everywhere is almost never the answer:
the query projected a NULL placeholder instead of computing the value, or the
join that supplies the column never matches. Neither shows up before execution,
and the empty-result checks do not fire because the other columns have rows.

The check reads the executed result only — no probe, no LLM call — and routes
once to reconstruction naming the columns. A question that asks for NULLs
itself (subtotal rows "in the format (null, ...)") is left alone, since there
an all-NULL column can be what was requested.
"""

from __future__ import annotations

import re
from typing import Any, Dict

from auto_ontology.retrieval.text_to_sql.agents.empty_like_result_check import (
    _parse_db_rows,
)
from auto_ontology.retrieval.text_to_sql.base import BaseAgent
from auto_ontology.retrieval.text_to_sql.state import (
    AgentState,
    get_original_question,
    get_question_for_processing,
)

_MENTIONS_NULL = re.compile(r"\bnulls?\b", re.IGNORECASE)


def all_null_columns(db_result: Any) -> list[str]:
    """Columns that are NULL in every row of a non-empty *db_result*."""
    rows = _parse_db_rows(db_result)
    if not rows:
        return []
    columns = list(dict.fromkeys(key for row in rows for key in row))
    return [
        column for column in columns if all(row.get(column) is None for row in rows)
    ]


def build_null_column_error(columns: list[str]) -> str:
    names = ", ".join(f"`{column}`" for column in columns)
    return (
        f"SQL executed, but output column(s) {names} are NULL in every returned "
        "row. A requested value that is NULL everywhere means the query does "
        "not compute it: either the SELECT projects a NULL placeholder, or the "
        "join or subquery that should supply it never matches (a LEFT JOIN on "
        "keys that hold different values, or on a column of another meaning). "
        "Compute each of these columns from the table that stores it, join it "
        "on keys that actually match, and keep the parts of the query that are "
        "already correct."
    )


class NullColumnCheckAgent(BaseAgent):
    """Route once to reconstruction when an output column is entirely NULL."""

    def __init__(self) -> None:
        super().__init__("null_column_check")

    def validate_input(self, state: AgentState) -> bool:
        return state.get("path_state", {}).get("sql_response_from_db") is not None

    def execute(self, state: AgentState) -> Dict[str, Any]:
        path_state = dict(state.get("path_state", {}))
        if path_state.get("null_column_retry_attempted"):
            return {"decision": "valid_sql", "path_state": path_state}

        question = " ".join(
            filter(
                None, (get_original_question(state), get_question_for_processing(state))
            )
        )
        if _MENTIONS_NULL.search(question):
            return {"decision": "valid_sql", "path_state": path_state}

        columns = all_null_columns(path_state.get("sql_response_from_db"))
        if not columns:
            return {"decision": "valid_sql", "path_state": path_state}

        path_state["null_column_retry_attempted"] = True
        path_state["error"] = build_null_column_error(columns)
        self.logger.info(
            "Output column(s) %s NULL in every row — routing to reconstruction",
            columns,
        )
        return {"decision": "invalid_sql", "path_state": path_state}
