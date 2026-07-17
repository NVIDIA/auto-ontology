# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Empty-result value repair.

Runs right after execution, only when the query came back empty. It checks the
filter literals against the columns' actual distinct values and, when a literal
is a near-miss of a real value (wrong case / spelling), routes once to the
existing reconstruction node with a targeted hint. On the common path (non-empty
result) it is a single boolean check — no probes, no LLM call.
"""

from __future__ import annotations

from typing import Any, Dict

from gsf.retrieval.text_to_sql.agents.empty_like_result_check import (
    _get_sql_code,
    _is_empty_db_result,
)
from gsf.retrieval.text_to_sql.base import BaseAgent
from gsf.retrieval.text_to_sql.connector_routing import resolve_connector_from_tables
from gsf.retrieval.text_to_sql.db_probe.executor import ProbeExecutor
from gsf.retrieval.text_to_sql.db_probe.literal_check import (
    build_value_repair_error,
    find_literal_mismatches,
)
from gsf.retrieval.text_to_sql.state import AgentState


class EmptyResultValueRepairAgent(BaseAgent):
    """Detect filter literals that don't exist in the DB and route to repair."""

    def __init__(self) -> None:
        super().__init__("empty_result_value_repair")

    def validate_input(self, state: AgentState) -> bool:
        path_state = state.get("path_state", {})
        if path_state.get("sql_response_from_db") is None:
            self.logger.warning("No SQL execution result found for value repair")
            return False
        return True

    def execute(self, state: AgentState) -> Dict[str, Any]:
        path_state = dict(state.get("path_state", {}))
        sql_code = _get_sql_code(path_state)
        db_result = path_state.get("sql_response_from_db")

        empty = _is_empty_db_result(db_result)

        # Common path: query returned rows → nothing to repair, zero cost.
        if not empty:
            return {"decision": "valid_sql", "path_state": path_state}

        if path_state.get("value_repair_attempted"):
            self.logger.info("Value repair already attempted — passing through")
            return {"decision": "valid_sql", "path_state": path_state}

        connectors = state.get("connectors") or []
        relevant_tables = list(path_state.get("relevant_tables") or [])
        connector = resolve_connector_from_tables(relevant_tables, connectors)
        dialect = getattr(connector, "dialect", None)

        with ProbeExecutor(connector) as executor:
            mismatches = find_literal_mismatches(executor, dialect, sql_code)

        if not mismatches:
            self.logger.info(
                "Empty result but no literal mismatches — passing through"
            )
            return {"decision": "valid_sql", "path_state": path_state}

        path_state["value_repair_attempted"] = True
        path_state["error"] = build_value_repair_error(mismatches)
        self.logger.info(
            "Empty result — routing to reconstruction to fix %d literal(s): %s",
            len(mismatches),
            [f"{m['table']}.{m['column']}='{m['used']}'" for m in mismatches],
        )
        return {"decision": "invalid_sql", "path_state": path_state}


__all__ = ["EmptyResultValueRepairAgent"]
