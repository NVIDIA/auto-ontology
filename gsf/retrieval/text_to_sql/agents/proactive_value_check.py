# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Proactive (pre-execution) value check.

Optional variant of the value-repair signal, gated by ``DB_PROBE_PROACTIVE``.
Unlike the post-execution empty-result check, this runs *before* execution and
regardless of the (not-yet-known) result, so it also catches wrong literals that
would return non-empty-but-wrong rows. It costs a few cheap ``DISTINCT`` probes
on every query that has a categorical equality/IN filter, which is why it is
opt-in. Runs at most once per request (guarded) to avoid reconstruction loops.
"""

from __future__ import annotations

from typing import Any, Dict

from gsf.retrieval.text_to_sql.agents.empty_like_result_check import _get_sql_code
from gsf.retrieval.text_to_sql.base import BaseAgent
from gsf.retrieval.text_to_sql.connector_routing import resolve_connector_from_tables
from gsf.retrieval.text_to_sql.db_probe.executor import ProbeExecutor
from gsf.retrieval.text_to_sql.db_probe.literal_check import (
    build_value_repair_error,
    find_literal_mismatches,
)
from gsf.retrieval.text_to_sql.state import AgentState


class ProactiveValueCheckAgent(BaseAgent):
    """Check filter literals against real DB values before executing the query."""

    def __init__(self) -> None:
        super().__init__("proactive_value_check")

    def validate_input(self, state: AgentState) -> bool:
        return True

    def execute(self, state: AgentState) -> Dict[str, Any]:
        path_state = dict(state.get("path_state", {}))
        sql_code = _get_sql_code(path_state)

        # One-shot: never re-check after a repair, so we can't loop here.
        if not sql_code.strip() or path_state.get("value_repair_attempted"):
            return {"decision": "valid_sql", "path_state": path_state}

        connectors = state.get("connectors") or []
        relevant_tables = list(path_state.get("relevant_tables") or [])
        connector = resolve_connector_from_tables(relevant_tables, connectors)
        dialect = getattr(connector, "dialect", None)

        with ProbeExecutor(connector) as executor:
            mismatches = find_literal_mismatches(executor, dialect, sql_code)

        if not mismatches:
            return {"decision": "valid_sql", "path_state": path_state}

        path_state["value_repair_attempted"] = True
        path_state["error"] = build_value_repair_error(mismatches)
        self.logger.info(
            "Proactive check — routing to reconstruction to fix %d literal(s): %s",
            len(mismatches),
            [f"{m['table']}.{m['column']}='{m['used']}'" for m in mismatches],
        )
        return {"decision": "invalid_sql", "path_state": path_state}


__all__ = ["ProactiveValueCheckAgent"]
