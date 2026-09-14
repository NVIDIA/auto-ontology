# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Proactive (pre-execution) value check.

Optional variant of the value-repair signal, gated by ``DB_PROBE_PROACTIVE``.
Unlike the post-execution empty-result check, this runs *before* execution and
regardless of the (not-yet-known) result, so it also catches wrong literals that
would return non-empty-but-wrong rows — including a numeric threshold that's
merely on the wrong scale (e.g. a percent vs. fraction mismatch), which a
non-empty-but-wrong result would otherwise hide. It costs a few cheap
``DISTINCT``/``MIN``/``MAX`` probes on every query that has a categorical
equality/IN filter or a numeric comparison filter, which is why it is opt-in.
Runs at most once per request (guarded) to avoid reconstruction loops.
"""

from __future__ import annotations

from typing import Any, Dict

from gsf.retrieval.text_to_sql.agents.empty_like_result_check import (
    _get_sql_code,
    _set_sql_code,
)
from gsf.retrieval.text_to_sql.base import BaseAgent
from gsf.retrieval.text_to_sql.connector_routing import resolve_connector_from_tables
from gsf.retrieval.text_to_sql.db_probe.executor import ProbeExecutor
from gsf.retrieval.text_to_sql.db_probe.literal_check import (
    build_value_repair_error,
    find_integer_division_mismatches,
    find_literal_mismatches,
    find_numeric_scale_mismatches,
    try_self_apply_integer_division_fixes,
)
from gsf.retrieval.text_to_sql.state import AgentState


def _describe_mismatch(m: dict) -> str:
    """One-line description of a mismatch for the routing-decision log line."""
    if m.get("kind") == "integer_division":
        return f"{m['numerator']} / {m['denominator']}"
    return f"{m['table']}.{m['column']}='{m['used']}'"


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

        # Cheap, already-fetched — no extra DB/graph round-trip. Same pattern
        # as JsonbPathCheckAgent: lets find_integer_division_mismatches tell
        # a known-integer column from an unresolvable one without probing.
        known_types: dict[tuple[str, str], str] = {}
        for table in relevant_tables:
            table_name = table.get("name")
            if not table_name:
                continue
            for col in table.get("columns") or []:
                if not isinstance(col, dict):
                    continue
                col_name = col.get("name")
                data_type = col.get("data_type")
                if col_name and data_type:
                    known_types[(table_name.lower(), col_name.lower())] = data_type

        with ProbeExecutor(connector) as executor:
            mismatches = find_literal_mismatches(executor, dialect, sql_code)
            mismatches += find_numeric_scale_mismatches(executor, dialect, sql_code)
        mismatches += find_integer_division_mismatches(
            dialect, sql_code, known_types=known_types
        )

        if not mismatches:
            return {"decision": "valid_sql", "path_state": path_state}

        # Integer-division mismatches have exactly one correct fix (cast the
        # numerator) — self-apply them directly instead of routing through
        # reconstruction, same reasoning as jsonb_path_check.py's dotted-key
        # self-apply. Every other mismatch kind still needs reconstruction's
        # judgment (which real value/scale fits the question's intent).
        sql_before_self_apply = sql_code
        sql_code, mismatches = try_self_apply_integer_division_fixes(
            mismatches, sql_code, dialect, known_types=known_types
        )
        if sql_code != sql_before_self_apply:
            _set_sql_code(path_state, sql_code)
            self.logger.info(
                "[%s] Proactive check — self-applied integer-division cast fix(es), "
                "%d mismatch(es) still need reconstruction",
                path_state.get("task_id", "?"),
                len(mismatches),
            )

        if not mismatches:
            # Every mismatch was self-applied — nothing left for reconstruction.
            path_state["value_repair_attempted"] = True
            return {"decision": "valid_sql", "path_state": path_state}

        path_state["value_repair_attempted"] = True
        path_state["error"] = build_value_repair_error(mismatches)
        # This error was derived from a live probe against the query's own
        # already-joined tables — the fix is always "use the real value we
        # just found there," never "go search for a new table." Skip
        # reconstruction's LLM error-classification for it (see
        # sql_reconstruction.py) so it can't be misread as missing_data.
        path_state["error_known_fixable"] = True
        self.logger.info(
            "[%s] Proactive check — routing to reconstruction to fix %d literal(s): %s",
            path_state.get("task_id", "?"),
            len(mismatches),
            [_describe_mismatch(m) for m in mismatches],
        )
        return {"decision": "invalid_sql", "path_state": path_state}


__all__ = ["ProactiveValueCheckAgent"]
