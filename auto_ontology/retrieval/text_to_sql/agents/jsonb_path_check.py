# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Proactive (pre-execution) JSONB key-path check.

Sibling of ``proactive_value_check.py``, checking ``->``/``->>`` JSONB key
paths instead of filter literals. Runs before execution so it also catches
paths that would return non-empty-but-wrong-because-NULL columns, not just
paths that happen to empty out the whole result. Gated by
``DB_PROBE_JSONB_PATH_CHECK`` since it costs a few cheap probes on every query
that navigates JSONB. Runs at most ``_MAX_REPAIR_ATTEMPTS`` times per request —
mirrors ``join_path_check.py``'s bounded counter rather than a one-shot flag,
since a reconstruction can trade one wrong key for another (a real key that
exists but is the wrong sibling) and a second pass, now armed with the
sibling-container list the first mismatch surfaced, often converges. The
graph's shared failed-attempt cap (see
``text_to_sql_graph._make_soft_check_router``) already bounds the
pathological case, so this only needs to stop *this* check from being the
sole thing driving many of those reconstructions, not prevent looping itself.
"""

from __future__ import annotations

from typing import Any, Dict

from auto_ontology.retrieval.text_to_sql.agents.empty_like_result_check import (
    _get_sql_code,
    _set_sql_code,
)
from auto_ontology.retrieval.text_to_sql.base import BaseAgent
from auto_ontology.retrieval.text_to_sql.connector_routing import (
    resolve_connector_from_tables,
)
from auto_ontology.retrieval.text_to_sql.db_probe.executor import ProbeExecutor
from auto_ontology.retrieval.text_to_sql.db_probe.jsonb_path_check import (
    build_jsonb_path_repair_error,
    find_jsonb_path_mismatches,
    try_self_apply_fixes,
)
from auto_ontology.retrieval.text_to_sql.state import AgentState

# Same cap as join_path_check.py's _MAX_REPAIR_ATTEMPTS, and for the same
# reason: bounded rather than one-shot, since a reconstruction can trade one
# wrong key for another (a real sibling key that exists but isn't the one
# meant) rather than converging on the first attempt.
_MAX_REPAIR_ATTEMPTS = 2


class JsonbPathCheckAgent(BaseAgent):
    """Check JSONB key paths against real DB keys before executing the query."""

    def __init__(self) -> None:
        super().__init__("jsonb_path_check")

    def validate_input(self, state: AgentState) -> bool:
        return True

    def execute(self, state: AgentState) -> Dict[str, Any]:
        path_state = dict(state.get("path_state", {}))
        sql_code = _get_sql_code(path_state)

        attempts = path_state.get("jsonb_path_repair_attempts", 0)
        if not sql_code.strip() or attempts >= _MAX_REPAIR_ATTEMPTS:
            return {"decision": "valid_sql", "path_state": path_state}

        connectors = state.get("connectors") or []
        relevant_tables = list(path_state.get("relevant_tables") or [])
        connector = resolve_connector_from_tables(relevant_tables, connectors)
        dialect = getattr(connector, "dialect", None)

        # Cheap, already-fetched — no extra DB/graph round-trip. Lets the
        # check distinguish "wrong key" from "not a JSON column at all" when
        # the live probe itself can't tell (see db_probe/jsonb_path_check.py).
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
            mismatches = find_jsonb_path_mismatches(
                executor, dialect, sql_code, known_types=known_types
            )

        if not mismatches:
            return {"decision": "valid_sql", "path_state": path_state}

        # Self-apply the one mismatch shape with a single unambiguous fix (a
        # flattened dotted key that's really nested access) before falling
        # back to reconstruction — see try_self_apply_fixes for why only
        # this shape is safe to fix mechanically. Cuts the reconstruct_sql →
        # validate_sql_query → execute_sql_query round-trip entirely for the
        # ~93% of historical firings that are this exact shape (measured
        # against full_125: 13/14 fixes reconstruction made on its own
        # matched this pattern anyway).
        fixed_sql, mismatches = try_self_apply_fixes(mismatches, sql_code)
        if fixed_sql != sql_code:
            _set_sql_code(path_state, fixed_sql)
            sql_code = fixed_sql
            self.logger.info(
                "[%s] JSONB path check — self-applied flattened-key fix(es), "
                "%d path(s) still need reconstruction",
                path_state.get("task_id", "?"),
                len(mismatches),
            )

        path_state["jsonb_path_repair_attempts"] = attempts + 1

        if not mismatches:
            # Every mismatch was self-applied — nothing left for reconstruction.
            return {"decision": "valid_sql", "path_state": path_state}

        path_state["error"] = build_jsonb_path_repair_error(mismatches)
        # This error was derived from a live probe against the query's own
        # already-joined tables — the fix is always "use the real key/
        # container we just found there," never "go search for a new
        # table." Skip reconstruction's LLM error-classification for it
        # (see sql_reconstruction.py) so it can't be misread as missing_data.
        path_state["error_known_fixable"] = True
        self.logger.info(
            "[%s] JSONB path check — routing to reconstruction to fix %d path(s): %s",
            path_state.get("task_id", "?"),
            len(mismatches),
            [
                f"{m['table']}.{m['column']}->'{m['container']}'->>'{m['used_key']}'"
                if m["container"]
                else f"{m['table']}.{m['column']}->>'{m['used_key']}'"
                for m in mismatches
            ],
        )
        return {"decision": "invalid_sql", "path_state": path_state}


__all__ = ["JsonbPathCheckAgent"]
