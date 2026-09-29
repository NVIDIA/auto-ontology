# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Merged pre-execution guard: literal/value check, join-path check, and JSONB
key-path check, run in a single graph node instead of three chained ones.

Each sub-check individually routes a failing SQL back to ``reconstruct_sql``
(see ``proactive_value_check.py``, ``join_path_check.py``,
``jsonb_path_check.py``). Chained as separate nodes, a query with e.g. both a
bad join *and* a bad literal costs two full reconstruction round-trips to
fix — each one only sees one problem at a time, which can also cause a fix
for one to be discovered while blind to the other still being wrong (see the
turn-4 oscillation in the archeology_scan_8 trace: the same join error
recurred two reconstruction cycles after it first appeared). Running all
three here and combining every failure into one ``path_state["error"]``
message means reconstruction gets the full picture in a single pass, and the
graph spends one node hop instead of up to three per cycle.

The value/literal check is independent of join correctness by construction —
it probes each filtered column's own table in isolation
(``db_probe/literal_check.py``'s ``_fetch_distinct``/``_fetch_range`` never
execute the query's ``JOIN``), so it always runs when enabled, regardless of
whether the join check passes. The JSONB check is *not* independent: it
validates ``->``/``->>`` key paths against the query's already-joined
tables, so if the join topology is wrong those tables (and the join the
JSONB path was checked against) are about to be rewritten anyway. It is
skipped for this cycle whenever the join check fails, matching the ordering
rationale already documented on ``JoinPathCheckAgent``.

Each sub-check keeps its own attempt-limiting state
(``join_path_repair_attempts``, ``jsonb_path_repair_attempts``,
``value_repair_attempted``) exactly as before, since this node calls the
existing agent classes' ``execute`` directly rather than reimplementing
their logic — only the orchestration (which run, in what order, how their
errors combine) is new.
"""

from typing import Any, Dict

from auto_ontology.retrieval.text_to_sql.agents.empty_like_result_check import (
    _get_sql_code,
)
from auto_ontology.retrieval.text_to_sql.agents.join_path_check import (
    JoinPathCheckAgent,
)
from auto_ontology.retrieval.text_to_sql.agents.jsonb_path_check import (
    JsonbPathCheckAgent,
)
from auto_ontology.retrieval.text_to_sql.agents.proactive_value_check import (
    ProactiveValueCheckAgent,
)
from auto_ontology.retrieval.text_to_sql.base import BaseAgent
from auto_ontology.retrieval.text_to_sql.db_probe.config import (
    is_db_probe_jsonb_path_check,
    is_db_probe_join_path_check,
    is_db_probe_proactive,
)
from auto_ontology.retrieval.text_to_sql.state import AgentState


class CombinedPrecheckAgent(BaseAgent):
    """Run the value, join-path, and JSONB-path prechecks as one node."""

    def __init__(self) -> None:
        super().__init__("combined_precheck")
        self._value_check = ProactiveValueCheckAgent()
        self._join_check = JoinPathCheckAgent()
        self._jsonb_check = JsonbPathCheckAgent()

    def validate_input(self, state: AgentState) -> bool:
        return True

    def execute(self, state: AgentState) -> Dict[str, Any]:
        path_state = dict(state.get("path_state", {}))
        sql_code = _get_sql_code(path_state)
        if not sql_code.strip():
            return {"decision": "valid_sql", "path_state": path_state}

        working_state: Dict[str, Any] = dict(state)
        working_state["path_state"] = path_state

        error_sections: list[str] = []
        any_invalid = False
        known_fixable = True

        def run_subcheck(agent: BaseAgent) -> bool:
            """Run one sub-check, fold its result into the shared state, and
            return whether it flagged the SQL as invalid."""
            nonlocal path_state, working_state, known_fixable
            result = agent.execute(working_state)
            path_state = result.get("path_state", path_state)
            working_state["path_state"] = path_state
            if result.get("decision") != "invalid_sql":
                return False
            error_sections.append(path_state.get("error") or "")
            known_fixable = known_fixable and bool(
                path_state.get("error_known_fixable")
            )
            return True

        if is_db_probe_proactive():
            any_invalid = run_subcheck(self._value_check) or any_invalid

        join_ok = True
        if is_db_probe_join_path_check():
            join_failed = run_subcheck(self._join_check)
            any_invalid = join_failed or any_invalid
            join_ok = not join_failed

        if join_ok and is_db_probe_jsonb_path_check():
            any_invalid = run_subcheck(self._jsonb_check) or any_invalid

        if not any_invalid:
            return {"decision": "valid_sql", "path_state": path_state}

        sections = [s for s in error_sections if s]
        if len(sections) > 1:
            # More than one check failed on the same SQL — without this,
            # reconstruction sees several unrelated-looking blocks of text
            # back to back and, in practice, tends to fix only the first one
            # it recognizes rather than treating them as a joint requirement.
            header = f"{len(sections)} separate issues were found in this SQL. Fix all."
            path_state["error"] = header + "\n\n" + "\n\n".join(sections)
        else:
            path_state["error"] = "\n\n".join(sections)
        if known_fixable:
            path_state["error_known_fixable"] = True
        else:
            path_state.pop("error_known_fixable", None)
        return {"decision": "invalid_sql", "path_state": path_state}
