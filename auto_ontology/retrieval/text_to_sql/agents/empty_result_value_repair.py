# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Empty-result value repair, and the last-resort retry behind it.

Runs right after execution, only when the query came back empty. It checks the
filter literals against the columns' actual distinct values and, when a literal
is a near-miss of a real value (wrong case / spelling), routes once to the
existing reconstruction node with a targeted hint. On the common path (non-empty
result) it is a single boolean check — no probes, no LLM call.

This is the last node before the response, so when the result is still empty and
neither this check nor the empty-LIKE one found anything to fix, it sends the
query back for one undiagnosed regeneration rather than returning nothing. The
measured causes of a surviving empty result are not literal typos: an extra
join predicate the question does not state, a date compared in the wrong format,
or a filter on the wrong column of the same table. None are repairable from a
probe. Unlike the targeted paths above, this retry omits ``error_known_fixable``
so a genuinely absent concept can still be searched. It must not treat a sibling
table — a summary, a current-only subset, or a similarly named process — as that
missing concept.

An empty result can legitimately be the correct answer, so the retry is
one-shot and never rejects the regenerated query for also coming back empty —
the cost of being wrong here is one extra reconstruction, not a lost answer.
"""

from __future__ import annotations

from typing import Any, Dict

from auto_ontology.retrieval.text_to_sql.agents.empty_like_result_check import (
    _get_sql_code,
    _is_empty_db_result,
)
from auto_ontology.retrieval.text_to_sql.base import BaseAgent
from auto_ontology.retrieval.text_to_sql.connector_routing import (
    resolve_connector_from_tables,
)
from auto_ontology.retrieval.text_to_sql.db_probe.executor import ProbeExecutor
from auto_ontology.retrieval.text_to_sql.db_probe.literal_check import (
    build_value_repair_error,
    find_literal_mismatches,
    find_numeric_scale_mismatches,
)
from auto_ontology.retrieval.text_to_sql.state import AgentState


def build_empty_result_retry_error() -> str:
    """Instruction for an empty result no check could diagnose.

    Names the causes that empty a query whose literals are real: an extra join
    predicate, a format mismatch, or a filter on the wrong column of the same
    table. A same-named column on a sibling table is a different population and
    is not a repair.
    """
    return (
        "SQL executed successfully but returned an empty result set. The query "
        "is valid and its filter literals were checked against the database, so "
        "this is not a typo in a value. Re-derive it from the schema, and "
        "reconsider in particular:\n"
        "- whether a join predicate the question does not state is dropping "
        "every row. Remove that predicate. Do not add a new one;\n"
        "- whether a date, time, or number is compared in the format and units "
        "the column really uses;\n"
        "- whether the filtered value is stored in a different column of the "
        "SAME table.\n"
        "Do not move the filter onto a same-named column of another table. A "
        "summary, a current-only subset, a short name, or a similarly named "
        "process is a different population. An empty result is not evidence "
        "that the value lives there."
    )


class EmptyResultValueRepairAgent(BaseAgent):
    """Repair a bad filter literal, or regenerate an empty result outright."""

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

        mismatches: list[dict[str, Any]] = []
        if path_state.get("value_repair_attempted"):
            self.logger.info("Value repair already attempted — skipping the probe")
        else:
            connectors = state.get("connectors") or []
            relevant_tables = list(path_state.get("relevant_tables") or [])
            connector = resolve_connector_from_tables(relevant_tables, connectors)
            dialect = getattr(connector, "dialect", None)

            with ProbeExecutor(connector) as executor:
                # Reached only when the query came back empty, so the extra
                # high-cardinality literal check is warranted here (and only here).
                mismatches = find_literal_mismatches(
                    executor, dialect, sql_code, result_was_empty=True
                )
                mismatches += find_numeric_scale_mismatches(executor, dialect, sql_code)

        if mismatches:
            path_state["value_repair_attempted"] = True
            path_state["error"] = build_value_repair_error(mismatches)
            # This error was derived from a live probe against the query's own
            # already-joined tables — the fix is always "use the real value we
            # just found there," never "go search for a new table." Skip
            # reconstruction's LLM error-classification for it (see
            # sql_reconstruction.py) so it can't be misread as missing_data.
            path_state["error_known_fixable"] = True
            self.logger.info(
                "[%s] Empty result — routing to reconstruction to fix %d literal(s): %s",
                path_state.get("task_id", "?"),
                len(mismatches),
                [f"{m['table']}.{m['column']}='{m['used']}'" for m in mismatches],
            )
            return {"decision": "invalid_sql", "path_state": path_state}

        if path_state.get("empty_result_retry_attempted"):
            self.logger.info(
                "Empty result persisted after an undiagnosed retry — passing through"
            )
            return {"decision": "valid_sql", "path_state": path_state}

        # Nothing diagnosable, and an empty result answers nothing. Regenerate
        # once, without error_known_fixable so the classifier may look elsewhere.
        path_state["empty_result_retry_attempted"] = True
        path_state["error"] = build_empty_result_retry_error()
        self.logger.info(
            "[%s] Empty result with no diagnosable cause — routing to "
            "reconstruction for one undiagnosed retry",
            path_state.get("task_id", "?"),
        )
        return {"decision": "invalid_sql", "path_state": path_state}


__all__ = ["EmptyResultValueRepairAgent", "build_empty_result_retry_error"]
