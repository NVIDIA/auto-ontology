# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``run_steps``: the agent first, then the node for its outcome, each reported
as it starts and as it ends."""

from typing import Any

from auto_ontology.retrieval.text_to_sql.node_labels import NODE_LABELS
from auto_ontology.retrieval.text_to_sql.pipeline import create_steps, run_steps


def _steps(decision: str, calls: list[str]) -> dict[str, Any]:
    def node(name: str, output: dict[str, Any]) -> Any:
        def run(_state: Any) -> dict[str, Any]:
            calls.append(name)
            return output

        return run

    return {
        "react_sql_agent": node(
            "react_sql_agent",
            {"path_state": {"sql_code": "SELECT 1"}, "decision": decision},
        ),
        "format_and_respond": node("format_and_respond", {"path_state": {"done": 1}}),
        "unconstructable_sql_response": node("unconstructable_sql_response", {}),
    }


def _phases(decision: str) -> list[tuple[str, str]]:
    state: Any = {"path_state": {}, "decision": ""}
    steps = _steps(decision, [])
    return [(phase, name) for phase, name, _ in run_steps(state, steps)]


def test_an_answer_is_formatted() -> None:
    assert _phases("valid_sql") == [
        ("start", "react_sql_agent"),
        ("end", "react_sql_agent"),
        ("start", "format_and_respond"),
        ("end", "format_and_respond"),
    ]


def test_no_answer_is_reported_as_unconstructable() -> None:
    assert _phases("unconstructable") == [
        ("start", "react_sql_agent"),
        ("end", "react_sql_agent"),
        ("start", "unconstructable_sql_response"),
        ("end", "unconstructable_sql_response"),
    ]


def test_start_is_reported_before_the_node_runs() -> None:
    """That lead time is what lets a client label the work in progress."""
    calls: list[str] = []
    state: Any = {"path_state": {}, "decision": ""}
    steps = run_steps(state, _steps("valid_sql", calls))

    assert next(steps)[:2] == ("start", "react_sql_agent")
    assert calls == []


def test_each_output_is_folded_into_the_state() -> None:
    state: Any = {"path_state": {}, "decision": ""}
    list(run_steps(state, _steps("valid_sql", [])))

    assert state["decision"] == "valid_sql"
    assert state["path_state"] == {"done": 1}


def test_every_node_has_a_streaming_label() -> None:
    assert set(create_steps()) <= set(NODE_LABELS)
