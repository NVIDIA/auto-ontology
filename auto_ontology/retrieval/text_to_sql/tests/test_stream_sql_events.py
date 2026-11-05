# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stream events: each node's start and end, the submitted SQL, then the answer.

The ``sql`` event fires once the agent has submitted its query, never for the
drafts it ran while exploring.
"""

import importlib
from types import ModuleType
from typing import Any, Iterator, cast

import pytest

from auto_ontology.retrieval.text_to_sql.state import TextToSQLPayload
from auto_ontology.utils import llm_invoke


@pytest.fixture(name="main")
def main_fixture(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Import the agent entry point without a configured LLM.

    ``main`` builds its reasoning client at import time and lets an unset
    ``REASONING_API_KEY`` raise, so importing it at module scope would fail
    collection on any machine without credentials. Nothing here calls the
    client — the pipeline itself is replaced per test.
    """

    monkeypatch.setattr(llm_invoke, "get_llm_client", lambda **_kwargs: None)
    monkeypatch.setattr(
        llm_invoke, "get_non_reasoning_llm_client", lambda **_kwargs: None
    )
    return importlib.import_module("auto_ontology.retrieval.text_to_sql.main")


def _answer_step(sql: str, decision: str = "valid_sql") -> dict[str, Any]:
    path_state = {"sql_code": sql} if sql else {}
    return {"react_sql_agent": {"path_state": path_state, "decision": decision}}


def _stream_items(steps: list[dict[str, Any]]) -> Iterator[tuple[str, str, Any]]:
    """Replay ``steps`` the way ``run_steps`` reports them: each node's start
    as it begins, then its output when it returns."""
    for step in steps:
        for step_name, output in step.items():
            yield ("start", step_name, None)
            yield ("end", step_name, output)


def _use_stream(
    main: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    items: Iterator[tuple[str, str, Any]],
) -> None:
    """``_build_state`` needs real connectors/retrievers and the steps need a
    live LLM — neither is what's under test, so both are replaced."""
    monkeypatch.setattr(main, "_build_state", lambda payload: {"path_state": {}})
    monkeypatch.setattr(main, "run_steps", lambda _state, _steps: items)


def _run(
    main: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    steps: list[dict[str, Any]],
) -> list[dict]:
    _use_stream(main, monkeypatch, _stream_items(steps))
    payload = cast(TextToSQLPayload, {"question": "how many orders?"})
    return list(main.stream_agent_response(payload))


def _sql_events(events: list[dict]) -> list[tuple[str, str]]:
    return [(e["node"], e["sql"]) for e in events if e["type"] == "sql"]


def test_the_submitted_sql_is_emitted_after_the_agent_step(
    main: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = _run(main, monkeypatch, [_answer_step("SELECT 1")])

    assert _sql_events(events) == [("react_sql_agent", "SELECT 1")]
    kinds = [(e["type"], e.get("phase") or e.get("node")) for e in events]
    assert kinds.index(("step", "end")) < kinds.index(("sql", "react_sql_agent"))


def test_an_unanswered_question_emits_no_sql(
    main: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = _run(
        main,
        monkeypatch,
        [
            _answer_step("", decision="unconstructable"),
            {"unconstructable_sql_response": {"path_state": {}}},
        ],
    )

    assert _sql_events(events) == []


def test_other_nodes_never_emit_sql(
    main: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = _run(
        main,
        monkeypatch,
        [{"format_and_respond": {"path_state": {"sql_code": "SELECT 1"}}}],
    )

    assert _sql_events(events) == []


def test_stream_ends_with_a_result_event(
    main: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = _run(
        main,
        monkeypatch,
        [
            _answer_step("SELECT 42"),
            {
                "format_and_respond": {
                    "path_state": {
                        "final_response": {
                            "response": "42",
                            "sql_code": "SELECT 42",
                        },
                    }
                }
            },
        ],
    )

    assert events[-1]["type"] == "result"
    assert events[-1]["answer"]["sql_code"] == "SELECT 42"


def test_generator_iterates_lazily(
    main: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Events must be yielded as they arrive, or nothing shows up live."""

    consumed: list[str] = []

    def items() -> Iterator[tuple[str, str, Any]]:
        consumed.append("first")
        yield from _stream_items([_answer_step("SELECT 1")])
        consumed.append("second")
        yield from _stream_items([{"format_and_respond": {"path_state": {}}}])

    _use_stream(main, monkeypatch, items())
    stream = main.stream_agent_response(cast(TextToSQLPayload, {"question": "q"}))

    assert next(stream)["phase"] == "start"
    assert next(stream)["phase"] == "end"
    assert next(stream)["type"] == "sql"
    assert consumed == ["first"]


def test_each_node_reports_start_then_end(
    main: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pair is what keeps the visible label honest: ``start`` names the
    work in progress, ``end`` carries the thought it produced."""

    events = _run(
        main,
        monkeypatch,
        [
            {
                "react_sql_agent": {
                    "path_state": {
                        "thoughts_log": [
                            {"node": "react_sql_agent", "text": "Searching for: orders"}
                        ],
                    },
                    "decision": "unconstructable",
                }
            }
        ],
    )

    steps = [e for e in events if e["type"] == "step"]
    assert [(s["phase"], s["node"], s["thought"]) for s in steps] == [
        ("start", "react_sql_agent", None),
        ("end", "react_sql_agent", "Searching for: orders"),
    ]


def test_start_precedes_the_nodes_own_update(
    main: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A slow node's ``start`` has to reach the client before that node
    returns — that lead time is the whole reason the event exists."""

    order: list[str] = []

    def items() -> Iterator[tuple[str, str, Any]]:
        yield ("start", "react_sql_agent", None)
        order.append("node finished")
        yield ("end", "react_sql_agent", {"path_state": {}})

    _use_stream(main, monkeypatch, items())
    stream = main.stream_agent_response(cast(TextToSQLPayload, {"question": "q"}))
    first = next(stream)

    assert (first["phase"], first["node"]) == ("start", "react_sql_agent")
    assert order == []  # the node had not returned yet
