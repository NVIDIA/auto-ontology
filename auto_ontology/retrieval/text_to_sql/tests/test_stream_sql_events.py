# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``sql`` events: the query the agent is about to run, streamed pre-execution.

The event fires only once a node has cleared the query for execution, never at
generation time — a draft that unified validation is about to reject must not
reach the client, since it is not what runs.
"""

import importlib
from types import ModuleType, SimpleNamespace
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
    client — the graph itself is replaced per test.

    ``_COMBINED_PRECHECK_IN_GRAPH`` is pinned off for the same reason it
    is pinned on in the tests that want it: it is derived from the graph
    ``main`` built at import, so ``DB_PROBE_PROACTIVE`` in the ambient
    environment would otherwise decide which node these tests expect the
    query to be cleared by. Tests covering the precheck branch override it.
    """

    monkeypatch.setattr(llm_invoke, "get_llm_client", lambda **_kwargs: None)
    module = importlib.import_module("auto_ontology.retrieval.text_to_sql.main")
    monkeypatch.setattr(module, "_COMBINED_PRECHECK_IN_GRAPH", False)
    return module


def _generated(sql: str) -> SimpleNamespace:
    """Stand-in for the Pydantic SQL response the generation agents return."""
    return SimpleNamespace(sql_code=sql)


def _generation_step(sql: str) -> dict[str, Any]:
    return {
        "construct_sql_from_candidates": {
            "path_state": {"sql_generation_result": _generated(sql)},
            "decision": "validate_sql_query",
        }
    }


def _validation_ok_step(sql: str, failed_attempt_count: int = 0) -> dict[str, Any]:
    return {
        "validate_sql_query": {
            "path_state": {
                "sql_generation_result": _generated(sql),
                "sql_code": sql,
                "failed_attempts": [{} for _ in range(failed_attempt_count)],
            },
            "decision": "valid_sql",
        }
    }


def _stream_items(steps: list[dict[str, Any]]) -> Iterator[tuple[str, Any]]:
    """Replay ``steps`` the way the compiled graph does.

    With ``stream_mode=["updates", "custom"]`` LangGraph yields ``(mode,
    chunk)``: the node's own start announcement on ``custom`` as it begins,
    then its state update on ``updates`` when it returns. Reproduced here so
    the tests exercise the real ordering rather than a convenient one.
    """
    for step in steps:
        for node_name in step:
            yield ("custom", {"type": "step_start", "node": node_name})
        yield ("updates", step)


def _run(
    main: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    steps: list[dict[str, Any]],
) -> list[dict]:
    """Stream ``steps`` through ``stream_agent_response`` and collect events.

    ``_build_state`` needs real connectors/retrievers and the compiled graph
    needs a live LLM — neither is what's under test, so both are replaced.
    """

    monkeypatch.setattr(main, "_build_state", lambda payload: {"path_state": {}})
    monkeypatch.setattr(
        main,
        "app",
        SimpleNamespace(
            stream=lambda _state, stream_mode=None, config=None: _stream_items(steps)
        ),
    )
    payload = cast(TextToSQLPayload, {"question": "how many orders?"})
    return list(main.stream_agent_response(payload))


def _sql_events(events: list[dict]) -> list[tuple[str, str]]:
    return [(e["node"], e["sql"]) for e in events if e["type"] == "sql"]


def test_sql_is_emitted_once_unified_validation_clears_it(
    main: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The query reaches the client before execution, not with the answer."""

    events = _run(
        main,
        monkeypatch,
        [
            _generation_step("SELECT 1"),
            _validation_ok_step("SELECT 1"),
            {"execute_sql_query": {"path_state": {"sql_code": "SELECT 1"}}},
        ],
    )

    assert _sql_events(events) == [("validate_sql_query", "SELECT 1")]

    # It has to land before the node that runs it, or it isn't "live".
    kinds = [(e["type"], e.get("node")) for e in events]
    assert kinds.index(("sql", "validate_sql_query")) < kinds.index(
        ("step", "execute_sql_query")
    )


def test_generation_alone_emits_nothing(
    main: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A draft is not shown — validation may still send it back."""

    events = _run(main, monkeypatch, [_generation_step("SELECT 1")])

    assert _sql_events(events) == []


def test_rejected_draft_never_reaches_the_client(
    main: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the query that survives unified validation is shown.

    The first attempt is semantically wrong, so validation routes it to
    reconstruction. Showing it would put a query on
    screen that never runs.
    """

    events = _run(
        main,
        monkeypatch,
        [
            _generation_step("SELECT bad"),
            {
                "validate_sql_query": {
                    "path_state": {
                        "sql_generation_result": _generated("SELECT bad"),
                        "sql_code": "SELECT bad",
                    },
                    "decision": "invalid_sql",
                }
            },
            {
                "reconstruct_sql": {
                    "path_state": {
                        "sql_generation_result": _generated("SELECT good"),
                        "sql_code": "SELECT bad",
                    },
                    "decision": "validate_sql_query",
                }
            },
            _validation_ok_step("SELECT good"),
        ],
    )

    assert _sql_events(events) == [("validate_sql_query", "SELECT good")]


def _precheck_step(sql: str, decision: str) -> dict[str, Any]:
    return {
        "precheck_combined": {
            "path_state": {"sql_code": sql},
            "decision": decision,
        }
    }


def test_proactive_value_check_is_the_only_gate_when_enabled(
    main: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the proactive check in the graph it, not validation, is the last hop."""

    monkeypatch.setattr(main, "_COMBINED_PRECHECK_IN_GRAPH", True)

    events = _run(
        main,
        monkeypatch,
        [_validation_ok_step("SELECT 1"), _precheck_step("SELECT 1", "valid_sql")],
    )

    assert _sql_events(events) == [("precheck_combined", "SELECT 1")]


def test_proactive_rejection_never_shows_the_query(
    main: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A literal mismatch sends a validated query to reconstruction.

    Emitting at validation would put a query on screen that the
    proactive check is about to reject — only the rewrite that survives it
    ever runs.
    """

    monkeypatch.setattr(main, "_COMBINED_PRECHECK_IN_GRAPH", True)

    events = _run(
        main,
        monkeypatch,
        [
            _validation_ok_step("SELECT bad_literal"),
            _precheck_step("SELECT bad_literal", "invalid_sql"),
            {
                "reconstruct_sql": {
                    "path_state": {
                        "sql_generation_result": _generated("SELECT good"),
                        "sql_code": "SELECT bad_literal",
                    },
                    "decision": "validate_sql_query",
                }
            },
            _validation_ok_step("SELECT good"),
            _precheck_step("SELECT good", "valid_sql"),
        ],
    )

    assert _sql_events(events) == [("precheck_combined", "SELECT good")]


def test_validation_waits_for_the_proactive_check(
    main: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The unified validation pass is not the final gate when prechecks exist."""

    monkeypatch.setattr(main, "_COMBINED_PRECHECK_IN_GRAPH", True)

    events = _run(main, monkeypatch, [_validation_ok_step("SELECT 1")])

    assert _sql_events(events) == []


def test_proactive_check_emits_without_a_preceding_validation_update(
    main: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """As the configured final gate, the precheck emits the SQL it clears."""

    monkeypatch.setattr(main, "_COMBINED_PRECHECK_IN_GRAPH", True)

    events = _run(main, monkeypatch, [_precheck_step("SELECT 1", "valid_sql")])

    assert _sql_events(events) == [("precheck_combined", "SELECT 1")]


def test_proactive_node_is_ignored_when_not_in_the_graph(
    main: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default build: unified validation is the last gate and emits there."""

    events = _run(
        main,
        monkeypatch,
        [_validation_ok_step("SELECT 1"), _precheck_step("SELECT 1", "valid_sql")],
    )

    assert _sql_events(events) == [("validate_sql_query", "SELECT 1")]


def test_unified_validation_emits_after_many_reconstructions(
    main: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The unified node remains the final gate when its LLM phase is skipped."""

    events = _run(main, monkeypatch, [_validation_ok_step("SELECT 1", 6)])

    assert _sql_events(events) == [("validate_sql_query", "SELECT 1")]


def test_invalid_unified_validation_does_not_emit(
    main: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed unified validation result is never shown as executable SQL."""

    events = _run(
        main,
        monkeypatch,
        [
            {
                "validate_sql_query": {
                    "path_state": {"sql_code": "SELECT 1"},
                    "decision": "invalid_sql",
                }
            }
        ],
    )

    assert _sql_events(events) == []


def test_unchanged_sql_is_not_re_emitted(
    main: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty result re-runs the same query through the same gate."""

    events = _run(
        main,
        monkeypatch,
        [
            _validation_ok_step("SELECT 1"),
            {"execute_sql_query": {"path_state": {"sql_code": "SELECT 1"}}},
            _validation_ok_step("SELECT 1"),
        ],
    )

    assert _sql_events(events) == [("validate_sql_query", "SELECT 1")]


def test_a_rebuilt_query_replaces_the_previous_one(
    main: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Execution failure sends the run back for reconstruction; the retry's
    query is emitted once it clears validation again."""

    events = _run(
        main,
        monkeypatch,
        [
            _validation_ok_step("SELECT a"),
            {
                "execute_sql_query": {
                    "path_state": {"sql_code": "SELECT a", "error": "boom"},
                    "decision": "invalid_sql",
                }
            },
            _validation_ok_step("SELECT b"),
        ],
    )

    assert _sql_events(events) == [
        ("validate_sql_query", "SELECT a"),
        ("validate_sql_query", "SELECT b"),
    ]


def test_nodes_without_sql_emit_nothing(
    main: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = _run(
        main,
        monkeypatch,
        [
            {"retrieve_candidates": {"path_state": {"relevant_tables": []}}},
            {"prepare_candidates": None},
        ],
    )

    assert _sql_events(events) == []


def test_validation_without_sql_emits_nothing(
    main: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A validation update without SQL must not produce an empty event."""

    events = _run(
        main,
        monkeypatch,
        [{"validate_sql_query": {"path_state": {}, "decision": "valid_sql"}}],
    )

    assert _sql_events(events) == []


def test_stream_ends_with_a_result_event(
    main: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The final answer still arrives the way it always did."""

    events = _run(
        main,
        monkeypatch,
        [
            {
                "format_and_respond": {
                    "path_state": {
                        "final_response": {
                            "response": "42",
                            "sql_code": "SELECT 42",
                        },
                    }
                }
            }
        ],
    )

    assert events[-1]["type"] == "result"
    assert events[-1]["answer"]["sql_code"] == "SELECT 42"


def test_generator_iterates_lazily(
    main: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Events must be yielded as they arrive, or nothing shows up live."""

    consumed: list[str] = []

    def items() -> Iterator[tuple[str, Any]]:
        consumed.append("first")
        yield from _stream_items([_validation_ok_step("SELECT 1")])
        consumed.append("second")
        yield from _stream_items(
            [{"execute_sql_query": {"path_state": {"sql_code": "SELECT 1"}}}]
        )

    monkeypatch.setattr(main, "_build_state", lambda payload: {"path_state": {}})
    monkeypatch.setattr(
        main,
        "app",
        SimpleNamespace(stream=lambda _state, stream_mode=None, config=None: items()),
    )

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
                "reconstruct_sql": {
                    "path_state": {
                        "sql_generation_result": _generated("SELECT good"),
                        "thoughts_log": [
                            {"node": "reconstruct_sql", "text": "fixing the join"}
                        ],
                    },
                    "decision": "validate_sql_query",
                }
            }
        ],
    )

    steps = [e for e in events if e["type"] == "step"]
    assert [(s["phase"], s["node"], s["thought"]) for s in steps] == [
        ("start", "reconstruct_sql", None),
        ("end", "reconstruct_sql", "fixing the join"),
    ]


def test_entry_router_is_not_streamed(
    main: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The entry router only selects a graph path and is invisible to clients."""

    events = _run(
        main,
        monkeypatch,
        [{"_entry_router": {"path_state": {"final_response": {"response": "routed"}}}}],
    )

    assert [event["type"] for event in events] == ["result"]
    assert events[0]["answer"] == {"response": "routed"}


def test_start_precedes_the_nodes_own_update(
    main: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A slow node's ``start`` has to reach the client before that node
    returns — that lead time is the whole reason the event exists."""

    order: list[str] = []

    def items() -> Iterator[tuple[str, Any]]:
        yield ("custom", {"type": "step_start", "node": "reconstruct_sql"})
        order.append("node finished")
        yield ("updates", {"reconstruct_sql": {"path_state": {}}})

    monkeypatch.setattr(main, "_build_state", lambda payload: {"path_state": {}})
    monkeypatch.setattr(
        main,
        "app",
        SimpleNamespace(stream=lambda _state, stream_mode=None, config=None: items()),
    )

    stream = main.stream_agent_response(cast(TextToSQLPayload, {"question": "q"}))
    first = next(stream)

    assert (first["phase"], first["node"]) == ("start", "reconstruct_sql")
    assert order == []  # the node had not returned yet


def test_unknown_custom_payloads_are_ignored(
    main: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The custom channel is shared; anything that isn't a start announcement
    must not turn into a phantom step."""

    def items() -> Iterator[tuple[str, Any]]:
        yield ("custom", {"type": "something_else", "node": "reconstruct_sql"})
        yield ("custom", {"type": "step_start"})  # no node name
        yield from _stream_items([_validation_ok_step("SELECT 1")])

    monkeypatch.setattr(main, "_build_state", lambda payload: {"path_state": {}})
    monkeypatch.setattr(
        main,
        "app",
        SimpleNamespace(stream=lambda _state, stream_mode=None, config=None: items()),
    )

    events = list(main.stream_agent_response(cast(TextToSQLPayload, {"question": "q"})))
    steps = [e for e in events if e["type"] == "step"]

    assert [s["node"] for s in steps] == [
        "validate_sql_query",
        "validate_sql_query",
    ]
