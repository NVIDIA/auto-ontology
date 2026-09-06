# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``sql`` events: the query the agent is about to run, streamed pre-execution.

The event fires only once a node has cleared the query for execution, never at
generation time — a draft that intent validation is about to reject must not
reach the client, since it is not what runs.
"""

import importlib
from types import ModuleType, SimpleNamespace
from typing import Any, Iterator, cast

import pytest

from gsf.retrieval.text_to_sql.state import TextToSQLPayload
from gsf.utils import llm_invoke


@pytest.fixture(name="main")
def main_fixture(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Import the agent entry point without a configured LLM.

    ``main`` builds its reasoning client at import time and lets an unset
    ``REASONING_API_KEY`` raise, so importing it at module scope would fail
    collection on any machine without credentials. Nothing here calls the
    client — the graph itself is replaced per test.
    """

    monkeypatch.setattr(llm_invoke, "get_llm_client", lambda **_kwargs: None)
    return importlib.import_module("gsf.retrieval.text_to_sql.main")


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


def _syntax_ok_step(sql: str, reconstruction_count: int = 0) -> dict[str, Any]:
    return {
        "validate_sql_query": {
            "path_state": {
                "sql_generation_result": _generated(sql),
                "sql_code": sql,
                "reconstruction_count": reconstruction_count,
            },
            "decision": "valid_sql",
        }
    }


def _intent_ok_step(sql: str) -> dict[str, Any]:
    return {
        "validate_intent": {
            "path_state": {
                "sql_generation_result": _generated(sql),
                "sql_code": sql,
            },
            "decision": "intent_valid",
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


def test_sql_is_emitted_once_intent_validation_clears_it(
    main: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The query reaches the client before execution, not with the answer."""

    events = _run(
        main,
        monkeypatch,
        [
            _generation_step("SELECT 1"),
            _syntax_ok_step("SELECT 1"),
            _intent_ok_step("SELECT 1"),
            {"execute_sql_query": {"path_state": {"sql_code": "SELECT 1"}}},
        ],
    )

    assert _sql_events(events) == [("validate_intent", "SELECT 1")]

    # It has to land before the node that runs it, or it isn't "live".
    kinds = [(e["type"], e.get("node")) for e in events]
    assert kinds.index(("sql", "validate_intent")) < kinds.index(
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
    """Only the query that survives intent validation is shown.

    The first attempt is syntactically fine but semantically wrong, so intent
    validation routes it to reconstruction. Showing it would put a query on
    screen that never runs.
    """

    events = _run(
        main,
        monkeypatch,
        [
            _generation_step("SELECT bad"),
            _syntax_ok_step("SELECT bad"),
            {
                "validate_intent": {
                    "path_state": {
                        "sql_generation_result": _generated("SELECT bad"),
                        "sql_code": "SELECT bad",
                    },
                    "decision": "intent_invalid",
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
            _syntax_ok_step("SELECT good"),
            _intent_ok_step("SELECT good"),
        ],
    )

    assert _sql_events(events) == [("validate_intent", "SELECT good")]


def _precheck_step(sql: str, decision: str) -> dict[str, Any]:
    return {
        "precheck_value_repair": {
            "path_state": {"sql_code": sql},
            "decision": decision,
        }
    }


def test_proactive_value_check_is_the_only_gate_when_enabled(
    main: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the proactive check in the graph it, not intent validation, is the
    last hop, so ``intent_valid`` alone must not emit."""

    monkeypatch.setattr(main, "_PROACTIVE_VALUE_CHECK_IN_GRAPH", True)

    events = _run(
        main,
        monkeypatch,
        [_intent_ok_step("SELECT 1"), _precheck_step("SELECT 1", "valid_sql")],
    )

    assert _sql_events(events) == [("precheck_value_repair", "SELECT 1")]


def test_proactive_rejection_never_shows_the_query(
    main: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A literal mismatch sends an intent-valid query to reconstruction.

    Emitting at ``intent_valid`` would put a query on screen that the
    proactive check is about to reject — only the rewrite that survives it
    ever runs.
    """

    monkeypatch.setattr(main, "_PROACTIVE_VALUE_CHECK_IN_GRAPH", True)

    events = _run(
        main,
        monkeypatch,
        [
            _intent_ok_step("SELECT bad_literal"),
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
            _syntax_ok_step("SELECT good"),
            _intent_ok_step("SELECT good"),
            _precheck_step("SELECT good", "valid_sql"),
        ],
    )

    assert _sql_events(events) == [("precheck_value_repair", "SELECT good")]


def test_skip_intent_branch_waits_for_the_proactive_check(
    main: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Past the reconstruction threshold ``route_sql_validation`` skips intent
    validation, but with the proactive check enabled it still routes through
    that node rather than straight to execution."""

    monkeypatch.setattr(main, "_PROACTIVE_VALUE_CHECK_IN_GRAPH", True)

    events = _run(main, monkeypatch, [_syntax_ok_step("SELECT 1", 6)])

    assert _sql_events(events) == []


def test_proactive_check_emits_without_a_preceding_intent_pass(
    main: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """It is the only gate on the skip-intent branch, so it has to emit."""

    monkeypatch.setattr(main, "_PROACTIVE_VALUE_CHECK_IN_GRAPH", True)

    events = _run(main, monkeypatch, [_precheck_step("SELECT 1", "valid_sql")])

    assert _sql_events(events) == [("precheck_value_repair", "SELECT 1")]


def test_proactive_node_is_ignored_when_not_in_the_graph(
    main: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default build: intent validation is the last gate and emits there."""

    events = _run(
        main,
        monkeypatch,
        [_intent_ok_step("SELECT 1"), _precheck_step("SELECT 1", "valid_sql")],
    )

    assert _sql_events(events) == [("validate_intent", "SELECT 1")]


def test_skipped_intent_validation_still_emits(
    main: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Past 5 reconstructions ``route_sql_validation`` goes straight from
    syntax validation to execution, so that node becomes the last gate."""

    events = _run(main, monkeypatch, [_syntax_ok_step("SELECT 1", 6)])

    assert _sql_events(events) == [("validate_sql_query", "SELECT 1")]


def test_syntax_validation_alone_does_not_emit(
    main: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Below the reconstruction threshold, intent validation runs next and may
    still reject the query."""

    events = _run(main, monkeypatch, [_syntax_ok_step("SELECT 1", 1)])

    assert _sql_events(events) == []


def test_unchanged_sql_is_not_re_emitted(
    main: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty result re-runs the same query through the same gate."""

    events = _run(
        main,
        monkeypatch,
        [
            _intent_ok_step("SELECT 1"),
            {"execute_sql_query": {"path_state": {"sql_code": "SELECT 1"}}},
            _intent_ok_step("SELECT 1"),
        ],
    )

    assert _sql_events(events) == [("validate_intent", "SELECT 1")]


def test_a_rebuilt_query_replaces_the_previous_one(
    main: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Execution failure sends the run back for reconstruction; the retry's
    query is emitted once it clears validation again."""

    events = _run(
        main,
        monkeypatch,
        [
            _intent_ok_step("SELECT a"),
            {
                "execute_sql_query": {
                    "path_state": {"sql_code": "SELECT a", "error": "boom"},
                    "decision": "invalid_sql",
                }
            },
            _intent_ok_step("SELECT b"),
        ],
    )

    assert _sql_events(events) == [
        ("validate_intent", "SELECT a"),
        ("validate_intent", "SELECT b"),
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


def test_intent_valid_without_sql_emits_nothing(
    main: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``IntentValidationAgent`` returns ``intent_valid`` early when there is
    no SQL to check — that must not produce an empty event."""

    events = _run(
        main,
        monkeypatch,
        [{"validate_intent": {"path_state": {}, "decision": "intent_valid"}}],
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
        yield from _stream_items([_intent_ok_step("SELECT 1")])
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
        yield from _stream_items([_intent_ok_step("SELECT 1")])

    monkeypatch.setattr(main, "_build_state", lambda payload: {"path_state": {}})
    monkeypatch.setattr(
        main,
        "app",
        SimpleNamespace(stream=lambda _state, stream_mode=None, config=None: items()),
    )

    events = list(main.stream_agent_response(cast(TextToSQLPayload, {"question": "q"})))
    steps = [e for e in events if e["type"] == "step"]

    assert [s["node"] for s in steps] == ["validate_intent", "validate_intent"]
