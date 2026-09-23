# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``announce_node_start``: the graph-side half of the live step label.

``app.stream()`` only yields a node's update once it has returned, so without
this announcement the label a client shows names the previous node for the
whole time the current one runs. These tests pin the announcement to the
*graph wrapper* every node goes through, and check it survives being called
outside a stream.
"""

import time
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from auto_ontology.retrieval.text_to_sql.text_to_sql_graph import (
    NODE_START_EVENT,
    announce_node_start,
    wrap_node_with_logging,
)


class _State(TypedDict):
    path_state: dict
    decision: str


def _compile(name: str, fn: Any) -> Any:
    graph = StateGraph(_State)
    graph.add_node(name, wrap_node_with_logging(name, fn))
    graph.set_entry_point(name)
    graph.add_edge(name, END)
    return graph.compile()


def test_start_is_streamed_before_the_node_returns() -> None:
    """The announcement must arrive while the node is still working — that
    lead time is the entire point."""

    def slow(_state: _State) -> dict[str, Any]:
        time.sleep(0.3)
        return {"path_state": {}, "decision": "done"}

    app = _compile("reconstruct_sql", slow)

    started = time.perf_counter()
    timings: list[tuple[str, float]] = []
    for mode, _chunk in app.stream(
        {"path_state": {}, "decision": ""}, stream_mode=["updates", "custom"]
    ):
        timings.append((mode, time.perf_counter() - started))

    modes = [mode for mode, _ in timings]
    assert modes == ["custom", "updates"]

    custom_at = timings[0][1]
    updates_at = timings[1][1]
    assert custom_at < updates_at
    # The node sleeps 0.3s; the announcement must not wait for it.
    assert updates_at - custom_at > 0.2


def test_every_wrapped_node_announces_itself() -> None:
    """The wrapper is applied by ``_make_node`` to every node in the graph, so
    announcing here is what makes the behaviour universal rather than
    per-agent."""

    app = _compile(
        "validate_sql_query",
        lambda _state: {"path_state": {}, "decision": "valid_sql"},
    )

    custom = [
        chunk
        for mode, chunk in app.stream(
            {"path_state": {}, "decision": ""}, stream_mode=["updates", "custom"]
        )
        if mode == "custom"
    ]

    assert custom == [{"type": NODE_START_EVENT, "node": "validate_sql_query"}]


def test_announcing_outside_a_stream_is_not_fatal() -> None:
    """``get_agent_response`` and the tests call the graph without a stream
    writer; a missing progress event must not take the run down."""

    announce_node_start("validate_sql_query")


def test_node_still_runs_when_announcing_fails(monkeypatch: Any) -> None:
    """Progress reporting is best-effort — a broken writer must not stop the
    agent from doing its work."""

    import langgraph.config

    def explode() -> Any:
        raise RuntimeError("no writer here")

    monkeypatch.setattr(langgraph.config, "get_stream_writer", explode)

    calls: list[str] = []

    def node(_state: _State) -> dict[str, Any]:
        calls.append("ran")
        return {"path_state": {}, "decision": "done"}

    wrapped = wrap_node_with_logging("validate_sql_query", node)
    result = wrapped({"path_state": {}, "decision": ""})

    assert calls == ["ran"]
    assert result["decision"] == "done"
