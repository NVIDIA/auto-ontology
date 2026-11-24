# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A window that fails extraction stops the build before anything is kept.

The draft replaces the stored graph, so a skipped window would swap a complete
graph for one missing that window's turns.
"""

from __future__ import annotations

import pytest

from auto_ontology.conversations.semantic import extract, pipeline
from auto_ontology.conversations.semantic import store as semantic_store
from auto_ontology.conversations.semantic.extract import (
    EdgeItem,
    GraphExtractionResult,
    NodeItem,
    extract_graph,
)
from auto_ontology.conversations.semantic.windows import build_windows


def _turns(count: int) -> list[dict]:
    return [
        {"turn_index": i, "question": "q" * 400, "answer": "", "time_anchor": None}
        for i in range(count)
    ]


@pytest.fixture
def second_window_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = iter([GraphExtractionResult(), None])
    monkeypatch.setattr(extract, "_extract_window", lambda *_: next(calls))


def test_a_failed_window_raises_and_names_its_turns(
    second_window_fails: None,
) -> None:
    windows = build_windows(_turns(2), max_chars=500)
    assert len(windows) == 2

    with pytest.raises(RuntimeError, match=r"window 2 of 2 \(turns 1-1\)"):
        extract_graph(windows)


def test_a_failed_window_is_not_written(
    second_window_fails: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pipeline, "fetch_turns", lambda *_, **__: _turns(2))
    written: list[str] = []
    monkeypatch.setattr(
        semantic_store, "replace_graph", lambda **kw: written.append(kw) or {}
    )

    with pytest.raises(RuntimeError):
        pipeline.populate_conversation_graph(
            database_name="beam", conversation_id="2", window_chars=500
        )

    assert written == []


def test_an_edge_survives_when_a_later_same_name_node_is_grounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    turns = [
        {
            "turn_index": 0,
            "question": "the hub cap is now 4800 parcels",
            "answer": "",
            "time_anchor": None,
        }
    ]
    result = GraphExtractionResult(
        nodes=[
            NodeItem(
                name="Hub cap", type="Metric", kinds=["fact"], key="cap", value="3000"
            ),
            NodeItem(
                name="Hub cap",
                type="Metric",
                kinds=["fact"],
                key="cap",
                value="4800",
                source_turns=[0],
            ),
            NodeItem(name="Hub", type="Place", kinds=["object"], source_turns=[0]),
        ],
        edges=[EdgeItem(source="Hub cap", target="Hub", role="about")],
    )
    monkeypatch.setattr(extract, "_extract_window", lambda *_: result)

    graph = extract_graph(build_windows(turns, max_chars=500))

    assert [d.value for d in graph.nodes if d.name == "Hub cap"] == ["4800"]
    assert [(e.source, e.target) for e in graph.edges] == [("hub cap", "hub")]
