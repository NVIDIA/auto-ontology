# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for :mod:`auto_ontology.retrieval.conversation.semantic_search`.

The embed endpoint and the vector store are stubbed. What is under test is
which rows the concept search is willing to answer with, and how many turns
the matched attributes are allowed to contribute between them.
"""

from __future__ import annotations

import contextlib
from typing import Any, Iterator
from unittest.mock import patch

from auto_ontology.retrieval.conversation.semantic_search import (
    _matched_ids,
    _matched_path_endpoints,
    _node_spans,
)


def _hit(row_id: str, type_: str, distance: float) -> dict:
    return {
        "text": f"{type_}: {row_id}",
        "_distance": distance,
        "metadata": {"id": row_id, "type": type_, "label": "Conversation"},
    }


def _path_hit(source_id: str, target_id: str, distance: float) -> dict:
    return {
        "text": f"{source_id} --> {target_id}",
        "_distance": distance,
        "metadata": {
            "source_id": source_id,
            "target_id": target_id,
            "label": "Conversation",
        },
    }


class _FakeVdb:
    def __init__(self, hits: list[dict]) -> None:
        self._hits = hits
        self.top_k: int | None = None

    def retrieval(self, vectors: list, top_k: int, **kwargs: Any) -> list[list[dict]]:
        self.top_k = top_k
        return [self._hits[:top_k]]


@contextlib.contextmanager
def _stub(hits: list[dict], paths: list[dict] | None = None) -> Iterator[_FakeVdb]:
    vdb = _FakeVdb(hits)
    module = "auto_ontology.retrieval.conversation.semantic_search"
    with (
        patch(f"{module}.get_embed_params"),
        patch(f"{module}.embed_texts", return_value=[[0.1, 0.2]]),
        patch(f"{module}.get_conversation_semantic_vdb", return_value=vdb),
        patch(f"{module}.get_one_hop_path_vdb", return_value=_FakeVdb(paths or [])),
    ):
        yield vdb


def test_the_subject_node_is_not_answered_with() -> None:
    """Its description summarises the transcript, so it overlaps any question
    about it while adding nothing the turns do not already say."""
    hits = [_hit("subject", "Subject", 0.11), _hit("a", "Metric", 0.22)]

    with _stub(hits):
        assert _matched_ids("q", database_name="mem", top_k=2) == ["a"]


def test_the_excluded_subject_does_not_consume_a_slot() -> None:
    """The caller asked for two nodes and must get two, not one.

    This is the whole reason the query over-fetches: the subject matches broad
    questions strongly and would otherwise take the nearest slot.
    """
    hits = [
        _hit("subject", "Subject", 0.10),
        _hit("a", "Metric", 0.20),
        _hit("b", "Technology", 0.30),
    ]

    with _stub(hits) as vdb:
        assert _matched_ids("q", database_name="mem", top_k=2) == ["a", "b"]
    assert vdb.top_k is not None and vdb.top_k > 2


def test_no_more_rows_are_returned_than_were_asked_for() -> None:
    """Over-fetching is headroom for dropped subjects, not a wider search."""
    hits = [_hit(name, "Metric", 0.1) for name in ("a", "b", "c", "d", "e")]

    with _stub(hits):
        assert len(_matched_ids("q", database_name="mem", top_k=2)) == 2


def test_a_matching_path_contributes_both_of_its_endpoints() -> None:
    """A path vector matches a question that names the relation and neither
    end, so the ends have to be pulled in behind it."""
    with _stub([], paths=[_path_hit("project", "flask", 0.12)]):
        found = _matched_path_endpoints("q", database_name="mem", path_k=5)

    assert found == ["project", "flask"]


def test_path_endpoints_come_source_first() -> None:
    """The source of a supersedes is the current value and of a precedes the
    earlier event -- the half worth keeping when the budget runs out."""
    with _stub([], paths=[_path_hit("current", "stale", 0.1)]):
        found = _matched_path_endpoints("q", database_name="mem", path_k=5)

    assert found.index("current") < found.index("stale")


def test_an_endpoint_on_two_matching_paths_is_listed_once() -> None:
    paths = [_path_hit("a", "b", 0.1), _path_hit("a", "c", 0.2)]

    with _stub([], paths=paths):
        found = _matched_path_endpoints("q", database_name="mem", path_k=5)

    assert found == ["a", "b", "c"]


def test_the_path_route_can_be_switched_off() -> None:
    with _stub([], paths=[_path_hit("a", "b", 0.1)]):
        assert _matched_path_endpoints("q", database_name="mem", path_k=0) == []


def _node(conversation_id: str, *turn_indexes: int) -> dict[str, Any]:
    return {"conversation_id": conversation_id, "turn_indexes": list(turn_indexes)}


def test_turns_are_taken_breadth_first_across_attributes() -> None:
    """One from each before a second from any.

    The case this route exists for -- an interval between two events -- needs
    one turn from each of two attributes. Depth-first would spend the budget
    inside the first and answer that the second was never mentioned.
    """
    attributes = [_node("1", 10, 11, 12), _node("1", 40)]

    spans = _node_spans(attributes, widen=0, limit=2)

    assert sorted(index for _, index, _ in spans) == [10, 40]


def test_each_cited_turn_gets_its_own_window() -> None:
    """Not one span from an attribute's first turn to its last.

    An attribute about a revised quota cites the original turn and the
    revision, which is the point of it; a single span would read everything
    in between.
    """
    spans = _node_spans([_node("1", 12, 48)], widen=0, limit=10)

    assert spans == [("1", 12, 12), ("1", 48, 48)]
