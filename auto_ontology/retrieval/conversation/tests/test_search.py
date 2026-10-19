# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for :mod:`auto_ontology.retrieval.conversation.search`.

The embed endpoint and the store are stubbed. What is under test is the three
things this module adds over a bare similarity search: transcript ordering,
widening, and reading the text back from the rows rather than the vectors.
"""

from __future__ import annotations

import contextlib
from typing import Any, Iterator
from unittest.mock import patch

import pytest

from auto_ontology.retrieval.conversation.search import _windows, search_turns


def _hit(turn_index: int, distance: float, conversation_id: str = "1") -> dict:
    return {
        "text": f"embedding text for turn {turn_index}",
        "_distance": distance,
        "metadata": {
            "conversation_id": conversation_id,
            "turn_index": turn_index,
            "label": "Conversation",
        },
    }


def _row(turn_index: int, conversation_id: str = "1") -> dict[str, Any]:
    return {
        "id": f"t{turn_index}",
        "conversation_id": conversation_id,
        "turn_index": turn_index,
        "session_index": 0,
        "question": f"q{turn_index}",
        "answer": f"a{turn_index}",
        "time_anchor": "March-15-2024",
    }


class _FakeVdb:
    def __init__(self, hits: list[dict]) -> None:
        self._hits = hits
        self.where: dict | None = None

    def retrieval(self, vectors: list, top_k: int, **kwargs: Any) -> list[list[dict]]:
        self.where = kwargs.get("where")
        return [self._hits]


# Distinguishes "the test did not say" from "the test said None", which is
# the case under test in test_a_query_that_does_not_embed_returns_nothing.
_DEFAULT_VECTOR = object()


@contextlib.contextmanager
def _stub(
    hits: list[dict], rows: list[dict], vector: Any = _DEFAULT_VECTOR
) -> Iterator[tuple[_FakeVdb, list]]:
    vdb = _FakeVdb(hits)
    calls: list = []

    def _fetch(database_name: str, *, spans: Any = None) -> list[dict]:
        calls.append(spans)
        return rows

    with (
        patch("auto_ontology.retrieval.conversation.search.get_embed_params"),
        patch(
            "auto_ontology.retrieval.conversation.search.embed_texts",
            return_value=[[0.1, 0.2] if vector is _DEFAULT_VECTOR else vector],
        ),
        patch(
            "auto_ontology.retrieval.conversation.search.get_conversation_vdb",
            return_value=vdb,
        ),
        patch(
            "auto_ontology.retrieval.conversation.search.fetch_turns",
            side_effect=_fetch,
        ),
    ):
        yield vdb, calls


def test_hits_come_back_in_transcript_order_not_distance_order() -> None:
    """An answer must not arrive above the question that prompted it."""
    hits = [_hit(12, 0.21), _hit(3, 0.44), _hit(7, 0.33)]
    rows = [_row(3), _row(7), _row(12)]

    with _stub(hits, rows):
        found = search_turns("q", database_name="mem")

    assert [t.turn_index for t in found] == [3, 7, 12]


def test_text_comes_from_the_rows_not_the_vectors() -> None:
    """The vector holds embedding text; the row holds what was said."""
    with _stub([_hit(3, 0.2)], [_row(3)]):
        found = search_turns("q", database_name="mem")

    assert found[0].question == "q3"
    assert found[0].answer == "a3"
    assert "embedding text" not in found[0].question


def test_widening_asks_for_the_neighbouring_turns() -> None:
    with _stub([_hit(5, 0.2)], [_row(4), _row(5), _row(6)]) as (_vdb, calls):
        search_turns("q", database_name="mem", widen=1)

    assert calls == [[("1", 4, 6)]]


def test_a_widened_neighbour_is_not_reported_as_a_match() -> None:
    with _stub([_hit(5, 0.2)], [_row(4), _row(5), _row(6)]):
        found = search_turns("q", database_name="mem", widen=1)

    assert [(t.turn_index, t.matched) for t in found] == [
        (4, False),
        (5, True),
        (6, False),
    ]
    assert found[1].distance == pytest.approx(0.2)


def test_the_search_is_scoped_to_one_memory() -> None:
    """Without the filter this would search every transcript in the store."""
    with _stub([_hit(1, 0.2)], [_row(1)]) as (vdb, _calls):
        search_turns("q", database_name="mem")

    assert vdb.where == {"database_name": "mem", "label": "Conversation"}


def test_a_blank_query_searches_nothing() -> None:
    with _stub([_hit(1, 0.2)], [_row(1)]) as (_vdb, calls):
        assert search_turns("   ", database_name="mem") == []
    assert calls == []


def test_a_query_that_does_not_embed_returns_nothing() -> None:
    with _stub([_hit(1, 0.2)], [_row(1)], vector=None):
        assert search_turns("q", database_name="mem") == []


def test_an_empty_memory_returns_nothing() -> None:
    with _stub([], []):
        assert search_turns("q", database_name="mem") == []


# ── window merging ───────────────────────────────────────────────────────────


def test_distant_hits_stay_separate_windows() -> None:
    """One window per conversation would read every turn in between."""
    windows = _windows({("1", 5): 0.1, ("1", 80): 0.2}, widen=1)

    assert sorted(windows) == [("1", 4, 6), ("1", 79, 81)]


def test_overlapping_windows_merge() -> None:
    windows = _windows({("1", 5): 0.1, ("1", 6): 0.2}, widen=1)

    assert windows == [("1", 4, 7)]


def test_windows_that_exactly_meet_merge() -> None:
    """(..9] and [10..) are one unbroken stretch, not two predicates."""
    windows = _windows({("1", 5): 0.1, ("1", 8): 0.2}, widen=1)

    assert windows == [("1", 4, 9)]


def test_a_window_never_starts_below_turn_zero() -> None:
    windows = _windows({("1", 0): 0.1}, widen=3)

    assert windows == [("1", 0, 3)]


def test_conversations_get_their_own_windows() -> None:
    windows = _windows({("1", 5): 0.1, ("2", 5): 0.2}, widen=0)

    assert sorted(windows) == [("1", 5, 5), ("2", 5, 5)]


def test_no_widening_asks_only_for_the_hits() -> None:
    windows = _windows({("1", 5): 0.1, ("1", 80): 0.2}, widen=0)

    assert sorted(windows) == [("1", 5, 5), ("1", 80, 80)]
