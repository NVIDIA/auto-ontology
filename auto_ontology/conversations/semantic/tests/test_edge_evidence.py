# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Whether an edge's quote is really in the turn it cites.

The only check in this pipeline that asks whether an *edge* is true rather than
well-formed. Everything else -- consolidation, disambiguation, the subject pass,
the write guards -- compares nodes.
"""

from __future__ import annotations

from auto_ontology.conversations.semantic.extract import (
    EdgeDraft,
    report_unquoted_edges,
)
from auto_ontology.conversations.semantic.windows import Window, build_windows


def _window(*bodies: str) -> Window:
    return build_windows(
        [
            {"turn_index": i, "question": body, "answer": "", "time_anchor": None}
            for i, body in enumerate(bodies)
        ],
        max_chars=10**6,
    )[0]


def _edge(evidence: str, *, verified: set[int] | None = None) -> EdgeDraft:
    return EdgeDraft(
        source="weather app",
        target="localstorage",
        role="built_with",
        turn_indexes={0},
        verified_turn_indexes=verified or set(),
        evidence=evidence,
    )


class TestFindingTheQuote:
    """What the free check can actually establish, via ``Window.find_quote``."""

    def test_a_verbatim_quote_resolves_to_its_turn(self) -> None:
        window = _window("I switched the cache to localStorage last week", "unrelated")
        assert window.find_quote("switched the cache to localStorage") == 0

    def test_whitespace_differences_do_not_break_the_match(self) -> None:
        """The model reflows what it copies; the turn's own line breaks are not
        part of the claim."""
        window = _window("I switched the cache\nto   localStorage")
        assert window.find_quote("switched the cache to localStorage") == 0

    def test_a_quote_in_no_turn_resolves_to_nothing(self) -> None:
        window = _window("I switched the cache to localStorage")
        assert window.find_quote("switched the cache to IndexedDB") is None

    def test_a_quote_in_two_turns_resolves_to_nothing(self) -> None:
        """Ambiguous attribution is not attribution: a quote that could have come
        from either turn proves neither."""
        window = _window("we used localStorage here", "we used localStorage here too")
        assert window.find_quote("we used localStorage here") is None


class TestBuckets:
    def test_an_edge_whose_quote_matched_is_not_reported(self) -> None:
        assert report_unquoted_edges([_edge("said so", verified={0})]) == []

    def test_an_edge_with_no_quote_is_not_reported(self) -> None:
        """Absence of evidence is the normal case, not a claim the transcript was
        misread -- the extractor omits a quote freely."""
        assert report_unquoted_edges([_edge("")]) == []

    def test_an_edge_quoting_something_absent_is_reported(self) -> None:
        """The one case where the model asserted something checkable and the
        check failed."""
        fabricated = report_unquoted_edges([_edge("a quote in no turn")])
        assert [edge.evidence for edge in fabricated] == ["a quote in no turn"]

    def test_a_whitespace_only_quote_counts_as_no_quote(self) -> None:
        assert report_unquoted_edges([_edge("   ")]) == []

    def test_the_buckets_are_independent(self) -> None:
        edges = [
            _edge("matched", verified={0}),
            _edge(""),
            _edge("absent"),
            _edge("also absent"),
        ]
        assert len(report_unquoted_edges(edges)) == 2

    def test_nothing_is_dropped(self) -> None:
        """Reported, not filtered. An unmeasured filter that deletes edges is how
        a verifier quietly removes a correct graph."""
        edges = [_edge("absent")]
        report_unquoted_edges(edges)
        assert len(edges) == 1
