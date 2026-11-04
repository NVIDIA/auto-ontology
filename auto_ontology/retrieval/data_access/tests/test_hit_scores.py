# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A hybrid hit is gated and compared by its own rules, a cosine hit by the old ones.

The cosine cases pin that nothing changed for ``VDB_SEARCH_MODE=semantic``;
the hybrid cases pin that the gate and the cosine-tuned cutoffs read a hybrid
hit's vector distance rather than its RRF ordering -- a hit only BM25 found
must never look "close" -- and the branch-agreement tie rule.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from auto_ontology.retrieval.data_access.hit_scores import (
    ambiguous_pair,
    cosine_distance,
    passes_relevance_gate,
)
from auto_ontology.retrieval.entity_coverage.agents.coverage import _filter_by_distance
from auto_ontology.retrieval.interactive.entity_resolution import _ambiguity_check


def _cosine(score: float, **extra: Any) -> dict:
    return {"score": score, **extra}


def _hybrid(
    rrf: float,
    lexical: float | None = None,
    semantic: float | None = None,
    **extra: Any,
) -> dict:
    return {
        "score": 1 - rrf,
        "rrf": rrf,
        "lexical_score": lexical,
        "semantic_score": semantic,
        "semantic_distance": None if semantic is None else 2 * (1 - semantic),
        **extra,
    }


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("score, passes", [(0.64, True), (0.65, True), (0.66, False)])
def test_a_cosine_hit_is_gated_on_max_distance(score: float, passes: bool) -> None:
    assert passes_relevance_gate(_cosine(score), 0.65) is passes


@pytest.mark.parametrize(
    "semantic, passes",
    [(0.70, True), (0.675, True), (0.65, False)],  # distances 0.6, 0.65, 0.7
)
def test_a_hybrid_hit_is_gated_on_its_vector_distance(
    semantic: float, passes: bool
) -> None:
    hit = _hybrid(0.9, lexical=3.0, semantic=semantic)

    assert passes_relevance_gate(hit, max_distance=0.65) is passes


def test_a_hybrid_hits_rrf_does_not_get_it_past_the_gate() -> None:
    # First in both searches (rrf 1.0, score 0.0), but semantically far: the
    # catalog has no answer, and a fused rank cannot say so.
    hit = _hybrid(1.0, lexical=5.0, semantic=0.5)

    assert hit["score"] == 0.0
    assert not passes_relevance_gate(hit, max_distance=0.65)


def test_a_hit_only_bm25_found_never_passes_the_gate() -> None:
    assert not passes_relevance_gate(_hybrid(0.5, lexical=9.0), max_distance=10.0)


def test_coverage_filters_hybrid_hits_on_vector_distance() -> None:
    hits = [_hybrid(0.6, None, 0.9, id="near"), _hybrid(1.0, 5.0, 0.5, id="far")]

    assert [h["id"] for h in _filter_by_distance(hits, max_distance=0.65)] == ["near"]


# ---------------------------------------------------------------------------
# Cosine distance for the cosine-tuned cutoffs
# ---------------------------------------------------------------------------


def test_a_cosine_hit_is_its_own_distance() -> None:
    assert cosine_distance(_cosine(0.3)) == 0.3


def test_a_hybrid_hit_reports_its_vector_distance_not_its_rrf() -> None:
    hit = _hybrid(0.95, lexical=2.0, semantic=0.8)

    assert cosine_distance(hit) == pytest.approx(0.4)


def test_a_hit_only_bm25_found_is_never_close() -> None:
    hit = _hybrid(0.5, lexical=9.0, semantic=None)

    assert cosine_distance(hit) == math.inf
    assert cosine_distance(hit, missing=1.0) == 1.0


# ---------------------------------------------------------------------------
# Ties
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gap, tie", [(0.1, True), (0.3, False)])
def test_cosine_hits_tie_on_a_small_gap(gap: float, tie: bool) -> None:
    assert ambiguous_pair(_cosine(0.2), _cosine(0.2 + gap), 0.25) is tie


def test_hybrid_hits_both_searches_rank_alike_are_not_a_tie() -> None:
    best = _hybrid(0.99, lexical=5.0, semantic=0.9)
    second = _hybrid(0.97, lexical=4.0, semantic=0.8)

    assert not ambiguous_pair(best, second, max_gap=0.0)


def test_hybrid_hits_the_searches_order_differently_are_a_tie() -> None:
    best = _hybrid(0.99, lexical=5.0, semantic=0.8)
    second = _hybrid(0.98, lexical=4.0, semantic=0.9)

    assert ambiguous_pair(best, second, max_gap=1.0)


def test_a_hybrid_hit_only_one_search_found_loses_that_search() -> None:
    # Vector search has both, ranks best higher; BM25 found only best.
    best = _hybrid(0.99, lexical=5.0, semantic=0.9)
    second = _hybrid(0.5, lexical=None, semantic=0.8)
    assert not ambiguous_pair(best, second, max_gap=1.0)

    # BM25 found only the *second* one: the searches disagree.
    best = _hybrid(0.5, lexical=None, semantic=0.9)
    second = _hybrid(0.5, lexical=5.0, semantic=None)
    assert ambiguous_pair(best, second, max_gap=1.0)


def test_hybrid_hits_neither_search_can_separate_are_a_tie() -> None:
    hit = _hybrid(0.9, lexical=3.0, semantic=0.8)

    assert ambiguous_pair(hit, dict(hit), max_gap=0.0)


# ---------------------------------------------------------------------------
# The clarification flow's ambiguity check
# ---------------------------------------------------------------------------


def test_ambiguity_check_is_unchanged_for_cosine_hits() -> None:
    hits = [
        _cosine(0.30, query_entity="tie"),
        _cosine(0.32, query_entity="tie"),
        _cosine(0.10, query_entity="clear"),
        _cosine(0.60, query_entity="clear"),
        _cosine(0.20, query_entity="lone"),
        _cosine(0.90, query_entity="lone"),  # past CLARIFY_MAX_DISTANCE
    ]

    assert _ambiguity_check({"retrieved_column_attributes": hits}) == {"tie"}


def test_ambiguity_check_reads_branch_agreement_for_hybrid_hits() -> None:
    hits = [
        # Searches disagree on the top two: a tie.
        _hybrid(0.99, lexical=5.0, semantic=0.80, query_entity="tie"),
        _hybrid(0.98, lexical=4.0, semantic=0.90, query_entity="tie"),
        # Searches agree, however close the RRF scores: not a tie.
        _hybrid(0.99, lexical=5.0, semantic=0.90, query_entity="clear"),
        _hybrid(0.98, lexical=4.0, semantic=0.89, query_entity="clear"),
        # Second hit too far by vector distance: not a contender.
        _hybrid(0.99, lexical=5.0, semantic=0.80, query_entity="lone"),
        _hybrid(0.98, lexical=4.0, semantic=0.55, query_entity="lone"),
    ]

    assert _ambiguity_check({"retrieved_column_attributes": hits}) == {"tie"}
