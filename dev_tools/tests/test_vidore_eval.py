# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the ViDoRe retrieval metrics.

The benchmark corpus is untracked, so these exercise the pure scoring logic
against hand-built judgements instead. That is the part worth pinning: a metric
bug would shift both arms together and look like a real result.
"""

from __future__ import annotations

from pathlib import Path

from dev_tools.vidore_eval import load_qrels, ndcg_at_k, recall_at_k

#: Two relevant pages, one highly relevant.
GAINS = {("doc_a", 3): 2, ("doc_a", 7): 1}


def test_ndcg_is_one_for_the_ideal_ranking() -> None:
    ranked = [("doc_a", 3), ("doc_a", 7), ("doc_b", 1)]
    assert ndcg_at_k(ranked, GAINS, 5) == 1.0


def test_ndcg_penalises_putting_the_lesser_page_first() -> None:
    ideal = ndcg_at_k([("doc_a", 3), ("doc_a", 7)], GAINS, 5)
    swapped = ndcg_at_k([("doc_a", 7), ("doc_a", 3)], GAINS, 5)
    assert swapped < ideal


def test_ndcg_is_zero_when_nothing_relevant_is_retrieved() -> None:
    assert ndcg_at_k([("doc_b", 1), ("doc_b", 2)], GAINS, 5) == 0.0


def test_ndcg_ignores_hits_past_the_cutoff() -> None:
    ranked = [("doc_b", 1), ("doc_b", 2), ("doc_a", 3)]
    assert ndcg_at_k(ranked, GAINS, 2) == 0.0
    assert ndcg_at_k(ranked, GAINS, 3) > 0.0


def test_ndcg_can_reach_one_with_more_relevant_pages_than_the_cutoff() -> None:
    """The ideal ranking is truncated at k, so k perfect hits score 1.0.

    Without truncation every query judged more broadly than the cutoff would be
    capped below 1.0, which would read as a retrieval failure.
    """
    gains = {("doc_a", page): 1 for page in range(1, 11)}
    ranked = [("doc_a", page) for page in range(1, 6)]
    assert ndcg_at_k(ranked, gains, 5) == 1.0


def test_ndcg_of_empty_judgements_is_zero_not_an_error() -> None:
    assert ndcg_at_k([("doc_a", 3)], {}, 5) == 0.0


def test_recall_counts_judged_pages_found() -> None:
    assert recall_at_k([("doc_a", 3), ("doc_b", 9)], GAINS, 5) == 0.5
    assert recall_at_k([("doc_a", 3), ("doc_a", 7)], GAINS, 5) == 1.0


def test_recall_is_bounded_by_the_cutoff() -> None:
    """A query judged over more pages than k cannot reach 1.0."""
    gains = {("doc_a", page): 1 for page in range(1, 21)}
    ranked = [("doc_a", page) for page in range(1, 11)]
    assert recall_at_k(ranked, gains, 10) == 0.5


def test_load_qrels_groups_by_query_and_keeps_grades(tmp_path: Path) -> None:
    qrels_dir = tmp_path / "qrels"
    qrels_dir.mkdir()
    (qrels_dir / "qrels_en.csv").write_text(
        """query_id,doc_id,page_number,score
q1,doc_a,3,2
q1,doc_a,7,1
q2,doc_b,1,1
"""
    )

    qrels = load_qrels(tmp_path)

    assert qrels == {"q1": GAINS, "q2": {("doc_b", 1): 1}}
