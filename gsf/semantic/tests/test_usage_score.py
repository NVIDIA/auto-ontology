# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ranking inputs for the SqlAttribute suggester.

These exist because the previous scorer silently returned 0.0 for every query
for its entire life: it matched property names against
``^count_monthly_(\\d{4})_(\\d{2})$`` while the writer produced
``count_{month}_{year}``. Nothing ever wrote a name the regex accepted, so every
expression tied on 0.0 and the "ranking" was dict insertion order.

Nothing caught it because nothing asserted on a *real* property bag — the shape
the writer actually produces. That is what these tests do.
"""

from __future__ import annotations

from gsf.semantic.sql_attribute_suggester import _rank_expressions, _usage_score


def test_score_reads_total_counter() -> None:
    assert _usage_score({"total_counter": 42}) == 42.0


def test_score_of_real_property_bag_is_not_zero() -> None:
    """The shape ``Query.__init__`` actually writes.

    The old scorer returned 0.0 for exactly this input. Pinning the real key
    names is the point of the test, so a future rename cannot quietly break
    ranking again.
    """
    props = {
        "name": "query_abc",
        "sql_full_query": "SELECT customer_id FROM customer",
        "count_8_2026": 7,
        "total_counter": 7,
        "last_query_timestamp": "2026-08-11T00:00:00Z",
        "nodes_count": 12,
        "join_count": 0,
        "union_count": 0,
    }
    assert _usage_score(props) == 7.0


def test_missing_or_unusable_counter_scores_zero() -> None:
    assert _usage_score({}) == 0.0
    assert _usage_score({"total_counter": None}) == 0.0
    assert _usage_score({"total_counter": "not a number"}) == 0.0


def test_expressions_are_ranked_by_usage() -> None:
    """The behaviour the bug removed: a busier query outranks a quiet one."""
    sqls = [
        {
            "sql_id": "quiet",
            "sql_text": "SELECT * FROM film WHERE rating = 'G'",
            "props": {"total_counter": 1},
        },
        {
            "sql_id": "busy",
            "sql_text": "SELECT * FROM film WHERE release_year > 2020",
            "props": {"total_counter": 500},
        },
    ]
    ranked = _rank_expressions(sqls)

    assert ranked, "no expressions extracted"
    scores = [score for _, score, _ in ranked]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] > scores[-1], (
        "every expression scored the same — the ranking is inert, which is "
        "precisely the bug this replaced"
    )


def test_ranking_carries_source_query_ids() -> None:
    sqls = [
        {
            "sql_id": "sql-1",
            "sql_text": "SELECT * FROM film WHERE rating = 'G'",
            "props": {"total_counter": 3},
        }
    ]
    ranked = _rank_expressions(sqls)
    assert ranked
    assert all(ids == ["sql-1"] for _, _, ids in ranked)
