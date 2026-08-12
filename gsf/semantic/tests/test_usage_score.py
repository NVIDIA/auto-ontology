# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ranking inputs for the SqlAttribute suggester — currently inert, on purpose.

**These tests pin a known bug rather than the behaviour anyone wants.**
`_usage_score` returns 0.0 for everything, so candidate expressions are never
ranked; ordering is whatever the dict happened to be. The cause and the one-line
fix are in that function's docstring and in
the note on ``_usage_score``.

Pinning it is deliberate. Turning ranking on changes which SqlAttributes the
semantic layer suggests, and these tests are supposed to leave that
component's output untouched — so the fix lands on its own, with its own tests,
not as a side effect of a port. Until then these tests exist to make the
inertness *explicit and load-bearing*: nothing caught this for the life of the
feature precisely because no test asserted anything about the score.

**When the fix lands**, every assertion here inverts. That is the signal it
worked, not a regression.
"""

from __future__ import annotations

from gsf.semantic.sql_attribute_suggester import _rank_expressions, _usage_score

REAL_PROPS = {
    "name": "query_abc",
    "sql_full_query": "SELECT customer_id FROM customer",
    "count_8_2026": 7,
    "total_counter": 7,
    "last_query_timestamp": "2026-08-11T00:00:00Z",
    "nodes_count": 12,
    "join_count": 0,
    "union_count": 0,
}
"""Exactly what ``Query.__init__`` writes. The key names are the point."""


def test_score_is_currently_always_zero() -> None:
    """Known bug, pinned so a change to it is visible.

    If this starts failing, ranking has been turned on — check that was
    intended and update this module rather than reverting the change.
    """
    assert _usage_score(REAL_PROPS) == 0.0
    assert _usage_score({"total_counter": 500}) == 0.0
    assert _usage_score({}) == 0.0


def test_the_data_needed_to_rank_is_present() -> None:
    """The inputs exist and are non-zero; only the scorer ignores them.

    This is what makes the bug a wiring fault rather than missing data, and
    what makes the fix a one-liner.
    """
    assert REAL_PROPS["total_counter"] > 0
    assert any(key.startswith("count_") for key in REAL_PROPS)


def test_every_expression_currently_ties() -> None:
    """The user-visible consequence: a busy query ranks no higher than a quiet one."""
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
    scores = {score for _, score, _ in ranked}
    assert scores == {0.0}, (
        "scores are no longer uniformly zero, so ranking is live — see "
        "the bug note in _usage_score and update this module"
    )


def test_ranking_still_carries_source_query_ids() -> None:
    """Unaffected by the scoring bug, and needed to link an attribute back."""
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
