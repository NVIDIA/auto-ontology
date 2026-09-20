# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the undiagnosed-empty-result retry in EmptyResultValueRepairAgent.

The targeted literal repair only covers a value that is a near-miss of a real
one. Everything else that leaves a result empty — a filter on a column that
doesn't hold the value, an over-restrictive join, a format mismatch — reaches
the end of the graph undiagnosed, and is regenerated once instead of being
returned as an empty answer.
"""

from gsf.retrieval.text_to_sql.agents.empty_result_value_repair import (
    EmptyResultValueRepairAgent,
)


def _state(db_result, **path_state):
    """Minimal state: no connectors, so every probe fails and finds no mismatch."""
    return {
        "connectors": [],
        "path_state": {
            "sql_code": "SELECT * FROM t WHERE a = 'x' AND b = 'y'",
            "sql_response_from_db": db_result,
            "relevant_tables": [],
            **path_state,
        },
    }


def test_non_empty_result_passes_through_untouched():
    """The common path must stay free: no probe, no retry, no state churn."""
    out = EmptyResultValueRepairAgent().execute(_state([{"a": 1}]))

    assert out["decision"] == "valid_sql"
    assert "empty_result_retry_attempted" not in out["path_state"]


def test_undiagnosed_empty_result_is_regenerated_once():
    """Nothing to repair, but an empty result answers nothing — go back once."""
    out = EmptyResultValueRepairAgent().execute(_state([]))

    assert out["decision"] == "invalid_sql"
    assert out["path_state"]["empty_result_retry_attempted"] is True
    assert "empty result set" in out["path_state"]["error"]


def test_undiagnosed_retry_lets_reconstruction_classify_the_error():
    """Unlike the literal repair, this has no probe evidence behind it, so
    reconstruction must stay free to decide the data lives elsewhere and search
    for other tables (see sql_reconstruction's missing_data path).
    """
    out = EmptyResultValueRepairAgent().execute(_state([]))

    assert "error_known_fixable" not in out["path_state"]


def test_retry_happens_at_most_once():
    """A second empty result is accepted rather than looped on."""
    out = EmptyResultValueRepairAgent().execute(
        _state([], empty_result_retry_attempted=True)
    )

    assert out["decision"] == "valid_sql"


def test_retry_still_runs_after_a_spent_literal_repair():
    """The proactive check may already have used up the literal repair earlier
    in the run. That says nothing about this empty result, so the undiagnosed
    retry is still owed.
    """
    out = EmptyResultValueRepairAgent().execute(
        _state([], value_repair_attempted=True)
    )

    assert out["decision"] == "invalid_sql"
    assert out["path_state"]["empty_result_retry_attempted"] is True
