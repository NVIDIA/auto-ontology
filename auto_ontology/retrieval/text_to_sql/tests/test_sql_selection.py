# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Selection behaviour: which candidate ships, and why."""

from __future__ import annotations

import json

import pytest

from auto_ontology.retrieval.text_to_sql.agents import sql_selection
from auto_ontology.retrieval.text_to_sql.agents.sql_selection import (
    SQLSelectionAgent,
)
from auto_ontology.retrieval.text_to_sql.sql_skeleton import (
    skeleton,
    skeleton_similarity,
)


class _Response:
    """Stand-in for QueryResponse."""

    def __init__(self, rows=None, error=None):
        self.error = error
        self.result = [json.dumps(rows)] if rows is not None else ["[]"]


class _Candidate:
    def __init__(self, sql):
        self.sql_code = sql


def _state(sqls, examples=None):
    candidates = [_Candidate(s) for s in sqls]
    return {
        "connectors": [object()],
        "sql_examples": examples or [],
        "path_state": {
            "sql_candidates": candidates,
            "sql_generation_result": candidates[0],
            "relevant_tables": [],
        },
    }


@pytest.fixture
def results(monkeypatch):
    """Route each candidate's SQL to a scripted result."""
    table: dict[str, _Response] = {}

    def fake_run_sql(sql, _connector):
        return table.get(sql.strip(), _Response(error="unmapped"))

    monkeypatch.setattr(sql_selection, "_run_sql", fake_run_sql)
    monkeypatch.setattr(
        sql_selection, "resolve_connector_from_tables", lambda *_: object()
    )
    return table


def _run(state):
    out = SQLSelectionAgent().execute(state)
    return out["path_state"]


def test_single_candidate_is_left_alone(monkeypatch):
    """A pool of one has nothing to decide, and must not be executed."""
    calls: list[str] = []

    def tracking(sql, _connector):
        calls.append(sql)
        return _Response([{"a": 1}])

    monkeypatch.setattr(sql_selection, "_run_sql", tracking)
    state = _state(["SELECT 1"])
    path = _run(state)
    assert path["sql_generation_result"] is state["path_state"]["sql_candidates"][0]
    assert "sql_selection" not in path
    assert calls == []


def test_majority_result_wins_over_slot_zero(results):
    """Three candidates agreeing beat the incumbent's lone answer."""
    results["A"] = _Response([{"n": 1}])
    results["B"] = _Response([{"n": 2}])
    results["C"] = _Response([{"n": 2}])
    results["D"] = _Response([{"n": 2}])
    path = _run(_state(["A", "B", "C", "D"]))
    assert path["sql_selection"]["winner"] == 1
    assert path["sql_selection"]["votes"] == 3
    assert path["sql_generation_result"].sql_code == "B"


def test_slot_zero_wins_an_intra_group_tie(results):
    """Inside the winning group the lowest slot ships."""
    results["A"] = _Response([{"n": 7}])
    results["B"] = _Response([{"n": 7}])
    path = _run(_state(["A", "B"]))
    assert path["sql_selection"]["winner"] == 0
    assert path["sql_generation_result"].sql_code == "A"


def test_empty_results_do_not_form_a_winning_bloc(results):
    """Three empties must not outvote the one candidate that found rows.

    This is the rule the measurement said mattered most: over-constrained
    predicates fail identically, so their agreement is not evidence.
    """
    results["A"] = _Response([])
    results["B"] = _Response([])
    results["C"] = _Response([])
    results["D"] = _Response([{"n": 42}])
    path = _run(_state(["A", "B", "C", "D"]))
    assert path["sql_selection"]["winner"] == 3
    assert path["sql_generation_result"].sql_code == "D"


def test_failed_candidates_are_ignored(results):
    results["A"] = _Response(error="syntax error")
    results["B"] = _Response([{"n": 1}])
    path = _run(_state(["A", "B"]))
    assert path["sql_selection"]["winner"] == 1


def test_everything_failing_keeps_slot_zero(results):
    results["A"] = _Response(error="boom")
    results["B"] = _Response(error="boom")
    path = _run(_state(["A", "B"]))
    assert path["sql_selection"]["reason"] == "no_candidate_executed"
    assert path["sql_generation_result"].sql_code == "A"


def test_all_empty_falls_back_rather_than_giving_up(results):
    """With no non-empty candidate, an empty answer is still better than none."""
    results["A"] = _Response([])
    results["B"] = _Response([])
    path = _run(_state(["A", "B"]))
    assert path["sql_selection"]["reason"] == "result_vote"
    assert path["sql_generation_result"].sql_code == "A"


def test_row_order_and_duplicates_do_not_split_a_group(results):
    """Same rows in a different order are the same answer."""
    results["A"] = _Response([{"n": 1}, {"n": 2}])
    results["B"] = _Response([{"n": 2}, {"n": 1}, {"n": 1}])
    results["C"] = _Response([{"n": 9}])
    path = _run(_state(["A", "B", "C"]))
    assert path["sql_selection"]["votes"] == 2
    assert path["sql_selection"]["winner"] == 0


def test_pattern_fit_breaks_a_vote_tie(results, monkeypatch):
    """With votes level, the candidate shaped like the examples wins."""
    monkeypatch.setenv("BIRD_SQL_PATTERN_ALPHA", "1.0")
    results["SELECT COUNT(*) FROM t WHERE x = 'a'"] = _Response([{"n": 1}])
    results["SELECT y FROM t"] = _Response([{"n": 2}])
    examples = [{"sql": "SELECT COUNT(*) FROM other WHERE col = 'v'"}]
    # Slot 1 matches the example's shape; slot 0 would win a bare tie.
    state = _state(
        ["SELECT y FROM t", "SELECT COUNT(*) FROM t WHERE x = 'a'"], examples
    )
    path = _run(state)
    assert path["sql_selection"]["winner"] == 1


def test_pattern_alpha_zero_leaves_a_pure_vote(results, monkeypatch):
    monkeypatch.setenv("BIRD_SQL_PATTERN_ALPHA", "0")
    results["SELECT COUNT(*) FROM t WHERE x = 'a'"] = _Response([{"n": 1}])
    results["SELECT y FROM t"] = _Response([{"n": 2}])
    examples = [{"sql": "SELECT COUNT(*) FROM other WHERE col = 'v'"}]
    state = _state(
        ["SELECT y FROM t", "SELECT COUNT(*) FROM t WHERE x = 'a'"], examples
    )
    path = _run(state)
    assert path["sql_selection"]["winner"] == 0


def test_votes_still_outrank_pattern_fit(results, monkeypatch):
    """Pattern is a tie-break, not an override: two votes beat one good shape."""
    monkeypatch.setenv("BIRD_SQL_PATTERN_ALPHA", "1.0")
    results["SELECT COUNT(*) FROM t WHERE x = 'a'"] = _Response([{"n": 1}])
    results["SELECT y FROM t"] = _Response([{"n": 2}])
    results["SELECT y FROM t2"] = _Response([{"n": 2}])
    examples = [{"sql": "SELECT COUNT(*) FROM other WHERE col = 'v'"}]
    state = _state(
        [
            "SELECT y FROM t",
            "SELECT y FROM t2",
            "SELECT COUNT(*) FROM t WHERE x = 'a'",
        ],
        examples,
    )
    path = _run(state)
    assert path["sql_selection"]["winner"] == 0
    assert path["sql_selection"]["votes"] == 2


def test_candidates_with_blank_sql_are_skipped(results):
    results["B"] = _Response([{"n": 1}])
    path = _run(_state(["   ", "B"]))
    assert path["sql_selection"]["winner"] == 1


class TestSkeleton:
    def test_identifiers_and_literals_are_abstracted(self):
        got = skeleton(
            "SELECT MAX(`Free Meal Count`) FROM frpm WHERE `County` = 'Alameda'"
        )
        assert got == "select max ( C ) from T where C = L"

    def test_aliases_do_not_change_the_shape(self):
        plain = skeleton("SELECT a.x FROM t1 AS a JOIN t2 AS b ON a.k = b.k")
        bare = skeleton("SELECT x FROM t1 JOIN t2 ON k = k")
        assert plain == bare

    def test_similarity_is_scale_free_across_schemas(self):
        left = skeleton("SELECT COUNT(*) FROM a WHERE x = 'v'")
        right = skeleton("SELECT COUNT(*) FROM other WHERE col = 'w'")
        assert skeleton_similarity(left, right) == 1.0

    def test_different_shapes_score_below_identical(self):
        count = skeleton("SELECT COUNT(*) FROM a WHERE x = 'v'")
        plain = skeleton("SELECT y FROM a")
        assert skeleton_similarity(count, plain) < 1.0

    def test_empty_input_is_not_similar_to_anything(self):
        assert skeleton_similarity("", "select C from T") == 0.0
