# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the elaboration checks.

Every positive case here is a query the agent really submitted and lost a
point for, and every negative case is one it got right — including the two an
earlier, looser version of these checks wrongly flagged. The checks bounce a
submission, so a false positive costs a turn and invites the model to break a
correct query; that makes the negative cases the more important half of this
file.
"""

from __future__ import annotations

import pytest

from auto_ontology.retrieval.text_to_sql.sql_agent.elaboration import (
    find_gratuitous_aggregation,
    find_gratuitous_distinct_count,
    find_inert_joins,
    find_strict_boundary,
    find_undeduplicated_rows,
    rows_differ,
)

DIALECT = "sqlite"


# ----------------------------------------------------------------------
# An aggregate the question never asked for
# ----------------------------------------------------------------------


def test_a_sum_the_question_never_asked_for_is_offered_without_it():
    # Two rows match, and the agent added the SUM itself after reasoning
    # that a question phrased in the singular must have a single-row answer.
    sql = (
        'SELECT SUM(f."Enrollment (Ages 5-17)") AS total FROM schools AS s '
        'JOIN frpm AS f ON s."CDSCode" = f."CDSCode" '
        "WHERE s.\"EdOpsCode\" = 'SSS' AND s.\"City\" = 'Fremont'"
    )
    question = (
        "How many students from the ages of 5 to 17 are enrolled at the State "
        "Special School school in Fremont for the 2014-2015 academic year?"
    )
    variant = find_gratuitous_aggregation(sql, question, DIALECT)
    assert variant is not None
    assert "SUM" not in variant.sql.upper()
    assert "Enrollment (Ages 5-17)" in variant.sql


@pytest.mark.parametrize(
    "question",
    [
        "What is the total enrollment across all schools in Fremont?",
        "What is the combined enrollment of the Fremont schools?",
        "What is the average enrollment altogether?",
    ],
)
def test_a_question_that_asks_for_a_figure_leaves_the_aggregate_alone(question):
    sql = 'SELECT SUM(f."Enrollment (Ages 5-17)") FROM frpm AS f'
    assert find_gratuitous_aggregation(sql, question, DIALECT) is None


def test_a_grouped_aggregate_is_reporting_per_group_and_is_left_alone():
    sql = (
        "SELECT s.City, SUM(f.Enrollment) FROM schools AS s "
        "JOIN frpm AS f ON s.CDSCode = f.CDSCode GROUP BY s.City"
    )
    assert find_gratuitous_aggregation(sql, "Enrollment by city?", DIALECT) is None


def test_counting_and_picking_extremes_are_not_treated_as_collapsing():
    # "How many" is what COUNT is for and "the highest" is what MAX is for;
    # flagging either would fire on a large share of every slice for no gain.
    for sql in (
        "SELECT COUNT(*) FROM schools",
        "SELECT MAX(p.potential) FROM player AS p",
    ):
        assert find_gratuitous_aggregation(sql, "How many? highest?", DIALECT) is None


# ----------------------------------------------------------------------
# Deduplicating a count nothing asked to deduplicate
# ----------------------------------------------------------------------


def test_a_distinct_count_with_nothing_to_deduplicate_is_offered_plain():
    # One table, no join, so no row multiplication to undo: the agent saw
    # 1569 rows and 189 distinct players and counted the players.
    sql = (
        "SELECT COUNT(DISTINCT player_api_id) FROM Player_Attributes "
        "WHERE preferred_foot = 'left' AND attacking_work_rate = 'low'"
    )
    question = (
        "Among the players whose preferred foot was the left foot when "
        "attacking, how many of them would remain in his position?"
    )
    variant = find_gratuitous_distinct_count(sql, question, DIALECT)
    assert variant is not None
    assert "DISTINCT" not in variant.sql.upper()
    assert "player_api_id" in variant.sql


@pytest.mark.parametrize(
    "question",
    [
        "How many distinct players prefer the left foot?",
        "How many different majors do the members have?",
        "How many unique molecules are there?",
    ],
)
def test_a_question_that_asks_for_values_keeps_its_distinct(question):
    sql = "SELECT COUNT(DISTINCT player_api_id) FROM Player_Attributes"
    assert find_gratuitous_distinct_count(sql, question, DIALECT) is None


def test_a_distinct_count_over_a_join_is_left_alone():
    # Here the join really does multiply rows, which is the case the
    # DISTINCT exists to correct.
    sql = (
        "SELECT COUNT(DISTINCT c.client_id) FROM client AS c "
        "JOIN disp AS d ON c.client_id = d.client_id"
    )
    assert find_gratuitous_distinct_count(sql, "How many clients?", DIALECT) is None


# ----------------------------------------------------------------------
# Identical rows nothing asked to repeat
# ----------------------------------------------------------------------


def _result(rows: list[dict]) -> dict:
    return {"ok": True, "row_count": len(rows), "rows": rows}


def test_identical_multi_column_rows_are_reported():
    # The join to the expense table has several rows per member, and the
    # agent dropped the cost column without deduplicating.
    sql = (
        "SELECT m.first_name, m.last_name, mj.major_name FROM expense AS e "
        "JOIN member AS m ON e.link_to_member = m.member_id "
        "JOIN major AS mj ON m.link_to_major = mj.major_id WHERE e.cost > 100"
    )
    rows = [
        {"first_name": "Sacha", "last_name": "Harrison", "major_name": "Liberal Arts"},
        {"first_name": "Elijah", "last_name": "Allen", "major_name": "Elementary"},
        {"first_name": "Elijah", "last_name": "Allen", "major_name": "Elementary"},
        {"first_name": "Sacha", "last_name": "Harrison", "major_name": "Liberal Arts"},
    ]
    finding = find_undeduplicated_rows(sql, DIALECT, _result(rows))
    assert finding is not None
    assert "4 rows" in finding and "only 2" in finding


def test_one_repeated_value_may_be_real_data_and_is_left_alone():
    # Two drivers can share a surname, a user can earn the same badge
    # twice, a superhero can have the same value for two attributes. All
    # three were correct answers that an earlier version of this flagged.
    sql = "SELECT surname FROM drivers WHERE nationality = 'Italian'"
    rows = [{"surname": "Rossi"}, {"surname": "Rossi"}, {"surname": "Bandini"}]
    assert find_undeduplicated_rows(sql, DIALECT, _result(rows)) is None


def test_a_query_that_already_deduplicates_is_left_alone():
    sql = "SELECT DISTINCT m.first_name, m.last_name FROM member AS m"
    rows = [{"a": "x", "b": "y"}, {"a": "x", "b": "y"}]
    assert find_undeduplicated_rows(sql, DIALECT, _result(rows)) is None


def test_an_aggregate_decides_its_own_grain():
    sql = "SELECT city, COUNT(*) FROM schools GROUP BY city"
    rows = [{"city": "Fremont", "n": 2}, {"city": "Fremont", "n": 2}]
    assert find_undeduplicated_rows(sql, DIALECT, _result(rows)) is None


def test_a_result_with_no_repeats_says_nothing():
    sql = "SELECT a.x, a.y FROM a"
    rows = [{"x": 1, "y": 2}, {"x": 1, "y": 3}]
    assert find_undeduplicated_rows(sql, DIALECT, _result(rows)) is None


def test_a_failed_execution_is_not_this_check_s_business():
    sql = "SELECT a.x, a.y FROM a"
    assert (
        find_undeduplicated_rows(
            sql, DIALECT, {"ok": False, "error": "boom", "rows": []}
        )
        is None
    )


# ----------------------------------------------------------------------
# A strict boundary that may exclude the answer
# ----------------------------------------------------------------------


def test_a_strict_comparison_between_two_columns_is_offered_inclusively():
    # "after the account opened" became t.date > a.date, which excluded
    # the transaction on the opening date — and that was the answer.
    sql = (
        "SELECT t.amount FROM trans AS t JOIN account AS a "
        "ON t.account_id = a.account_id WHERE t.date > a.date "
        "ORDER BY t.date ASC LIMIT 1"
    )
    variant = find_strict_boundary(sql, DIALECT)
    assert variant is not None
    assert ">=" in variant.sql


def test_a_threshold_the_question_stated_is_left_strict():
    # "greater than 3000USD" is genuinely exclusive, and a literal threshold
    # is never the model guessing at a boundary.
    sql = "SELECT a.account_id FROM account AS a JOIN trans AS t ON a.account_id = t.account_id WHERE t.amount > 3000"
    assert find_strict_boundary(sql, DIALECT) is None


# ----------------------------------------------------------------------
# Joins that carry no weight
# ----------------------------------------------------------------------


def test_a_join_to_a_table_nothing_reads_is_reported():
    # `disp` came from reading "own an amount of money" as a disposition
    # type, and the answer never reads a single column of it.
    sql = (
        "SELECT DISTINCT a.account_id FROM account AS a "
        "JOIN disp AS d ON a.account_id = d.account_id AND d.type = 'OWNER' "
        'JOIN "order" AS o ON a.account_id = o.account_id '
        "WHERE a.date < '1997-01-01' AND o.amount > 3000"
    )
    findings = find_inert_joins(sql, DIALECT, None)
    assert findings
    assert any("disp" in f for f in findings)


def test_a_detour_is_reported_when_the_two_ends_are_directly_related():
    # The answer reads only two tables, which have a direct relationship,
    # but the query routed between them through two others.
    sql = (
        "SELECT CAST(SUM(CASE WHEN c.gender = 'F' THEN 1 ELSE 0 END) AS REAL) "
        "* 100 / COUNT(c.client_id) FROM disp AS d "
        "JOIN account AS a ON d.account_id = a.account_id "
        "JOIN district AS dt ON a.district_id = dt.district_id "
        "JOIN client AS c ON d.client_id = c.client_id "
        "WHERE dt.\"A3\" = 'Prague'"
    )
    linked = {("client", "district")}

    def has_direct_link(left: str, right: str) -> bool:
        return (left, right) in linked or (right, left) in linked

    findings = find_inert_joins(sql, DIALECT, has_direct_link)
    assert findings
    assert any("directly related" in f for f in findings)


def test_a_genuine_bridge_in_a_many_to_many_is_left_alone():
    # The negative control for the test above: client and account have no
    # direct relationship, so going through disp is the only way to connect
    # them and the bridge's columns are rightly invisible to the answer.
    sql = (
        "SELECT c.client_id, a.date FROM client AS c "
        "JOIN disp AS d ON c.client_id = d.client_id "
        "JOIN account AS a ON d.account_id = a.account_id"
    )
    assert find_inert_joins(sql, DIALECT, lambda left, right: False) == []


def test_counting_rows_means_every_joined_table_contributes():
    # An earlier version of this check flagged `member`, because COUNT(*)
    # names no column — but the joined rows are themselves the answer, so no
    # table in the query is inert.
    sql = (
        "SELECT COUNT(*) FROM member AS m "
        "JOIN major AS j ON m.link_to_major = j.major_id "
        "WHERE j.college = 'College of Engineering'"
    )
    assert find_inert_joins(sql, DIALECT, None) == []


def test_a_deduplicated_count_is_read_as_a_deliberate_existence_filter():
    # An earlier version flagged the two bridge tables here. The
    # COUNT(DISTINCT ...) is the author saying they knew the join multiplies
    # rows, which is what a deliberate "only those that have one" looks like.
    sql = (
        "SELECT COUNT(DISTINCT c.client_id) FROM client AS c "
        "JOIN district AS dt ON c.district_id = dt.district_id "
        "JOIN disp AS dp ON dp.client_id = c.client_id "
        "JOIN account AS a ON a.account_id = dp.account_id "
        "WHERE dt.\"A2\" = 'Sokolov' AND c.gender = 'F'"
    )
    assert find_inert_joins(sql, DIALECT, None) == []


def test_unqualified_columns_stand_the_check_down():
    # Attributing a bare column to a table is a guess, and a wrong guess
    # here bounces a correct query.
    sql = "SELECT account_id FROM account AS a JOIN disp AS d ON a.account_id = d.account_id"
    assert find_inert_joins(sql, DIALECT, None) == []


def test_a_subquery_stands_the_check_down():
    sql = (
        "SELECT t.amount FROM trans AS t JOIN account AS a "
        "ON t.account_id = a.account_id WHERE t.account_id = "
        "(SELECT l.account_id FROM loan AS l ORDER BY l.amount DESC LIMIT 1)"
    )
    assert find_inert_joins(sql, DIALECT, None) == []


def test_an_unparseable_draft_is_nobody_here_s_problem():
    assert find_inert_joins("SELECT FROM WHERE", DIALECT, None) == []
    assert find_strict_boundary("SELECT FROM WHERE", DIALECT) is None
    assert find_gratuitous_aggregation("SELECT FROM WHERE", "q?", DIALECT) is None


# ----------------------------------------------------------------------
# Comparing the two results
# ----------------------------------------------------------------------


def test_results_are_compared_as_sets_not_as_ordered_rows():
    left = {"ok": True, "row_count": 2, "rows": [{"a": 1}, {"a": 2}]}
    right = {"ok": True, "row_count": 2, "rows": [{"a": 2}, {"a": 1}]}
    assert not rows_differ(left, right)


def test_a_different_row_count_is_a_difference():
    left = {"ok": True, "row_count": 1, "rows": [{"a": 375}]}
    right = {"ok": True, "row_count": 2, "rows": [{"a": 40}, {"a": 335}]}
    assert rows_differ(left, right)


def test_a_variant_that_cannot_run_counts_as_agreement():
    # The submission is then the only query that works, and complaining
    # about a comparison we failed to make would be noise.
    left = {"ok": True, "row_count": 1, "rows": [{"a": 1}]}
    right = {"ok": False, "error": "no such column", "row_count": 0, "rows": []}
    assert not rows_differ(left, right)
