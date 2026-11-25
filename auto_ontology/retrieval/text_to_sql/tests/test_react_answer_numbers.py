# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A submitted answer that cites none of its rows' numbers goes back once (#252)."""

from __future__ import annotations

import json

import pytest
import sqlglot

from auto_ontology.retrieval.text_to_sql.agents.sql_execution import QueryResponse
from auto_ontology.retrieval.text_to_sql.react.answer_numbers import (
    contradicted_numbers,
    numbers_check,
    render_rows,
    rows_from_result,
)
from auto_ontology.retrieval.text_to_sql.react.context import ToolContext

KESTREL_QUESTION = "How many ASNs did Kestrel Components Ltd ship, and how many units?"
KESTREL_SQL = (
    "SELECT COUNT(DISTINCT a.asn_id) AS asn_count, SUM(l.units) AS total_units "
    "FROM asn a JOIN asn_lines l ON l.asn_id = a.asn_id "
    "JOIN suppliers s ON s.supplier_id = a.supplier_id "
    "WHERE s.name = 'Kestrel Components Ltd'"
)
KESTREL_ROWS = [{"asn_count": 17999, "total_units": 9550743}]

BACKORDER_QUESTION = (
    "how many customers had backorders? break that by year, from 2000 to 2020?"
)
BACKORDER_ROWS = [
    {"year": 2013, "customers_with_backorders": 583},
    {"year": 2014, "customers_with_backorders": 612},
    {"year": 2015, "customers_with_backorders": 634},
    {"year": 2016, "customers_with_backorders": 512},
]


def test_the_kestrel_answer_from_another_query_is_caught() -> None:
    """Reproduced on main: rows 17,999 / 9,550,743, prose "1 ASN, 587 units"."""
    response = "Kestrel Components Ltd shipped 1 ASN with a total of 587 units."

    assert contradicted_numbers(
        response, KESTREL_ROWS, KESTREL_QUESTION, KESTREL_SQL
    ) == ["587"]


def test_the_wwi_backorder_answer_from_another_query_is_caught() -> None:
    """Reproduced on main: rows 583/612/634/512, prose from an earlier query."""
    response = (
        "Backorders only show up in 2013–2016: 2013: 1,872 customers, "
        "2014: 2,174, 2015: 2,488, 2016: 1,004."
    )

    assert contradicted_numbers(response, BACKORDER_ROWS, BACKORDER_QUESTION) == [
        "1,872",
        "2,174",
        "2,488",
        "1,004",
    ]


@pytest.mark.parametrize(
    "response",
    [
        "Kestrel shipped 17,999 ASNs and 9,550,743 units.",
        "Kestrel shipped 17999 ASNs and 9550743 units.",
        "Kestrel shipped about 18K ASNs and 9.55M units.",
        "Kestrel shipped 18.0k ASNs and 9.6 million units.",
        "Kestrel shipped roughly 9.55 million units across ~18,000 ASNs.",
        "Kestrel's units total $9.55M.",
        "Kestrel's units total $9.55 M across 18 K ASNs.",
    ],
)
def test_the_kestrel_answer_from_its_rows_passes(response: str) -> None:
    assert contradicted_numbers(response, KESTREL_ROWS, KESTREL_QUESTION) == []


def test_decimal_cells_serialized_as_text_are_numbers() -> None:
    """Postgres SUMs are DECIMAL; the result JSON holds them as strings."""
    rows = [{"asn_count": 17999, "total_units": "9550743.00"}]

    assert contradicted_numbers("1 ASN with 587 units.", rows) == ["587"]
    assert contradicted_numbers("9.55M units in 17,999 ASNs.", rows) == []


def test_a_percentage_may_restate_a_ratio_or_a_percent() -> None:
    assert contradicted_numbers("49.3% of orders were late.", [{"r": 0.4932}]) == []
    assert contradicted_numbers("49.3% of orders were late.", [{"p": 49.32}]) == []
    assert contradicted_numbers("49.3 percent were late.", [{"r": 0.4932}]) == []
    assert contradicted_numbers("61.8% of orders were late.", [{"r": 0.4932}]) == [
        "61.8%"
    ]


def test_the_same_backorder_answer_stated_from_its_rows_passes() -> None:
    response = "2013: 583 customers, 2014: 612, 2015: 634, 2016: 512 (2,341 in all)."

    assert contradicted_numbers(response, BACKORDER_ROWS, BACKORDER_QUESTION) == []


def test_a_row_count_or_column_sum_supports_a_number() -> None:
    rows = [{"name": f"c{i}", "units": 25} for i in range(20)]

    assert contradicted_numbers("20 customers bought 500 units.", rows) == []


def test_context_beside_a_row_value_is_not_a_contradiction() -> None:
    """Measured on WWI: answers cite their row and add exploration context."""
    rows = [{"salesperson_to_employee_ratio": 0.5263157894736842}]
    response = "The ratio is about 0.526: 10 of the 19 employees are salespeople."

    assert contradicted_numbers(response, rows) == []


def test_citing_a_small_row_value_counts_as_citing_the_rows() -> None:
    """Measured on WWI: "The count is 0. All 227 items have a tax rate." """
    rows = [{"count": 0}]

    assert (
        contradicted_numbers("The count is 0. All 227 items have a rate.", rows) == []
    )


def test_years_dates_times_and_ordinals_are_not_checked() -> None:
    rows = [{"orders": 73595}]
    response = (
        "73,595 orders. Data ends March 31, 2016 (2016-05-31T00:00:00) at 23:59; "
        "Black Friday was 29 November. It is the 21st year."
    )

    assert contradicted_numbers(response, rows) == []
    assert contradicted_numbers("Orders since 2013, by 31 May.", rows) == []


def test_question_sql_and_quoted_sql_numbers_are_not_checked() -> None:
    rows = [{"orders": 812}]
    sql = "SELECT COUNT(*) AS orders FROM orders WHERE customer_id = 4321 LIMIT 250"

    assert (
        contradicted_numbers("Customer 4321 placed some orders.", rows, "", sql) == []
    )
    assert contradicted_numbers("Orders over 500 units.", rows, "orders > 500?") == []
    assert (
        contradicted_numbers("Ran `WHERE qty > 9000`:\n```sql\nLIMIT 777\n```", rows)
        == []
    )


def test_identifiers_top_n_and_small_counts_are_not_checked() -> None:
    rows = [{"registration": "WWI-321-A", "trips": 40}]
    response = "Of the top 25 vehicles, 3 qualify: WWI-321-A (#5531) on 48mmx75m tape."

    assert contradicted_numbers(response, rows) == []


def test_rows_without_numbers_empty_or_null_results_are_not_judged() -> None:
    names = [{"customername": "Tailspin Toys (Head Office)"}]

    assert contradicted_numbers("It bought 291 units.", names) == []
    assert contradicted_numbers("There are 1,872 orders.", []) == []
    assert contradicted_numbers("The total is 9,550,743.", [{"total": None}]) == []


def test_null_cells_beside_numbers_are_skipped() -> None:
    rows = [{"year": 2013, "n": None}, {"year": 2014, "n": 583}]

    assert contradicted_numbers("583 in 2014, none in 2013.", rows) == []
    assert contradicted_numbers("1,872 in 2014.", rows) == ["1,872"]


def test_rows_are_read_from_json_record_strings() -> None:
    result = [json.dumps([{"order_count": 9}, {"order_count": None}])]

    assert rows_from_result(result) == [{"order_count": 9}, {"order_count": None}]
    assert rows_from_result(None) == []


def test_rows_are_rendered_with_nulls_and_a_cap() -> None:
    rows = [{"id": i, "v": None} for i in range(25)]

    shown = render_rows(rows)

    assert shown.startswith("25 rows, first 20 shown:\nid | v\n0 | NULL")
    assert render_rows([]) == "0 rows."


def _ctx(**kwargs: object) -> ToolContext:
    return ToolContext(retriever=None, connectors=[], **kwargs)


def _ran(rows: list[dict]) -> QueryResponse:
    return QueryResponse(result=[json.dumps(rows)], sliced=False)


def test_the_check_sends_back_once_with_the_numbers_and_the_rows() -> None:
    ctx = _ctx(question=KESTREL_QUESTION)
    response = "1 ASN with a total of 587 units."

    first = numbers_check(ctx, KESTREL_SQL, response, _ran(KESTREL_ROWS))
    second = numbers_check(ctx, KESTREL_SQL, response, _ran(KESTREL_ROWS))

    assert first is not None
    assert "cites 587," in first
    assert "asn_count | total_units\n17999 | 9550743" in first
    assert KESTREL_SQL in first
    assert second is None


def test_the_check_stays_out_of_a_last_attempt_or_without_queries_left() -> None:
    response = "1 ASN with a total of 587 units."

    last = _ctx(last_answer_attempt=True)
    no_queries = _ctx(can_run_queries=lambda: False)

    assert numbers_check(last, KESTREL_SQL, response, _ran(KESTREL_ROWS)) is None
    assert numbers_check(no_queries, KESTREL_SQL, response, _ran(KESTREL_ROWS)) is None


def test_the_check_can_be_turned_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEXT_TO_SQL_REACT_ANSWER_NUMBERS_CHECK", "false")

    assert numbers_check(_ctx(), KESTREL_SQL, "587 units", _ran(KESTREL_ROWS)) is None


def test_a_percentage_of_two_cells_or_of_a_column_passes() -> None:
    assert contradicted_numbers("49.3% were late.", [{"late": 493, "all": 1000}]) == []
    shares = [{"region": "North", "units": 300}, {"region": "South", "units": 700}]
    assert contradicted_numbers("North has 30% of units.", shares) == []


def test_a_cell_over_itself_does_not_support_100_percent() -> None:
    assert contradicted_numbers("100% of orders were late.", [{"late": 493}]) == [
        "100%"
    ]
    rows = [{"late": 493, "all": 1000}]
    assert contradicted_numbers("100% of orders were late.", rows) == ["100%"]


def test_a_negative_sql_number_keeps_its_sign() -> None:
    sql = "SELECT id, balance FROM accounts WHERE balance < -100"
    rows = [{"id": 501, "balance": -250}]

    assert (
        contradicted_numbers("Every account below -100 is listed.", rows, "", sql) == []
    )


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT id, balance FROM accounts LIMIT 100",
        "SELECT id, balance - 100 AS balance FROM accounts",
    ],
)
def test_a_positive_sql_number_does_not_cover_its_negative(sql: str) -> None:
    rows = [{"id": 501, "balance": -250}]

    assert contradicted_numbers("There are -100 accounts.", rows, "", sql) == ["-100"]


@pytest.mark.parametrize(
    ("sql", "dialect"),
    [
        ("SELECT TOP 10 id, balance FROM accounts WHERE balance < -100", "tsql"),
        ("SELECT `id`, `balance` FROM `accounts` WHERE `balance` < -100", "mysql"),
    ],
)
def test_a_negative_number_keeps_its_sign_in_the_connector_dialect(
    sql: str, dialect: str
) -> None:
    rows = [{"id": 501, "balance": -250}]
    response = "Every account below -100 is listed."

    # The default dialect cannot read either query, and the fallback has no sign.
    assert contradicted_numbers(response, rows, "", sql) == ["-100"]
    assert contradicted_numbers(response, rows, "", sql, dialect=dialect) == []
    statement = sqlglot.parse_one(sql, read=dialect)
    assert contradicted_numbers(response, rows, "", sql, statement) == []
    assert numbers_check(_ctx(), sql, response, _ran(rows), statement) is None


def test_unparsable_sql_still_names_its_numbers() -> None:
    sql = "SELECT TOP 5000 id FROM accounts WHERE ((("

    assert contradicted_numbers("We list 5000 accounts.", [{"id": 501}], "", sql) == []


def test_a_scaled_number_is_compared_at_its_scale() -> None:
    rows = [{"revenue": 1234567.89}]

    assert contradicted_numbers("Revenue was $1.2M.", rows) == []
    assert contradicted_numbers("Revenue was $1.5M.", rows) == ["$1.5M"]


def test_a_spaced_letter_that_may_not_be_a_scale_is_read_both_ways() -> None:
    assert contradicted_numbers("It is 300 m long.", [{"length": 300}]) == []
    assert contradicted_numbers("Group 3 B has 1,872.", [{"n": 583}]) == ["1,872"]


def test_both_readings_count_in_text_cells_and_the_question() -> None:
    rows = [{"description": "300 m long", "n": 7}]
    assert contradicted_numbers("The length is 300 metres.", rows) == []
    race = [{"winner": "Ada", "lane": 42}]
    question = "Who won the 300 m event?"
    assert contradicted_numbers("Ada won the 300 metre event.", race, question) == []
