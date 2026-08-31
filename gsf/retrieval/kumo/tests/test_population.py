# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Questions that cover more entities than one run can score.

A total over part of a population, presented as a total, is wrong in a way the
reader cannot see. These cover which questions are refused, which are answered
with what was left out named, and what the refusal says.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from gsf.retrieval.kumo.pql_gen import (
    PqlPopulationTooLargeError,
    _count_entities,
    _population_refusal,
    _resolve_indices,
    _UNCOUNTED_POPULATION,
)

PQL = "PREDICT SUM(orders.amt, 0, 30) FOR EACH customers.cid"


class _Warehouse:
    """A source of ``rows`` entities that honours LIMIT and COUNT."""

    def __init__(self, rows: int, counts: bool = True) -> None:
        self.rows = rows
        self.counts = counts
        self.queries: list[str] = []

    def execute(self, sql: str) -> pd.DataFrame:
        self.queries.append(sql)
        if "COUNT(" in sql.upper():
            if not self.counts:
                raise RuntimeError("the warehouse refused the count")
            return pd.DataFrame({"c": [self.rows]})
        limit = int(sql.rsplit("LIMIT", 1)[1])
        return pd.DataFrame({"cid": list(range(min(limit, self.rows)))})


def test_a_population_that_fits_is_not_counted_twice() -> None:
    """A count is only worth asking for when the LIMIT could not say."""
    warehouse = _Warehouse(rows=40)

    ids, population = _resolve_indices(PQL, None, warehouse, 100, {"customers": "c"})

    assert (len(ids), population) == (40, 40)
    assert not any("COUNT(" in q.upper() for q in warehouse.queries)


def test_a_population_at_the_cap_is_counted() -> None:
    """A LIMIT that came back full cannot say how much it left behind."""
    warehouse = _Warehouse(rows=48_391)

    ids, population = _resolve_indices(PQL, None, warehouse, 2_000, {"customers": "c"})

    assert (len(ids), population) == (2_000, 48_391)
    assert any("COUNT(" in q.upper() for q in warehouse.queries)


def test_a_population_that_cannot_be_counted_is_not_treated_as_small() -> None:
    """An unknown population read as a small one is what lets a slice pass as a whole."""
    warehouse = _Warehouse(rows=48_391, counts=False)

    _, population = _resolve_indices(PQL, None, warehouse, 2_000, {"customers": "c"})

    assert population == _UNCOUNTED_POPULATION


def test_ids_already_loaded_report_their_own_count() -> None:
    """Nothing needs asking when the entities are already in hand."""
    ids, population = _resolve_indices(
        PQL, None, _Warehouse(rows=0), 2, available_entity_ids={"customers": [1, 2, 3]}
    )

    assert (ids, population) == ([1, 2], 3)


def test_a_count_failure_is_not_a_zero() -> None:
    class Refusing:
        def execute(self, sql: str) -> pd.DataFrame:
            raise RuntimeError("permission denied")

    assert _count_entities(Refusing(), "customers", "cid") == _UNCOUNTED_POPULATION


def test_the_refusal_says_what_was_asked_and_what_to_ask_instead() -> None:
    message = _population_refusal(PQL, 48_391, 10_000)

    assert "48,391" in message
    assert "10,000" in message
    assert "customers" in message
    assert "narrower" in message


def test_the_refusal_does_not_send_the_asker_to_an_environment_variable() -> None:
    """A cap is an operator's decision; the asker can only narrow the question."""
    message = _population_refusal(PQL, 48_391, 10_000)

    assert "KUMO_" not in message
    assert "environment" not in message.lower()


def test_an_uncountable_population_is_not_reported_as_a_number() -> None:
    message = _population_refusal(PQL, _UNCOUNTED_POPULATION, 10_000)

    assert "unknown number" in message
    assert str(_UNCOUNTED_POPULATION) not in message


def test_a_refusal_is_not_something_the_repair_loop_can_fix() -> None:
    """No rewriting of the PQL makes the population smaller."""
    assert issubclass(PqlPopulationTooLargeError, ValueError)


def test_a_truncated_answer_names_what_it_left_out() -> None:
    from gsf.retrieval.kumo.predictor import _format_result

    class Result:
        success = True
        question = "who will spend most?"
        pql = PQL
        note = None
        num_entities = 2_000
        population = 48_391
        truncated = True
        columns = ["cid"]
        rows: list[dict[str, Any]] = [{"cid": 1}]

    response = _format_result(Result())["response"]

    assert "2,000 of 48,391" in response
    assert "not the whole population" in response


def test_an_answer_over_everything_does_not_apologise_for_it() -> None:
    from gsf.retrieval.kumo.predictor import _format_result

    class Result:
        success = True
        question = "who will spend most?"
        pql = PQL
        note = None
        num_entities = 40
        population = 40
        truncated = False
        columns = ["cid"]
        rows: list[dict[str, Any]] = [{"cid": 1}]

    response = _format_result(Result())["response"]

    assert "KumoRFM scored 40 entities." in response
    assert "whole population" not in response


def test_predict_all_refuses_a_population_it_cannot_cover() -> None:
    """``predict_all`` is ported code with no caller in GSF today.

    Pinned anyway: it is the one refusal site whose raise nothing would catch,
    so whoever wires it up should find the contract already stated rather than
    discover it as an uncaught exception in production.
    """
    from gsf.retrieval.kumo import pql_gen

    warehouse = _Warehouse(rows=48_391)

    with pytest.raises(PqlPopulationTooLargeError, match="48,391"):
        pql_gen.predict_all(
            PQL,
            kumo_model=None,
            connector=warehouse,
            entity_sql=None,
            table_names={"customers": "c"},
            max_entities=2_000,
        )
