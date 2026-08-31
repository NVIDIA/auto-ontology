# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What one prediction may spend, and how it fails when it cannot.

Reading whole tables into memory is unbounded work. Bounding it by taking the
first N rows answers a different question than the one asked: an arbitrary slice
chops histories and drops entities that were meant to be scored, and says
nothing about having done so. So the budget refuses, and says what would fit.
"""

import time

import pandas as pd
import pytest

from gsf.retrieval.kumo.budget import Budget, BudgetExceeded, Spend


def _frame(rows: int, heavy: bool = False) -> pd.DataFrame:
    if heavy:
        return pd.DataFrame({"note": ["x" * 500] * rows})
    return pd.DataFrame({"a": range(rows), "b": range(rows)})


def _budget(**overrides) -> Budget:
    settings = {"max_tables": 20, "max_bytes": 10 * 1024**2, "max_seconds": 300.0}
    settings.update(overrides)
    return Budget(**settings)


def test_a_prediction_within_its_budget_is_answered() -> None:
    spend = Spend(_budget())

    spend.add_table("orders", _frame(10))
    spend.check_deadline("reading tables")

    assert spend.nbytes > 0


def test_too_much_data_is_refused_rather_than_truncated() -> None:
    """The refusal is the feature: a silent truncation answers a different question."""
    spend = Spend(_budget(max_bytes=1_000))

    with pytest.raises(BudgetExceeded, match="more than it can"):
        spend.add_table("transactions", _frame(10_000))


def test_too_many_tables_is_refused() -> None:
    spend = Spend(_budget(max_tables=2))

    spend.add_table("a", _frame(1))
    spend.add_table("b", _frame(1))
    with pytest.raises(BudgetExceeded, match="tables"):
        spend.add_table("c", _frame(1))


def test_the_deadline_stops_work_that_has_run_too_long() -> None:
    spend = Spend(_budget(max_seconds=0.01))
    time.sleep(0.05)

    with pytest.raises(BudgetExceeded, match="Preparing this prediction"):
        spend.check_deadline("reading tables")


def test_a_refusal_says_what_would_make_the_request_fit() -> None:
    """An agent can act on this; "failed" tells it nothing."""
    spend = Spend(_budget(max_bytes=100))

    with pytest.raises(BudgetExceeded) as raised:
        spend.add_table("events", _frame(1_000))

    message = str(raised.value)
    assert "events" in message
    assert "narrower slice" in message


def test_a_narrow_frame_of_long_strings_is_measured_by_what_it_costs() -> None:
    """Why bytes replaced a cell count.

    One column of long strings weighs far more than two of integers over the
    same rows, so counting rows or cells bounds the shape of the data without
    bounding what holding it costs.
    """
    heavy = _frame(2_000, heavy=True)
    light = _frame(2_000)
    assert heavy.memory_usage(deep=True).sum() > light.memory_usage(deep=True).sum()

    spend = Spend(_budget(max_bytes=100_000))

    with pytest.raises(BudgetExceeded):
        spend.add_table("notes", heavy)


@pytest.mark.parametrize("value", ["0", "-5", "not-a-number", ""])
def test_a_nonsense_limit_falls_back_to_the_default(
    value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A misconfigured cap must not become an unbounded one."""
    monkeypatch.setenv("KUMO_MAX_TABLES", value)

    assert Budget.from_env().max_tables == 20


def test_a_budget_of_one_table_is_written_as_one_table() -> None:
    """A refusal is read by a person, and 'more than 1 tables' reads as a bug."""
    spend = Spend(Budget(max_tables=1, max_bytes=1 << 40, max_seconds=300))
    spend.add_table("customers", pd.DataFrame({"a": [1]}))

    with pytest.raises(BudgetExceeded, match=r"more than 1 table\."):
        spend.add_table("orders", pd.DataFrame({"a": [1]}))


def test_a_larger_budget_is_written_with_a_thousands_separator() -> None:
    spend = Spend(Budget(max_tables=1_500, max_bytes=1 << 40, max_seconds=300))
    spend.tables = 1_500

    with pytest.raises(BudgetExceeded, match="1,500 tables"):
        spend.add_table("orders", pd.DataFrame({"a": [1]}))
