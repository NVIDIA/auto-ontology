# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for inferring how a date column stores its values."""

from __future__ import annotations

import pytest

from auto_ontology.semantic.date_format import infer_date_format, is_date_type


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (["2012-08-24", "1990-01-01", "2019-12-31"], "YYYY-MM-DD"),
        (["2012-08-24 15:04:05", "1990-01-01 00:00:00"], "YYYY-MM-DD HH:MM:SS"),
        (["2012-08-24T15:04:05", "1990-01-01T00:00:00"], "YYYY-MM-DDTHH:MM:SS"),
        (["2014/07/09", "2015/01/31"], "YYYY/MM/DD"),
        (["20120824", "19900101"], "YYYYMMDD"),
        (["12:30:00", "09:15:45"], "HH:MM:SS"),
    ],
)
def test_unambiguous_notations_are_reported(values: list[str], expected: str) -> None:
    assert infer_date_format(values) == expected


def test_fractional_seconds_are_reported_at_the_width_stored() -> None:
    """``%f`` spans one to six digits, and an equality predicate cannot.
    Every codebase_community timestamp ends in a single ``.0``. Reporting six
    places would tell the model to write a literal the column never holds.
    """
    assert (
        infer_date_format(["2010-07-19 19:39:07.0", "2010-09-15 21:08:26.0"])
        == "YYYY-MM-DD HH:MM:SS.f"
    )
    assert (
        infer_date_format(["2010-07-19 19:39:07.123456"])
        == "YYYY-MM-DD HH:MM:SS.ffffff"
    )


def test_a_varying_fraction_is_reported_at_its_widest() -> None:
    assert (
        infer_date_format(["2010-07-19 19:39:07.0", "2010-09-15 21:08:26.123"])
        == "YYYY-MM-DD HH:MM:SS.fff"
    )


def test_day_first_is_settled_by_a_day_past_the_twelfth() -> None:
    """``13/07/2011`` cannot be month-first, so the sample decides it.
    This is why the inference validates rather than pattern-matches: no shape
    analysis distinguishes ``DD/MM`` from ``MM/DD``, but a month of 13 does not
    exist and ``strptime`` says so.
    """
    assert infer_date_format(["03/04/2011", "13/07/2011"]) == "DD/MM/YYYY"
    assert infer_date_format(["04/03/2011", "07/13/2011"]) == "MM/DD/YYYY"


def test_an_undecidable_order_is_left_unstated() -> None:
    """Every value below reads both ways, so neither may be claimed.
    Reporting a coin flip is worse than reporting nothing: the model treats the
    notation as fact and would build predicates on it.
    """
    assert infer_date_format(["03/04/2011", "05/06/2011"]) is None


def test_six_digits_split_by_whether_the_leading_four_are_a_year() -> None:
    """``201208`` is a year-month; ``950324`` is a two-digit-year date.
    Both are six digits and both parse as ``%y%m%d``, so length alone cannot
    tell them apart. debit_card_specializing stores the former and financial the
    latter, and a predicate written for one finds nothing in the other.
    """
    assert infer_date_format(["201208", "201301", "201412"]) == "YYYYMM"
    assert infer_date_format(["950324", "970213", "981231"]) == "YYMMDD"


def test_a_column_mixing_notations_yields_nothing() -> None:
    assert infer_date_format(["2012-08-24", "24/08/2012"]) is None


def test_values_that_are_not_dates_yield_nothing() -> None:
    assert infer_date_format(["not a date", "also not"]) is None
    assert infer_date_format([]) is None
    assert infer_date_format([None, "", "   "]) is None


def test_an_implausible_year_is_not_accepted_as_one() -> None:
    """A four-digit run only reads as a year inside a believable range.
    Without the range check ``9503-24`` style values would be read as ISO dates
    in the year 9503, and the notation reported for the column would be wrong
    rather than absent.
    """
    assert infer_date_format(["9503-01-01"]) is None


def test_nulls_and_duplicates_do_not_disturb_the_reading() -> None:
    values = [None, "2012-08-24", "2012-08-24", "", "1990-01-01"]
    assert infer_date_format(values) == "YYYY-MM-DD"


def test_non_string_values_are_read_through_their_rendering() -> None:
    """A driver returning real date objects still yields a notation.
    Postgres hands back ``datetime.date``; SQLite hands back the stored text.
    Both reach here as whatever ``str`` makes of them, and the ISO rendering of
    a native date is the form a predicate compares against anyway.
    """
    from datetime import date

    assert infer_date_format([date(2012, 8, 24), date(1990, 1, 1)]) == "YYYY-MM-DD"


@pytest.mark.parametrize(
    ("declared", "expected"),
    [
        ("DATE", True),
        ("datetime", True),
        ("TIMESTAMP WITHOUT TIME ZONE", True),
        ("TEXT", False),
        ("INTEGER", False),
        (None, False),
    ],
)
def test_is_date_type(declared: str | None, expected: bool) -> None:
    assert is_date_type(declared) is expected
