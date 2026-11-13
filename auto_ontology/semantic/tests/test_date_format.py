# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for inferring how a date column stores its values."""

from __future__ import annotations

from datetime import datetime

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
    ("values", "expected"),
    [
        (["2024.03.15", "2023.12.01"], "YYYY.MM.DD"),
        (["2024. 03. 15.", "2023. 12. 01."], "YYYY. MM. DD."),
        (["2024년 03월 15일"], "YYYY년 MM월 DD일"),
        (["2024年03月15日"], "YYYY年MM月DD日"),
    ],
)
def test_year_first_forms_outside_the_slash_and_dash_families_are_read(
    values: list[str], expected: str
) -> None:
    """Dotted and East Asian year-first columns used to get no hint at all.
    ``_CANDIDATES`` held ``DD.MM.YYYY`` but no year-first dotted form, and no
    spelling that puts a character between the components, so the sample parsed
    nowhere and the module stayed silent about a column the model still has to
    write predicates against.
    """
    assert infer_date_format(values) == expected


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (["2024. 3. 15."], "YYYY. M. DD."),
        (["2024년 3월 15일"], "YYYY년 M월 DD일"),
        (["2024年3月15日"], "YYYY年M月DD日"),
        (["2024. 3. 5."], "YYYY. M. D."),
    ],
)
def test_an_unpadded_component_is_reported_one_digit_wide(
    values: list[str], expected: str
) -> None:
    """The width is read per component, whatever separates them.
    Narrowing used to bail out on any notation carrying a space, which is every
    East Asian spelling and the dotted one with spaces after each dot, so
    ``2024. 3. 15.`` was reported as ``YYYY. MM. DD.`` and the model would write
    a literal the column never holds.
    """
    assert infer_date_format(values) == expected


def test_a_narrow_component_does_not_hide_an_undecidable_order() -> None:
    """``4/04/20`` stores a one-digit month and still reads both ways.
    The reading is compared on the padded spelling of each pattern, so the width
    a column happens to use cannot turn a coin flip into a confident claim.
    """
    assert infer_date_format(["4/4/20"]) is None
    assert infer_date_format(["4/04/20"]) is None


def test_a_component_is_widened_to_the_widest_value_stored() -> None:
    """``13`` cannot be held by ``D``, so the whole column is called ``DD``.
    Same rule fractional seconds already follow: report the form a literal has
    to be able to hold.
    """
    assert infer_date_format(["4/4/20", "13/6/20"]) == "DD/M/YY"


def test_repeated_whitespace_keeps_the_widths_it_measures() -> None:
    """``strptime`` matches a run of spaces in the format against any run of
    whitespace, so the sample still says how many fractional digits are stored.
    Aligning on one literal space dropped every width and reported ``.f`` for a
    value holding three digits.
    """
    assert infer_date_format(["2010-07-19  19:39:07.123"]) == "YYYY-MM-DD HH:MM:SS.fff"


def test_a_space_padded_day_keeps_the_fraction_width() -> None:
    """``strptime`` reads ``%d`` space-padded, so ``2010-07- 9`` is a valid sample.

    Without that alternative the alignment failed, every width measured for the
    value was dropped, and the notation said ``.f`` for a column storing three
    fractional digits -- a notation that describes a different stored width.
    """
    assert infer_date_format(["2010-07- 9 19:39:07.123"]) == "YYYY-MM-DD HH:MM:SS.fff"


def test_a_space_padded_hour_is_measured_when_strptime_reads_it() -> None:
    """Python 3.13 reads ``%H`` space-padded where 3.12 refuses it.

    Where the running interpreter accepts the sample the alignment has to measure
    the hour like any other, and where it refuses the value is not this format.
    """
    value = "2010-07-09T 9:00:00"
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%S")
    except ValueError:
        assert infer_date_format([value]) is None
    else:
        assert infer_date_format([value]) == "YYYY-MM-DDTHH:MM:SS"


def test_a_compact_date_keeps_the_component_boundaries_strptime_uses() -> None:
    """``2024131`` is 2024-01-31, because a month is at most 12.
    A digit group that took as many digits as it could would split it as year
    2024, month 13 and day 1, and report the month at two digits and the day at
    one -- the opposite of what the column stores.
    """
    assert infer_date_format(["2024131"]) == "YYYYMDD"


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
