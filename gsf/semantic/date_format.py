# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Infer how a date column stores its values.
Date columns are the one type kept out of sample-value profiling, on the
reasoning that a concrete date carries no business meaning. That holds for
*meaning* and fails for *mechanics*: a query has to compare against the stored
form, and nothing else in the prompt says what it is. ``950324`` and
``1995-03-24`` describe the same day and admit no common predicate, so a model
that guesses wrong writes SQL that runs, returns nothing, and looks correct.
The format is inferred by validation rather than by pattern-matching a shape:
each candidate is handed to :func:`datetime.strptime` against every sampled
value, and only a candidate that parses all of them is reported. That makes an
impossible reading fail on its own — ``13/07/2011`` cannot be month-first — and
keeps the inference honest when the sample disagrees with itself, since a mixed
column simply yields no claim rather than a confident wrong one.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime

# Declared type substrings that make a column temporal.
DATE_TYPE_TOKENS = ("date", "time", "timestamp", "datetime")

# How many distinct values the inference looks at. Enough that a day-first
# column almost certainly shows a day past the 12th, cheap enough to run on
# every date column of every table.
_MAX_VALUES = 200

# Longest value worth trying to parse; anything beyond this is prose, not a date.
_MAX_VALUE_LEN = 40

# Candidates in descending specificity, each with the notation reported to the
# model. Order settles only genuine ties — a candidate that cannot parse every
# value is out regardless of where it sits.
_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("%Y-%m-%d %H:%M:%S.%f", "YYYY-MM-DD HH:MM:SS.f"),
    ("%Y-%m-%dT%H:%M:%S.%f", "YYYY-MM-DDTHH:MM:SS.f"),
    ("%Y-%m-%d %H:%M:%S", "YYYY-MM-DD HH:MM:SS"),
    ("%Y-%m-%dT%H:%M:%S", "YYYY-MM-DDTHH:MM:SS"),
    ("%Y-%m-%d %H:%M", "YYYY-MM-DD HH:MM"),
    ("%Y-%m-%d", "YYYY-MM-DD"),
    ("%Y/%m/%d", "YYYY/MM/DD"),
    ("%d/%m/%Y", "DD/MM/YYYY"),
    ("%m/%d/%Y", "MM/DD/YYYY"),
    ("%d-%m-%Y", "DD-MM-YYYY"),
    ("%m-%d-%Y", "MM-DD-YYYY"),
    ("%d.%m.%Y", "DD.MM.YYYY"),
    ("%d/%m/%y", "DD/MM/YY"),
    ("%m/%d/%y", "MM/DD/YY"),
    ("%d-%m-%y", "DD-MM-YY"),
    ("%m-%d-%y", "MM-DD-YY"),
    ("%d.%m.%y", "DD.MM.YY"),
    ("%Y%m%d", "YYYYMMDD"),
    ("%Y%m", "YYYYMM"),
    ("%y%m%d", "YYMMDD"),
    ("%H:%M:%S", "HH:MM:SS"),
    ("%H:%M", "HH:MM"),
    ("%Y", "YYYY"),
)

# Readings that differ only in which component comes first. When both parse
# every value the sample genuinely cannot tell them apart, and saying nothing
# beats a coin flip the model would take as fact.
_INDISTINGUISHABLE: tuple[frozenset[str], ...] = (
    frozenset({"DD/MM/YYYY", "MM/DD/YYYY"}),
    frozenset({"DD-MM-YYYY", "MM-DD-YYYY"}),
    frozenset({"DD/MM/YY", "MM/DD/YY"}),
    frozenset({"DD-MM-YY", "MM-DD-YY"}),
    frozenset({"DD.MM.YY", "MM.DD.YY"}),
    frozenset({"D/M/YY", "M/D/YY"}),
    frozenset({"D-M-YY", "M-D-YY"}),
    frozenset({"D.M.YY", "M.D.YY"}),
    frozenset({"D/M/YYYY", "M/D/YYYY"}),
    frozenset({"D-M-YYYY", "M-D-YYYY"}),
    frozenset({"D.M.YYYY", "M.D.YYYY"}),
)

# A four-digit run only reads as a year inside this range. Without the guard
# ``950324`` parses as year 9503 plus month 24 — rejected — but ``201208``
# parses as both YYYYMM and YYMMDD, and only the year check picks the right one.
_YEAR_MIN, _YEAR_MAX = 1900, 2099


def is_date_type(data_type: str | None) -> bool:
    """Whether a declared column type is a date/time type."""
    if not data_type:
        return False
    lowered = data_type.lower()
    return any(token in lowered for token in DATE_TYPE_TOKENS)


def _plausible_year(parsed: datetime) -> bool:
    return _YEAR_MIN <= parsed.year <= _YEAR_MAX


def _parses_all(pattern: str, values: list[str]) -> bool:
    """Whether every value reads as *pattern*, landing in a believable year.
    The year check is what separates the two readings of a six-digit date.
    ``%y`` is exempt: two digits carry no century, so Python's 1969–2068 window
    is the only year available and rejecting it would rule the format out
    entirely.
    """
    century_known = "%Y" in pattern
    for value in values:
        try:
            parsed = datetime.strptime(value, pattern)
        except ValueError:
            return False
        if century_known and not _plausible_year(parsed):
            return False
    return True


def _widen_fraction(label: str, values: list[str]) -> str:
    """Restate a fractional-seconds notation at the width actually stored.
    ``%f`` accepts one to six digits, so a column holding ``19:39:07.0`` parses
    under the same pattern as one holding ``19:39:07.123456``. The label must
    not paper over that: an equality predicate has to reproduce the stored
    string exactly, and codebase_community's timestamps all end in a single
    ``.0``. Where the width varies the widest is reported, since that is the
    form a literal has to be able to hold.
    """
    widths = {len(value.rpartition(".")[2]) for value in values if "." in value}
    widest = max(widths, default=1)
    return label.replace(".f", "." + "f" * widest)


def _narrow_padding(label: str, values: list[str]) -> str:
    """Restate a date notation at the digit width actually stored.

    ``strptime`` accepts an unpadded month or day, so ``4/4/20`` and
    ``04/04/20`` both parse as ``%m/%d/%y``. The notation must not blur them:
    the model writes an equality predicate against the stored string, and
    regional_sales stores ``4/4/20``, which ``04/04/20`` never matches. Only
    date-only notations are narrowed -- a label carrying a time would have its
    minutes rewritten along with its month.
    """
    if any(ch in label for ch in (" ", "T", ":")):
        return label
    separator = next((sep for sep in ("/", "-", ".") if sep in label), None)
    if separator is None:
        return label
    if any(len(part) == 1 for value in values for part in value.split(separator)):
        return label.replace("MM", "M").replace("DD", "D")
    return label


def infer_date_format(values: Iterable[object]) -> str | None:
    """The notation every sampled value follows, or None if that is not one thing.
    Returns a human notation (``YYYY-MM-DD``) rather than a strptime pattern,
    because the reader is a language model writing SQL against the column, not
    a parser.
    """
    seen: list[str] = []
    unique: set[str] = set()
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if not text or len(text) > _MAX_VALUE_LEN or text in unique:
            continue
        unique.add(text)
        seen.append(text)
        if len(seen) >= _MAX_VALUES:
            break
    if not seen:
        return None

    matches = [
        (pattern, _narrow_padding(label, seen))
        for pattern, label in _CANDIDATES
        if _parses_all(pattern, seen)
    ]
    if not matches:
        return None
    pattern, winner = matches[0]
    labels = {label for _, label in matches}
    for group in _INDISTINGUISHABLE:
        if winner in group and len(group & labels) > 1:
            return None
    return _widen_fraction(winner, seen) if "%f" in pattern else winner
