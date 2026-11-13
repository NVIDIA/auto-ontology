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

import re
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

# Candidates in descending specificity. A candidate is only a `strptime` format:
# the notation handed to the model is derived from it and from the sample, so a
# padded and an unpadded spelling of the same form share one entry.
_CANDIDATES: tuple[str, ...] = (
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%Y.%m.%d",
    "%Y. %m. %d.",
    "%Y년 %m월 %d일",
    "%Y年%m月%d日",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%d-%m-%Y",
    "%m-%d-%Y",
    "%d.%m.%Y",
    "%d/%m/%y",
    "%m/%d/%y",
    "%d-%m-%y",
    "%m-%d-%y",
    "%d.%m.%y",
    "%Y%m%d",
    "%Y%m",
    "%y%m%d",
    "%H:%M:%S",
    "%H:%M",
    "%Y",
)

# Readings that differ only in which component comes first. When both parse
# every value the sample genuinely cannot tell them apart, and saying nothing
# beats a coin flip the model would take as fact. Pairs are written at the
# widest spelling each component can take, because that is the form the two
# patterns are compared on: `4/04/20` stores a narrower month but reads both ways
# all the same.
_INDISTINGUISHABLE: tuple[frozenset[str], ...] = (
    frozenset({"DD/MM/YYYY", "MM/DD/YYYY"}),
    frozenset({"DD-MM-YYYY", "MM-DD-YYYY"}),
    frozenset({"DD/MM/YY", "MM/DD/YY"}),
    frozenset({"DD-MM-YY", "MM-DD-YY"}),
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


# Each `strptime` token and the notation it stands for. `%m` and `%d` are the only
# tokens rendered one digit wide: a month or a day may be stored unpadded, and the
# notation has to say so for an equality predicate to match. Minutes, seconds and
# fractional seconds keep their width whatever the sample says, since a stored time
# is always zero-padded.
_NOTATION: dict[str, str] = {
    "%Y": "YYYY",
    "%y": "YY",
    "%m": "MM",
    "%d": "DD",
    "%H": "HH",
    "%M": "MM",
    "%S": "SS",
    "%f": "f",
}

# The digit runs `strptime` itself accepts, token by token. A month is one or two
# digits taken from the twelve that exist, so `2024131` reads as 2024-01-31 and not
# as month 13 plus day 1: a group that simply took as many digits as it could would
# split that value the wrong way and report the padding of the wrong component.
# `%d` and `%H` are the two tokens `strptime` also reads space-padded (`2010-07- 9`),
# so they keep that alternative here; the other numeric tokens refuse it.
_COMPONENT_RE: dict[str, str] = {
    "%Y": r"\d{4}",
    "%y": r"\d{2}",
    "%m": r"1[0-2]|0[1-9]|[1-9]",
    "%d": r"3[01]|[0-2]\d|\d| [1-9]",
    "%H": r"2[0-3]|[0-1]\d|\d| [0-9]",
    "%M": r"[0-5]\d|\d",
    "%S": r"6[01]|[0-5]\d|\d",
    "%f": r"\d{1,6}",
}

_NARROWABLE = frozenset({"%m", "%d"})

_TOKEN_RE = re.compile(r"%(?:[YymdHMSf])")


def _literal(text: str) -> str:
    """The literal part of a pattern as a regex, with `strptime`'s space rule.

    A run of spaces in the format matches any run of whitespace, so
    ``2010-07-19  19:39:07`` parses under a format holding a single space. Escaping
    the space literally would refuse to align that sample and silently discard every
    width measured for it.
    """
    return re.escape(text).replace(r"\ ", r"\s+").replace(" ", r"\s+")


def _stored_widths(pattern: str, values: list[str]) -> dict[str, int]:
    """The widest digit run each token holds across *values*.

    The pattern is turned into a regex with one bounded group per token, which
    aligns the sample on the format without assuming which character separates two
    components -- a separator can be a space, a dot-and-space, or a Korean ``월``. A
    token missing from the result was absent from a sample the regex could not
    align, and keeps its padded spelling.
    """
    parts: list[str] = []
    tokens: list[str] = []
    index = 0
    for match in _TOKEN_RE.finditer(pattern):
        token = match.group()
        tokens.append(token)
        parts.append(_literal(pattern[index : match.start()]))
        parts.append(rf"({_COMPONENT_RE[token]})")
        index = match.end()
    parts.append(_literal(pattern[index:]))
    aligned = re.compile("".join(parts))

    widths = dict.fromkeys(tokens, 0)
    for value in values:
        matched = aligned.fullmatch(value)
        if matched is None:
            continue
        for token, digits in zip(tokens, matched.groups()):
            widths[token] = max(widths[token], len(digits))
    return {token: width for token, width in widths.items() if width}


def _notation(pattern: str, values: list[str] | None = None) -> str:
    """Render *pattern* as the notation the model has to write.

    ``%f`` spans one to six digits and ``strptime`` accepts an unpadded month or
    day, so one pattern covers several stored widths. Both are restated at the
    width actually stored: an equality predicate has to reproduce the string the
    column holds, and ``04/04/20`` never matches a column storing ``4/4/20``.
    Where the width varies the widest is reported, since that is the form a
    literal has to be able to hold.

    Without *values* every token takes its widest spelling, which is the
    canonical form of the pattern -- padding is not a different reading, so two
    patterns are compared as candidates on that form, not on what a sample
    happens to store.
    """
    widths = _stored_widths(pattern, values) if values else {}
    out: list[str] = []
    index = 0
    for match in _TOKEN_RE.finditer(pattern):
        token = match.group()
        text = _NOTATION[token]
        width = widths.get(token, 0)
        if token == "%f" and width:
            text = text * width
        elif token in _NARROWABLE and width == 1:
            text = text[:1]
        out.append(pattern[index : match.start()])
        out.append(text)
        index = match.end()
    out.append(pattern[index:])
    return "".join(out)


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
        (pattern, _notation(pattern, seen))
        for pattern in _CANDIDATES
        if _parses_all(pattern, seen)
    ]
    if not matches:
        return None
    pattern, winner = matches[0]
    canonical = _notation(pattern)
    readings = {_notation(reading) for reading, _ in matches}
    for group in _INDISTINGUISHABLE:
        if canonical in group and len(group & readings) > 1:
            return None
    return winner
