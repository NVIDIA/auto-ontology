# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Does a submitted answer cite numbers its own rows don't hold? (#252)

The model writes ``response`` in the same ``submit`` call as the final SQL,
before that SQL runs, so the prose can describe an earlier exploratory query.
Asked how many ASNs Kestrel Components Ltd shipped, an answer said "1 ASN with
a total of 587 units" beside rows holding 17,999 and 9,550,743; on Wide World
Importers a backorders-by-year answer said 1,872 / 2,174 / 2,488 / 1,004
customers beside rows holding 583 / 612 / 634 / 512.

The check is deterministic and narrow, because a false alarm costs a turn.
It fires only when the rows hold numbers, no number the response cites is one
the rows support, and the response cites none of the rows' own values: the
shape of an answer describing a different query. An answer that cites its
rows and adds context from exploration passes.

A cited number is supported when, rounded to the precision it is written with
or within about 1%, it equals a value in the rows (a numeric cell, a number
inside a text cell, the row count, a column's sum, or for a short result a
ratio of two cells in a row or a cell's share of its column). Thousands separators,
currency signs, ``K``/``M``/``B``, thousand/million/billion, and a percentage
of a ratio are understood. Not checked: years, dates and times, numbers the
question states, numbers in the SQL (IDs, thresholds, ``LIMIT``), numbers in
code spans (quoted SQL), numbers inside identifiers such as ``WWI-321-A`` or
``#587``, ordinals, "top N", and small integers such as ranks or the count of
items listed.

:func:`numbers_check` is the ``submit`` hook. It sends an answer back at most
once per episode, with the unsupported numbers and the rows, so the model
restates the answer from them.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from auto_ontology.retrieval.text_to_sql.agents.sql_execution import (
        QueryResponse,
    )
    from auto_ontology.retrieval.text_to_sql.react.context import ToolContext

_SMALL = 12
_RELATIVE = 0.011
_ABSOLUTE = 0.006
_SHOWN_ROWS = 20
_RATIO_ROWS = 50
_RATIO_COLUMNS = 10
# Past this many rows the check is skipped rather than slowing the answer.
_MAX_ROWS = 10_000
_SHOWN_CELL_CHARS = 80

_SCALES = {
    "k": 1e3,
    "thousand": 1e3,
    "m": 1e6,
    "million": 1e6,
    "b": 1e9,
    "bn": 1e9,
    "billion": 1e9,
    "trillion": 1e12,
}
_NUMBER = re.compile(
    r"(?<![\w.\-/#])(?<!\d:)"
    r"(?P<sign>-?)[$€£]?(?P<digits>\d[\d,]*(?:\.\d+)?)"
    r"(?:(?P<percent>\s?%|\s(?:percent|pct)\b)"
    r"|(?P<scale>\s?(?:[kmb]|bn)\b|\s(?:thousand|million|billion|trillion)\b))?"
    r"(?![\w\-/%]|[.:]\d)",
    re.IGNORECASE,
)
_MONTH = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|"
    r"aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
)
# "29 November", "November 29", "March 31," name a day, not a measure.
_DAY_OF_MONTH = re.compile(
    rf"(?:\b\d{{1,2}}(?:st|nd|rd|th)?\s+{_MONTH}\b"
    rf"|\b{_MONTH}\.?\s+\d{{1,2}}(?:st|nd|rd|th)?\b)",
    re.IGNORECASE,
)
# Quoted SQL and identifiers: fenced blocks and inline code spans.
_CODE = re.compile(r"```.*?```|`[^`\n]*`", re.DOTALL)
_RANK_WORD = re.compile(r"\b(?:top|bottom|first|last|best|worst)\s+$", re.IGNORECASE)
_SQL_NUMBER = re.compile(r"(?<![A-Za-z_])\d+(?:\.\d+)?")


def enabled() -> bool:
    """On by default; ``TEXT_TO_SQL_REACT_ANSWER_NUMBERS_CHECK=false`` turns it off."""
    value = os.environ.get("TEXT_TO_SQL_REACT_ANSWER_NUMBERS_CHECK", "")
    return value.strip().lower() not in {"0", "false", "no", "off"}


@dataclass(frozen=True)
class _Cited:
    """One number as written: its value, and how precisely it was written."""

    text: str
    # (value, half a unit of its last written digit) per reading.
    readings: tuple[tuple[float, float], ...]
    percent: bool
    bare_integer: bool
    year: bool
    after_rank_word: bool


def _numbers(text: str) -> list[_Cited]:
    found = []
    for match in _NUMBER.finditer(text):
        digits = match.group("digits").rstrip(",")
        try:
            number = float(digits.replace(",", ""))
        except ValueError:
            continue
        if match.group("sign"):
            number = -number
        decimals = len(digits.split(".")[1]) if "." in digits else 0
        raw_suffix = match.group("scale") or ""
        suffix = raw_suffix.strip()
        scale = _SCALES.get(suffix.lower(), 1.0)
        # "300m" may be metres and "3 B" a grade: either reading supports the
        # number, and a small one is not checked.
        ambiguous = suffix == "m" or (
            raw_suffix[:1].isspace() and suffix.lower() in {"k", "m", "b", "bn"}
        )
        percent = bool(match.group("percent"))
        bare = not decimals and not percent and (scale == 1.0 or ambiguous)
        half_unit = 0.5 * 10.0**-decimals
        readings = [(number * scale, half_unit * scale)]
        if ambiguous:
            readings.append((number, half_unit))
        found.append(
            _Cited(
                text=match.group(0).strip().rstrip(","),
                readings=tuple(readings),
                percent=percent,
                bare_integer=bare,
                year=bare
                and re.fullmatch(r"\d{4}", digits) is not None
                and 1900 <= number <= 2100,
                after_rank_word=_RANK_WORD.search(text[: match.start()]) is not None,
            )
        )
    return found


def _prose(response: str) -> str:
    """The response without code spans and days of the month."""
    return _DAY_OF_MONTH.sub(" ", _CODE.sub(" ", response or ""))


def rows_from_result(result: list[Any] | None) -> list[dict[str, Any]]:
    """The rows of a ``QueryResponse.result``: JSON record strings or dicts."""
    rows: list[dict[str, Any]] = []
    for item in result or []:
        if isinstance(item, str):
            try:
                item = json.loads(item)
            except json.JSONDecodeError:
                continue
        if isinstance(item, dict):
            rows.append(item)
        elif isinstance(item, list):
            rows.extend(r for r in item if isinstance(r, dict))
    return rows


def _cell_number(cell: Any) -> float | None:
    """A numeric cell, or a text cell that is one number (a serialized DECIMAL)."""
    if isinstance(cell, bool) or cell is None:
        return None
    if isinstance(cell, (int, float)):
        return float(cell)
    if isinstance(cell, str):
        try:
            return float(cell.strip().replace(",", ""))
        except ValueError:
            return None
    return None


def _cell_values(rows: list[dict[str, Any]]) -> list[float]:
    """Every number the rows show: numeric cells and numbers inside text."""
    values: list[float] = []
    for row in rows:
        for cell in row.values():
            number = _cell_number(cell)
            if number is not None:
                values.append(number)
            elif isinstance(cell, str):
                values.extend(c.readings[0][0] for c in _numbers(cell))
    return values


def _derived_values(rows: list[dict[str, Any]]) -> list[float]:
    """What follows simply from the rows: the row count and each column's sum.

    For a short result also the ratios a response states as percentages: one
    cell over another in the same row, and a cell's share of its column.
    """
    numeric = [
        {c: n for c, v in row.items() if (n := _cell_number(v)) is not None}
        for row in rows
    ]
    sums: dict[str, float] = {}
    for row in numeric:
        for column, number in row.items():
            sums[column] = sums.get(column, 0.0) + number
    derived = [float(len(rows)), *sums.values()]
    if len(rows) <= _RATIO_ROWS:
        for row in numeric:
            cells = list(row.items())[:_RATIO_COLUMNS]
            derived.extend(a / b for _, a in cells for _, b in cells if a and b)
            derived.extend(n / sums[c] for c, n in cells if sums[c])
    return derived


def _supported(cited: _Cited, values: list[float]) -> bool:
    for value in values:
        targets = (value, value * 100) if cited.percent else (value,)
        for target in targets:
            for number, half_unit in cited.readings:
                allowed = max(_RELATIVE * abs(target), half_unit, _ABSOLUTE)
                if abs(number - target) <= allowed:
                    return True
    return False


def _stated(text: str) -> set[float]:
    return {c.readings[0][0] for c in _numbers(text or "")}


def _checked(response: str, question: str, sql: str) -> list[_Cited]:
    """The cited numbers worth checking against the rows."""
    stated = _stated(question) | {float(n) for n in _SQL_NUMBER.findall(sql or "")}
    return [
        c
        for c in _numbers(_prose(response))
        if not c.year
        and not c.after_rank_word
        and not (c.bare_integer and any(abs(n) <= _SMALL for n, _ in c.readings))
        and not any(n in stated for n, _ in c.readings)
    ]


def contradicted_numbers(
    response: str,
    rows: list[dict[str, Any]],
    question: str = "",
    sql: str = "",
) -> list[str]:
    """The numbers *response* cites, when none of them come from *rows*.

    Empty when the rows hold no numbers (an empty or all-NULL result included),
    when some checked number is supported, or when the response cites any of
    the rows' own values, even a small one such as 0. Otherwise the cited text
    of each checked number, in order.
    """
    cells = _cell_values(rows)
    if not cells:
        return []
    checked = _checked(response, question, sql)
    if not checked:
        return []
    support = cells + _derived_values(rows)
    if any(_supported(c, support) for c in checked):
        return []
    if any(_supported(c, cells) for c in _numbers(_prose(response)) if not c.year):
        return []
    return [c.text for c in checked]


def render_rows(rows: list[dict[str, Any]], limit: int = _SHOWN_ROWS) -> str:
    """The first *limit* rows as pipe-separated text, with the total count."""
    if not rows:
        return "0 rows."
    columns = list(rows[0])

    def cell(value: Any) -> str:
        text = "NULL" if value is None else str(value)
        return text if len(text) <= _SHOWN_CELL_CHARS else text[:79] + "…"

    lines = [" | ".join(columns)] + [
        " | ".join(cell(row.get(c)) for c in columns) for row in rows[:limit]
    ]
    shown = f", first {limit} shown" if len(rows) > limit else ""
    return f"{len(rows)} row{'s' if len(rows) != 1 else ''}{shown}:\n" + "\n".join(
        lines
    )


def numbers_check(
    ctx: ToolContext, sql: str, response: str, db_response: QueryResponse
) -> str | None:
    """Why ``submit`` sends this answer back, or None to accept it.

    Sends back at most once per episode, and never when the model could not
    answer again: on its last answer attempt, or when the next query, which
    resubmitting runs, would be refused.
    """
    if ctx.answer_numbers_checked or ctx.last_answer_attempt or not enabled():
        return None
    if db_response.sliced or not response:
        return None
    if ctx.can_run_queries is not None and not ctx.can_run_queries():
        return None
    rows = rows_from_result(db_response.result)
    if len(rows) > _MAX_ROWS:
        return None
    contradicted = contradicted_numbers(response, rows, ctx.question, sql)
    if not contradicted:
        return None
    ctx.answer_numbers_checked = True
    return (
        "Not accepted yet. The response was written before this query ran, and "
        f"it cites {', '.join(contradicted)}, which the query's rows neither "
        "contain nor add up to. Restate the response from the rows below and "
        "send this same query again; change the query only if these rows do "
        f"not answer the question.\n\nQuery:\n{sql}\n\nRows: " + render_rows(rows)
    )
