# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The values a generated filter is allowed to name.

The PQL prompt tells the model to match a filter to "the column's sampled
values". Nothing supplied them, so the model wrote whichever spelling the
question used: ``status = 'shipped'`` against a column holding ``'S'``. The
filter then matched no row and the prediction answered a question nobody asked.

The catalog already profiles these values at ingestion, so they are read from
there rather than from the warehouse: no query per request, and the same values
the rest of the catalog was built from.

Only where naming a value is both useful and safe. A value is useful when the
column is categorical and small enough to enumerate; naming one of ten statuses
helps, naming one of a million ids does not. A value is safe when it survives
the checks below, because everything here is written verbatim into a prompt.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)

MAX_VALUES_PER_COLUMN = 12
MAX_VALUE_LENGTH = 40
_UNSAFE_CHARACTERS = re.compile(r"[`\n\r\t{}$\\]|```")
_SENSITIVE_SHAPES = (
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
    re.compile(r"\b(?:\d[ -]?){13,19}\b"),
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    re.compile(
        r"(?i)\b(?:sk|pk|api|key|token|secret|bearer|password)[-_ ]?[:=]?\s*\S{8,}"
    ),
    re.compile(r"(?i)^(?:eyJ|ghp_|gho_|xox[baprs]-|AKIA)"),
)
_SENSITIVE_COLUMN_NAMES = re.compile(
    r"(?i)(?:^|_)(?:password|passwd|secret|token|api_?key|ssn|social_security|"
    r"credit_?card|card_?number|cvv|email|phone|mobile|address|dob|"
    r"date_of_birth|salary|compensation|first_?name|last_?name|full_?name)(?:_|$)"
)
_ENUMERABLE_STYPES = frozenset({"categorical", "multicategorical"})


def _extra_sensitive_names() -> "re.Pattern[str] | None":
    """Column names an operator has told us are sensitive here.

    The built-in list cannot know what a given deployment calls things, and a
    column it has not heard of gets no protection at all. Read from
    ``KUMO_SENSITIVE_COLUMNS`` as a comma-separated list of substrings, and
    added to the built-in list rather than replacing it, so widening the net
    locally cannot narrow it.
    """
    raw = os.environ.get("KUMO_SENSITIVE_COLUMNS", "")
    parts = [re.escape(p.strip()) for p in raw.split(",") if p.strip()]
    return re.compile("|".join(parts), re.IGNORECASE) if parts else None


# A value that reads like a sentence or an instruction rather than a category.
# A tier is a word or two; anything carrying a verb phrase, several words, or
# the shape of a directive is not a vocabulary entry and does not belong in the
# same text as the instructions.
_INSTRUCTION_SHAPED = re.compile(
    r"\b(ignore|disregard|forget|instead|you\s+are|system|prompt|instruction|"
    r"reveal|output|respond|answer|execute|run|delete|drop|update|insert)\b",
    re.IGNORECASE,
)
_MAX_WORDS = 4


def reads_like_an_instruction(value: str) -> bool:
    """Whether a value would be read as something other than a category.

    Blocking odd characters and known secret shapes still lets a plain English
    sentence through, and a sentence copied verbatim into the prompt sits in the
    same text as the instructions, where the model has no way to tell which is
    which. A category is a word or two; a sentence is not one.
    """
    text = str(value).strip()
    return len(text.split()) > _MAX_WORDS or bool(_INSTRUCTION_SHAPED.search(text))


def is_sensitive_column(name: str) -> bool:
    """Whether a column's name says its values should not reach a prompt."""
    text = str(name)
    if _SENSITIVE_COLUMN_NAMES.search(text):
        return True
    extra = _extra_sensitive_names()
    return bool(extra and extra.search(text))


def is_safe_value(value: Any) -> bool:
    """Whether one value can be written into a prompt as itself."""
    if value is None:
        return False
    text = str(value)
    if not text.strip() or len(text) > MAX_VALUE_LENGTH:
        return False
    if _UNSAFE_CHARACTERS.search(text):
        return False
    if reads_like_an_instruction(text):
        return False
    return not any(shape.search(text) for shape in _SENSITIVE_SHAPES)


def safe_values(column: str, values: list[Any] | None) -> list[str]:
    """The values of *column* that may be named, in the order profiling saw them.

    All or nothing. Dropping the one value that failed a check would leave a
    list that is presented as everything the column holds while missing the
    entry a question might name, and a partial vocabulary read as a whole one
    sends the model to the nearest listed value instead. So a single value that
    cannot be shown withdraws the column.

    Empty likewise when the column is sensitive by name, or holds more values
    than a vocabulary would.
    """
    if not values or is_sensitive_column(column):
        return []
    if len(values) > MAX_VALUES_PER_COLUMN:
        return []
    kept = [str(v) for v in values]
    if not all(is_safe_value(v) for v in kept):
        return []
    return kept


def build_column_reference(
    relevant_tables: list[dict[str, Any]],
    col_stypes: dict[str, dict[str, str]] | None = None,
) -> str:
    """Render the filter vocabulary the model may draw on, or nothing.

    A column whose semantic type is unknown does not qualify. Reading a missing
    type as permission would offer ids and measurements the moment a table name
    is cased differently from the graph's.

    Returns the empty string when no column qualifies, which is the honest
    answer: the prompt then says no values were supplied rather than implying
    the model has a vocabulary it was never given.
    """
    lines: list[str] = []
    for table in relevant_tables or []:
        if not isinstance(table, dict):
            continue
        name = str(table.get("name") or "").strip()
        if not name:
            continue
        stypes = (col_stypes or {}).get(name.lower(), {})
        entries: list[str] = []
        for column in table.get("columns") or []:
            if not isinstance(column, dict):
                continue
            column_name = str(column.get("name") or "").strip()
            if not column_name:
                continue
            if stypes.get(column_name.lower()) not in _ENUMERABLE_STYPES:
                continue
            values = safe_values(column_name, column.get("sample_values"))
            if values:
                entries.append(f"  {column_name}: {', '.join(repr(v) for v in values)}")
        if entries:
            lines.append(f"{name}")
            lines.extend(entries)
    if not lines:
        return ""
    return (
        "## Filter values\n"
        "Values seen in these columns when the catalog was profiled. Spell a "
        "filter on one of them the way it is spelled here rather than the way "
        "the question does. The list comes from a sample, so a rare value may "
        "be missing: it is how the values look, not proof of every value that "
        "exists. A column absent from this list has no vocabulary supplied, so "
        "do not guess one for it.\n" + "\n".join(lines)
    )
