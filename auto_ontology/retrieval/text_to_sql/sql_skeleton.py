# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structural skeleton of a SQL statement, and similarity between skeletons.

A skeleton keeps the query's shape and discards everything specific to one
database: identifiers become ``C`` (column), ``T`` (table), literals ``L`` and
numbers ``N``, while keywords and punctuation survive. So

    SELECT MAX(`Free Meal Count`) FROM frpm WHERE `County` = 'Alameda'

reduces to ``select max ( C ) from T where C = L``.

That makes two queries comparable across schemas, which is what lets a
candidate be scored against reference SQL written for a different database.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

# Words that carry structure and so survive into the skeleton. Anything else
# that looks like an identifier is rewritten as a column or table marker.
SQL_KEYWORDS = {
    "select",
    "distinct",
    "from",
    "where",
    "group",
    "by",
    "order",
    "having",
    "limit",
    "join",
    "inner",
    "left",
    "outer",
    "on",
    "and",
    "or",
    "not",
    "in",
    "is",
    "null",
    "like",
    "between",
    "as",
    "asc",
    "desc",
    "case",
    "when",
    "then",
    "else",
    "end",
    "cast",
    "real",
    "integer",
    "union",
    "all",
    "exists",
    "count",
    "sum",
    "avg",
    "max",
    "min",
    "iif",
    "strftime",
    "substr",
    "round",
    "julianday",
    "date",
    "datetime",
    "length",
    "abs",
    "coalesce",
    "div",
}

_ALIAS_PREFIX = re.compile(r"\b[Tt]\d+\s*\.|\b[a-z]{1,3}\d?\s*\.", re.ASCII)
_AS_ALIAS = re.compile(r"\bas\s+[`\"\[]?\w+[`\"\]]?", re.IGNORECASE)
_STRINGS = re.compile(r"'[^']*'")
_QUOTED = re.compile(r"[`\"\[][^`\"\]]+[`\"\]]")
_NUMBERS = re.compile(r"\b\d+(\.\d+)?\b")
_WORD = re.compile(r"[A-Za-z_]\w*")
_TOKEN = re.compile(r"[A-Za-z_]\w*|[(),*=<>!+/-]|N|L|C")


def skeleton(sql: object) -> str:
    """The structural skeleton of *sql* as a space-separated token string."""
    text = " ".join(str(sql or "").split())
    text = _STRINGS.sub(" L ", text)
    text = _QUOTED.sub(" C ", text)
    text = _NUMBERS.sub(" N ", text)
    # Aliases carry no structure; drop the declaration and the qualifier.
    text = _AS_ALIAS.sub(" ", text)
    text = _ALIAS_PREFIX.sub(" ", text)

    out: list[str] = []
    prev = ""
    for token in _TOKEN.findall(text):
        low = token.lower()
        if low in SQL_KEYWORDS:
            out.append(low)
        elif token in {"N", "L", "C"}:
            out.append(token)
        elif _WORD.fullmatch(token):
            # A bare word right after FROM or JOIN names a table; anything
            # else in identifier position is a column.
            out.append("T" if prev in {"from", "join"} else "C")
        else:
            out.append(token)
        if _WORD.fullmatch(token):
            prev = low
        elif token not in {"(", ")"}:
            prev = ""
    return " ".join(out)


def skeleton_similarity(left: str, right: str) -> float:
    """Token-level similarity of two skeletons, in ``[0, 1]``.

    Compared as token sequences rather than characters so that renaming one
    identifier cannot outweigh a difference in query structure.
    """
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left.split(), right.split()).ratio()
