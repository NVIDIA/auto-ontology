# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare what a value anchor names against the schema in scope.

Separate value anchors that name schema from the ones that name data.

Anchors are produced by matching phrases from the question against stored
values, with no knowledge of what the schema calls things. A phrase that is
part of a *column name* therefore still comes back as an anchor whenever some
unrelated column happens to hold the same text as data — a qualifier a schema
carries in a column name is exactly the kind of word another column stores as
a category. The anchor section then presents that unrelated column to SQL
generation as a verified match for a phrase that was never a filter value.

Telling the two apart needs the schema, which is why this runs after the
tables in scope are known rather than where the anchors are built. A phrase
that occurs inside a relation or column name is naming schema; the question is
asking *about* that thing, not filtering *on* it.

Matching is on whole word tokens, so a phrase is found inside a longer name
only at word boundaries, never inside a longer word. A phrase of digits alone
is never dropped: a bare number is a real filter value often enough, and
column names routinely carry numbers of their own (units, ranges, bracket
labels) that would swallow it, whereas the phrases this is meant to catch
always carry a word.

The same comparison read the other way finds tables. An anchor names the
table its value was matched in, and a table outside the scope retrieval built
cannot be referenced at all — so an anchor can report where a filter value
lives while the table holding it stays unreachable, which is a question lost
for want of a table already identified. Those tables belong in the candidate
pool the relevance filter judges, not in the answer: most of them are not
wanted, and the filter is what separates them.
"""

from __future__ import annotations

import re

_WORD = re.compile(r"[a-z0-9]+")
_LETTER = re.compile(r"[a-z]")


def _tokenized(text: str) -> str:
    """*text* as lowercase word tokens, space-padded for boundary matching.

    Punctuation becomes a separator on both sides of the comparison, so a
    phrase a question hyphenates reads as the same tokens a name spells with
    a bracket, an underscore, or a space. The padding is what keeps a short
    phrase from matching inside a longer word.
    """
    return " " + " ".join(_WORD.findall(text.lower())) + " "


def _schema_phrases(relevant_tables: list[dict]) -> list[str]:
    """Tokenized names of every table in scope and of their columns."""
    names: list[str] = []
    for table in relevant_tables or []:
        name = table.get("name")
        if name:
            names.append(_tokenized(str(name)))
        for column in table.get("columns") or []:
            column_name = column.get("name") if isinstance(column, dict) else column
            if column_name:
                names.append(_tokenized(str(column_name)))
    return names


def split_schema_named_anchors(
    value_anchors: list[dict] | None,
    relevant_tables: list[dict] | None,
) -> tuple[list[dict], list[dict]]:
    """Split *value_anchors* into ``(kept, naming_schema)``.

    An anchor lands in ``naming_schema`` when its phrase appears as whole
    words inside the name of a table in scope or one of their columns. With
    no tables in scope there is nothing to judge against, so everything is
    kept — a missing anchor costs more than a redundant one.
    """
    anchors = list(value_anchors or [])
    schema_phrases = _schema_phrases(relevant_tables or [])
    if not anchors or not schema_phrases:
        return anchors, []

    kept: list[dict] = []
    naming_schema: list[dict] = []
    for anchor in anchors:
        phrase = _tokenized(str(anchor.get("phrase") or ""))
        if _LETTER.search(phrase) and any(phrase in name for name in schema_phrases):
            naming_schema.append(anchor)
        else:
            kept.append(anchor)
    return kept, naming_schema


def anchor_tables_missing_from_scope(
    value_anchors: list[dict] | None,
    relevant_tables: list[dict] | None,
) -> list[str]:
    """Names of the tables anchors point at that no table in scope carries.

    Anchors naming schema are passed over: their column was never a match for
    the phrase, so the table holding it is no more likely to be wanted than
    any other. On this benchmark those reach a gold table a quarter as often
    as the anchors kept, which is the difference between a candidate worth
    offering and noise.

    Order follows the anchors, so the strongest match is proposed first, and a
    table named by several anchors is proposed once.
    """
    anchors, in_scope = list(value_anchors or []), list(relevant_tables or [])
    kept, _ = split_schema_named_anchors(anchors, in_scope)
    known = {
        str(table.get("name") or "").lower() for table in in_scope if table.get("name")
    }

    missing: dict[str, str] = {}
    for anchor in kept:
        name = str(anchor.get("tbl") or "").strip()
        if name and name.lower() not in known:
            missing.setdefault(name.lower(), name)
    return list(missing.values())


__all__ = ["anchor_tables_missing_from_scope", "split_schema_named_anchors"]
