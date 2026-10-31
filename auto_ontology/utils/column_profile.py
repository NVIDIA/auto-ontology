# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Storage and wording for a column's whole-table profile.

The profile is measured in :mod:`auto_ontology.semantic.column_profile` and
stored as JSON on ``catalog_column.profile``. :func:`profile_facts` is the one
place it is put into words, so the model that writes the column summary at
ingest and the SQL prompt at question time read the same sentences.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

#: Range endpoints longer than this are left out of the facts: a range of two
#: free-text values says nothing a filter can use.
_MAX_RANGE_VALUE_LEN = 30

#: A most-common value is worth naming in the prompt only when it dominates.
_DOMINANT_SHARE = 0.3


def dump_profile(profile: dict[str, Any] | None) -> str | None:
    """*profile* as stored JSON, or None when there is nothing in it."""
    if not profile:
        return None
    return json.dumps(profile, default=str, ensure_ascii=False)


def parse_profile(raw: Any) -> dict[str, Any] | None:
    """A stored profile back as a dict; None when absent or unreadable."""
    if isinstance(raw, dict):
        return raw
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        logger.warning("unreadable column profile ignored: %.80r", raw)
        return None
    return parsed if isinstance(parsed, dict) else None


def _quoted(value: Any) -> str:
    return repr(value) if isinstance(value, str) else str(value)


def _count(n: int) -> str:
    return f"{n:,}"


def _share(part: int, whole: int) -> str:
    share = part / whole
    if share < 0.01:
        return "under 1%"
    if share > 0.99 and part < whole:
        return "over 99%"
    return f"{share:.0%}"


def profile_facts(profile: dict[str, Any] | None, *, top_values: bool = False) -> str:
    """The measured facts of *profile* as one line of plain English.

    ``top_values`` lists the most common values with their counts, which the
    summary model needs and the SQL prompt does not: the prompt already shows
    sample values, so there only a value that fills a large share of the rows
    is named.
    """
    if not profile:
        return ""
    rows = profile.get("row_count")
    nulls = profile.get("null_count")
    distinct = profile.get("distinct_count")
    facts: list[str] = []

    if isinstance(rows, int) and isinstance(nulls, int):
        if nulls == 0:
            facts.append("no NULLs")
        elif rows:
            facts.append(
                f"{_count(nulls)} of {_count(rows)} rows NULL ({_share(nulls, rows)})"
            )
    empties = profile.get("empty_count")
    if isinstance(empties, int) and empties:
        facts.append(f"{_count(empties)} empty strings")

    non_null = (
        rows - nulls if isinstance(rows, int) and isinstance(nulls, int) else None
    )
    if isinstance(distinct, int):
        if distinct == 1:
            facts.append("a single distinct value")
        elif non_null and distinct == non_null:
            facts.append("every value distinct")
        else:
            facts.append(f"{_count(distinct)} distinct values")

    low, high = profile.get("min"), profile.get("max")
    if (
        low is not None
        and high is not None
        and low != high
        and len(str(low)) <= _MAX_RANGE_VALUE_LEN
        and len(str(high)) <= _MAX_RANGE_VALUE_LEN
    ):
        facts.append(f"from {_quoted(low)} to {_quoted(high)}")

    shortest, longest = profile.get("min_length"), profile.get("max_length")
    if isinstance(shortest, int) and shortest == longest and shortest > 0:
        facts.append(f"always {shortest} characters")

    facts.extend(profile.get("shape") or [])

    common = [(v, c) for v, c in profile.get("top_values") or [] if c > 1]
    if top_values and common:
        listed = ", ".join(f"{_quoted(v)} ({_count(c)})" for v, c in common)
        facts.append(f"most common: {listed}")
    elif common and non_null:
        value, count = common[0]
        if count / non_null >= _DOMINANT_SHARE and distinct != 1:
            facts.append(
                f"{_quoted(value)} in {_share(count, non_null)} of non-NULL rows"
            )

    return "; ".join(facts)


def profile_text(profile: dict[str, Any] | None) -> str:
    """What the SQL prompt shows for a column: the summary, then the facts."""
    if not profile:
        return ""
    summary = str(profile.get("summary") or "").strip()
    facts = profile_facts(profile)
    if summary and facts:
        return f"{summary} ({facts})"
    return summary or facts
