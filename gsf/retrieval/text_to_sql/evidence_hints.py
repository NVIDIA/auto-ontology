# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Derive short, deterministic SQL hints from BIRD-style evidence text.

Only runs when the question includes an Evidence section. Returns an empty
string when no evidence is present or no known patterns match.
"""

from __future__ import annotations

import re

_EVIDENCE_SPLIT = re.compile(r"\n\nEvidence:\s*", re.IGNORECASE)
_TIME_LITERAL = re.compile(r"\b(0:\d{1,2}:\d{2})\b")
_LIKE_M_SS = re.compile(
    r"(\w+(?:\.\w+)?)\s+LIKE\s+'M:SS%'",
    re.IGNORECASE,
)
_REFERS_TO = re.compile(
    r"refers to\s+(?:\([^)]*\)\s*)?(\w+(?:\.\w+)?)",
    re.IGNORECASE,
)
_REPRESENTED_BY_ISO_DATE = re.compile(
    r"can be represented by\s+'(\d{4}-\d{2}-\d{2})'",
    re.IGNORECASE,
)
_YEARMONTH_TABLE = re.compile(r"yearmonth\s+table", re.IGNORECASE)
_YYYYMM_REF = re.compile(r"refers to\s+'?(\d{6})'?", re.IGNORECASE)
_BETWEEN_YYYYMM = re.compile(
    r"Between\s+(\d{6})\s+And\s+(\d{6})",
    re.IGNORECASE,
)
_TIME_IN_QUESTION = re.compile(r"\b\d{1,2}:\d{2}:\d{2}\b")
_TRANSACTIONS_IN_Q = re.compile(r"\btransaction", re.IGNORECASE)
_SPENT_OR_PAID = re.compile(r"\b(spent|paid|price)\b", re.IGNORECASE)
_SLASH_DATE_IN_Q = re.compile(r"\b20\d{2}/\d{1,2}/\d{1,2}\b")


def _normalize_time_for_like(literal: str) -> str | None:
    """Map question time `0:01:54` to BIRD LIKE prefix `1:54`."""
    match = re.match(r"0:(\d{1,2}):(\d{2})$", literal)
    if not match:
        return None
    return f"{int(match.group(1))}:{match.group(2)}"


def extract_evidence(question: str) -> str | None:
    """Return the Evidence body, or None if the question has no Evidence section."""
    if not question:
        return None
    parts = _EVIDENCE_SPLIT.split(question, maxsplit=1)
    if len(parts) < 2:
        return None
    body = parts[1].strip()
    return body or None


def _question_without_evidence(question: str) -> str:
    parts = _EVIDENCE_SPLIT.split(question, maxsplit=1)
    return parts[0].strip()


def _time_like_hints(evidence: str, question: str) -> list[str]:
    col_match = _LIKE_M_SS.search(evidence)
    if not col_match:
        return []
    column = col_match.group(1)
    hints: list[str] = []
    for literal in _TIME_LITERAL.findall(_question_without_evidence(question)):
        pattern = _normalize_time_for_like(literal)
        if not pattern:
            continue
        hints.append(
            f"- Time `{literal}` → use `{column} LIKE '{pattern}%'` "
            f"(not `= '{literal}'` or `LIKE '{literal}%'`)"
        )
    return hints


def _refers_to_hints(evidence: str, *, skip_columns: set[str]) -> list[str]:
    """Extract column/table targets from 'refers to ...' evidence clauses."""
    hints: list[str] = []
    seen: set[str] = set()
    for match in _REFERS_TO.finditer(evidence):
        target = match.group(1)
        key = target.lower()
        if key in seen or key in skip_columns:
            continue
        seen.add(key)
        hints.append(
            f"- Evidence maps a term to `{target}` — use that exact "
            f"column/table from evidence, not a substitute"
        )
    return hints


def _table_routing_hints(evidence: str, question: str) -> list[str]:
    """Route between per-transaction vs monthly tables from evidence patterns."""
    hints: list[str] = []
    q = _question_without_evidence(question)
    ql = q.lower()
    el = evidence.lower()
    has_iso_day = bool(_REPRESENTED_BY_ISO_DATE.search(evidence))
    has_clock = bool(_TIME_IN_QUESTION.search(q))
    has_yyyymm = bool(_YYYYMM_REF.search(evidence))
    mentions_txn = bool(_TRANSACTIONS_IN_Q.search(q))

    if (
        has_iso_day
        or has_clock
        or (_SLASH_DATE_IN_Q.search(q) and _SPENT_OR_PAID.search(ql))
    ):
        hints.append(
            "- Evidence maps a calendar day to YYYY-MM-DD — filter the "
            "per-transaction table on Date (and Time if the question gives "
            "a clock time), not the monthly YYYYMM table."
        )

    if mentions_txn and has_yyyymm:
        yyyymm = _YYYYMM_REF.search(evidence)
        code = yyyymm.group(1) if yyyymm else "YYYYMM"
        hints.append(
            f"- Transactions in a month: use the per-transaction table "
            f"joined to the monthly table on customer id, with "
            f"monthly.Date = '{code}' from evidence."
        )

    if has_yyyymm and re.search(r"\b(product|consumed|purchase)\b", ql):
        yyyymm = _YYYYMM_REF.search(evidence)
        code = yyyymm.group(1) if yyyymm else "YYYYMM"
        hints.append(
            f"- Products in a month: join product, per-transaction, and "
            f"monthly tables on customer; filter monthly.Date = '{code}'."
        )

    if _YEARMONTH_TABLE.search(evidence) and _SPENT_OR_PAID.search(ql) and has_iso_day:
        hints.append(
            "- Identify the customer from the per-transaction table "
            "(Date + Price from evidence); then aggregate consumption "
            "from the monthly table for that customer."
        )

    if _YEARMONTH_TABLE.search(evidence) and (
        "first 4 string" in el
        or "5th and 6th" in el
        or _BETWEEN_YYYYMM.search(evidence)
    ):
        if not (mentions_txn and has_yyyymm):
            hints.append(
                "- Monthly consumption: Date is YYYYMM. Use SUBSTR(Date,1,4) "
                "for year and SUBSTR(Date,5,2) for month per evidence."
            )

    if re.search(r"\bwhich\s+(year|month)\b", ql):
        hints.append(
            "- 'Which year/month' asks for the period only — SELECT just "
            "the year or month expression, not the aggregated total."
        )

    return hints


def build_evidence_hints_block(question: str) -> str:
    """Build a compact hint block for the SQL prompt, or '' if not applicable."""
    evidence = extract_evidence(question)
    if not evidence:
        return ""

    time_hints = _time_like_hints(evidence, question)
    skip_columns: set[str] = set()
    like_match = _LIKE_M_SS.search(evidence)
    if like_match:
        skip_columns.add(like_match.group(1).lower())
    hints = (
        time_hints
        + _refers_to_hints(evidence, skip_columns=skip_columns)
        + _table_routing_hints(evidence, question)
    )
    if not hints:
        return ""

    return "## Evidence-derived rules\n" + "\n".join(hints)


__all__ = ["extract_evidence", "build_evidence_hints_block"]
