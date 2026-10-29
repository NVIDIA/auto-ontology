# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Close "which stored value means X" gaps from the data instead of the user.

The completeness check keeps a gap open while the exact value that identifies a
term (a status, a category label) is unspecified. When that term already
resolves to a column holding only a handful of distinct values, the stored
value is a fact the database can state, so asking the user for it spends a
clarification turn on something they can only answer with "whatever the data
uses" — and while that gap stays first in the ranked list it also keeps later
gaps from being asked at all.

This resolves such a gap only when the evidence is unambiguous: the column has
at most ``DB_PROBE_LOW_CARD_THRESHOLD`` distinct values, and exactly one of
them has every one of its words present in the gap's own term. A gap that
cannot be closed that way is returned untouched and still reaches the user.

Opt-in via ``DB_PROBE_VALUE_GAP_LOOKUP`` — the same convention as the other
live probes, since it issues a ``SELECT DISTINCT`` against the user's database.
"""

from __future__ import annotations

import logging
import os
import re
from typing import TYPE_CHECKING, Any, Callable, Optional

from sqlglot import exp

from auto_ontology.retrieval.text_to_sql.db_probe.config import (
    _TRUTHY,
    DB_PROBE_LOW_CARD_THRESHOLD,
)
from auto_ontology.retrieval.text_to_sql.db_probe.executor import ProbeExecutor

if TYPE_CHECKING:
    from .state import InteractiveSessionState

logger = logging.getLogger(__name__)

_WORD = re.compile(r"[a-z0-9]+")
_LETTERS = re.compile(r"[a-z]")

# The gap text names a missing *value*, not a missing formula or operator. The
# completeness prompt phrases these as "the exact value/operator that identifies
# a 'pending' status", "the specific label for pancreas", "the literal ...".
_VALUE_GAP = re.compile(r"\b(value|label|literal|constant)s?\b", re.IGNORECASE)

# Words that carry no identifying power on their own inside a gap term.
_MIN_WORD_LEN = 3


def is_value_gap_lookup_enabled() -> bool:
    """Whether data lookup of value gaps is on (opt-in, off by default)."""
    return os.environ.get("DB_PROBE_VALUE_GAP_LOOKUP", "").strip().lower() in _TRUTHY


def is_value_gap(missing: str) -> bool:
    """Whether a gap's "what is missing" text asks for a stored value."""
    return bool(_VALUE_GAP.search(missing or ""))


def _words(text: str) -> list[str]:
    return _WORD.findall((text or "").lower())


def match_value(values: list[str], term: str) -> Optional[str]:
    """The single stored value whose every word appears in *term*, or ``None``.

    A value qualifies when it carries at least one real word (letters, at least
    three characters), so a bare code such as ``1`` or ``A`` can never be
    matched by coincidence, and every one of its words is present in the term.
    More than one qualifying value is ambiguous, with one exception: a value
    that strictly contains another's words (``In Progress`` over ``Progress``)
    wins, because the longer one is the more specific match.
    """
    term_words = set(_words(term))
    if not term_words:
        return None
    hits: list[tuple[str, frozenset[str]]] = []
    for value in values:
        value_words = _words(str(value))
        if not value_words:
            continue
        if not any(len(w) >= _MIN_WORD_LEN and _LETTERS.search(w) for w in value_words):
            continue
        if all(w in term_words for w in value_words):
            hits.append((str(value), frozenset(value_words)))
    if not hits:
        return None
    if len(hits) == 1:
        return hits[0][0]
    longest = max(hits, key=lambda h: len(h[1]))
    others = [h for h in hits if h is not longest]
    if all(h[1] < longest[1] for h in others):
        return longest[0]
    return None


def _term_tokens(text: str) -> set[str]:
    return {w for w in _words(text) if len(w) >= _MIN_WORD_LEN}


def _hits_for_term(term: str, hits: list[tuple]) -> list[str]:
    """Attribute ids of resolved hits whose entity is the gap's term."""
    wanted = _term_tokens(term)
    ids: list[str] = []
    for norm, _text, _score, hit_id in hits:
        if not hit_id:
            continue
        have = _term_tokens(str(norm))
        if wanted and have and (wanted <= have or have <= wanted):
            ids.append(str(hit_id))
    return list(dict.fromkeys(ids))


def _distinct_values_sql(
    ctx: dict[str, Any], dialect: Optional[str], limit: int
) -> Optional[str]:
    table = str(ctx.get("table_name") or "")
    column = str(ctx.get("col_name") or "")
    if not table or not column:
        return None
    schema = str(ctx.get("schema_name") or "")
    table_node = exp.Table(
        this=exp.to_identifier(table, quoted=True),
        db=exp.to_identifier(schema, quoted=True) if schema else None,
    )
    col_node = exp.column(
        exp.to_identifier(column, quoted=True),
        table=exp.to_identifier(table, quoted=True),
    )
    col_sql = col_node.sql(dialect=dialect)
    return (
        f"SELECT DISTINCT {col_sql} AS value FROM {table_node.sql(dialect=dialect)} "
        f"WHERE {col_sql} IS NOT NULL LIMIT {limit + 1}"
    )


def _fetch_values(
    executor: ProbeExecutor, ctx: dict[str, Any], dialect: Optional[str]
) -> Optional[list[str]]:
    """Distinct values of the column, or ``None`` if too many or the probe fails."""
    sql = _distinct_values_sql(ctx, dialect, DB_PROBE_LOW_CARD_THRESHOLD)
    if sql is None:
        return None
    result = executor.run(sql, purpose="value_gap_distinct")
    if not result["ok"] or not result["rows"]:
        return None
    values = [
        next(iter(row.values()))
        for row in result["rows"]
        if isinstance(row, dict) and row
    ]
    if len(values) > DB_PROBE_LOW_CARD_THRESHOLD:
        return None
    return [str(v) for v in values if v is not None]


def _pick_connector(session: "InteractiveSessionState") -> Any:
    connectors = list(session.connectors or [])
    for connector in connectors:
        if getattr(connector, "database_name", None) == session.db_name:
            return connector
    return connectors[0] if len(connectors) == 1 else None


def resolve_value_gaps(
    session: "InteractiveSessionState",
    gaps: list[tuple[str, str]],
    *,
    fetch_contexts: Optional[Callable[..., dict[str, dict]]] = None,
) -> tuple[list[tuple[str, str]], list[str]]:
    """Split *gaps* into ``(still_open, notes)``.

    ``notes`` are evidence lines for the gaps that were closed from the data;
    ``still_open`` keeps every other gap in its original order. Never raises:
    any failure leaves the gaps exactly as they were.
    """
    if not gaps or not is_value_gap_lookup_enabled():
        return gaps, []
    hits = list(session._cached_resolved_hits or [])
    candidates = [(t, m) for t, m in gaps if is_value_gap(m)]
    if not hits or not candidates:
        return gaps, []
    connector = _pick_connector(session)
    if connector is None:
        return gaps, []

    try:
        if fetch_contexts is None:
            from auto_ontology.dal.attributes import fetch_attr_column_contexts

            fetch_contexts = fetch_attr_column_contexts
        all_ids = list(dict.fromkeys(str(h[3]) for h in hits if len(h) > 3 and h[3]))
        contexts = fetch_contexts(all_ids, database_name=session.db_name) or {}
    except Exception:  # noqa: BLE001 — decoration must never break the turn
        logger.exception("[%s] value-gap lookup: context fetch failed", session.task_id)
        return gaps, []

    dialect = getattr(connector, "dialect", None)
    executor = ProbeExecutor(connector)
    cache: dict[tuple[str, str, str], Optional[list[str]]] = {}
    resolved_terms: set[str] = set()
    notes: list[str] = []

    for term, missing in candidates:
        matches: list[tuple[dict, str]] = []
        for attr_id in _hits_for_term(term, hits):
            ctx = contexts.get(attr_id)
            if not ctx:
                continue
            key = (
                str(ctx.get("schema_name") or ""),
                str(ctx.get("table_name") or ""),
                str(ctx.get("col_name") or ""),
            )
            if key not in cache:
                try:
                    cache[key] = _fetch_values(executor, ctx, dialect)
                except Exception:  # noqa: BLE001
                    logger.exception("[%s] value-gap lookup failed", session.task_id)
                    cache[key] = None
            values = cache[key]
            value = match_value(values, term) if values else None
            if value is not None:
                matches.append((ctx, value))
        distinct = {(c.get("table_name"), c.get("col_name"), v) for c, v in matches}
        if len(distinct) != 1:
            continue  # nothing matched, or several columns disagree: leave it open
        ctx, value = matches[0]
        qualified = ".".join(
            part for part in (ctx.get("table_name"), ctx.get("col_name")) if part
        )
        notes.append(
            f'Stored value for "{term}": {qualified} = {value!r} '
            "(the only value in that column matching the term, read from the "
            "database; use it as an exact-match filter)."
        )
        resolved_terms.add(term)
        logger.info(
            "[%s] Value-gap lookup closed %r -> %s = %r",
            session.task_id,
            term,
            qualified,
            value,
        )

    still_open = [(t, m) for t, m in gaps if t not in resolved_terms]
    return still_open, notes


def record_value_notes(session: "InteractiveSessionState", notes: list[str]) -> None:
    """Keep *notes* on the session without repeats, for injection into evidence."""
    for note in notes:
        if note not in session._value_resolution_notes:
            session._value_resolution_notes.append(note)


__all__ = [
    "is_value_gap",
    "is_value_gap_lookup_enabled",
    "match_value",
    "record_value_notes",
    "resolve_value_gaps",
]
