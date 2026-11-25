# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provenance of saved custom analyses in a ReAct SQL answer.

``custom_analyses_used`` lists every saved analysis the episode's ``retrieve``
showed the model, each with the furthest stage that can be established:

``retrieved``
    The definition was put in front of the model. Nothing shows it was used.
``reference``
    Retrieved, and the submitted SQL reads a table the analysis reads and also
    repeats at least one of its predicates or aggregates (column names compared
    without their ``table__`` semantic prefix). Table overlap alone is not
    enough. This is still a heuristic, not proof the model followed the
    definition; the submitted SQL is generated, not the saved analysis.
``executed``
    The submitted SQL, which is what ran, is the analysis's SQL, or contains it
    whole as a CTE or subquery: the same statement once parsed and normalised,
    so whitespace, keyword case and a trailing semicolon do not matter. A
    repaired or reinterpreted query is never reported as executed. Saved SQL
    that uses semantic names cannot run as it is, so it tops out at
    ``reference``.

Each entry carries the analysis ``id`` and a ``fingerprint`` (SHA-256 of the
normalised SQL) that identifies the exact definition the stage refers to. An
empty list means no analysis was retrieved in this episode.
"""

from __future__ import annotations

import hashlib

import sqlglot
from sqlglot import exp

from auto_ontology.retrieval.text_to_sql.react.context import ToolContext

RETRIEVED = "retrieved"
REFERENCE = "reference"
EXECUTED = "executed"


def _normalise(sql: str | None, dialect: str | None) -> str | None:
    """*sql* as one canonical statement, or ``None`` if it is not exactly one."""
    if not sql or not sql.strip():
        return None
    try:
        statements = sqlglot.parse(sql, read=dialect)
    except sqlglot.errors.SqlglotError:
        return None
    if len(statements) != 1 or statements[0] is None:
        return None
    return statements[0].sql(dialect=dialect, normalize=True)


def _parse(sql: str | None, dialect: str | None) -> list[exp.Expression]:
    if not sql:
        return []
    try:
        return [e for e in sqlglot.parse(sql, read=dialect) if e is not None]
    except sqlglot.errors.SqlglotError:
        return []


def _tables(sql: str | None, dialect: str | None) -> set[str]:
    """Physical tables read, leaving out names that are only CTE aliases."""
    names: set[str] = set()
    for statement in _parse(sql, dialect):
        ctes = {c.alias_or_name.lower() for c in statement.find_all(exp.CTE)}
        names |= {
            t.name.lower()
            for t in statement.find_all(exp.Table)
            if t.name and t.name.lower() not in ctes
        }
    return names


def _bare_columns(node: exp.Expression) -> exp.Expression:
    """*node* with ``table__column`` semantic names reduced to ``column``."""
    node = node.copy()
    for column in node.find_all(exp.Column):
        name = column.name
        if "__" in name:
            column.set("this", exp.to_identifier(name.rsplit("__", 1)[1]))
        column.set("table", None)
    return node


def _fragments(sql: str | None, dialect: str | None) -> set[str]:
    """The comparisons and aggregates that make up the analysis's definition."""
    out: set[str] = set()
    for statement in _parse(sql, dialect):
        for node in statement.find_all(exp.Predicate, exp.AggFunc):
            out.add(_bare_columns(node).sql(dialect=dialect, normalize=True))
    return out


def _runs_whole(
    normalised: str | None, submitted_sql: str | None, dialect: str | None
) -> bool:
    """Whether the submission is, or embeds as a CTE or subquery, the saved SQL."""
    if normalised is None:
        return False
    statements = _parse(submitted_sql, dialect)
    if len(statements) != 1:
        return False
    return any(
        select.sql(dialect=dialect, normalize=True) == normalised
        for select in statements[0].find_all(exp.Select, exp.SetOperation)
    )


def _fingerprint(sql: str, normalised: str | None) -> str:
    return hashlib.sha256((normalised or " ".join(sql.split())).encode()).hexdigest()


def custom_analyses_used(
    ctx: ToolContext, submitted_sql: str | None, dialect: str | None = None
) -> list[dict[str, str]]:
    """Provenance entries for every analysis shown in this episode.

    *submitted_sql* is the query that ran, or ``None`` when no SQL was
    submitted, in which case nothing can be past ``retrieved``.
    """
    submitted_tables = _tables(submitted_sql, dialect)
    submitted_fragments = _fragments(submitted_sql, dialect)
    entries: list[dict[str, str]] = []
    for analysis in ctx.shown_analyses.values():
        sql = analysis.get("sql") or ""
        normalised = _normalise(sql, dialect)
        stage = RETRIEVED
        if _runs_whole(normalised, submitted_sql, dialect):
            stage = EXECUTED
        elif submitted_tables & _tables(sql, dialect) and (
            submitted_fragments & _fragments(sql, dialect)
        ):
            stage = REFERENCE
        entries.append(
            {
                "id": analysis["id"],
                "name": analysis.get("name", ""),
                "fingerprint": _fingerprint(sql, normalised),
                "stage": stage,
            }
        )
    return entries
