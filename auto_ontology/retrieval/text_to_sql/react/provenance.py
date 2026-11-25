# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provenance of saved custom analyses in a ReAct SQL answer.

``custom_analyses_used`` lists every saved analysis the episode's ``retrieve``
showed the model, each with the furthest stage that can be established:

``retrieved``
    The definition was put in front of the model. Nothing shows it was used.
``reference``
    Retrieved, the submitted SQL reads a table the analysis reads, and it
    keeps every WHERE/HAVING condition of the analysis (column names compared
    without their ``table__`` semantic prefix). Aggregates, joins and table
    overlap do not count, and a changed or opposite filter leaves the analysis
    at ``retrieved``; so does an analysis with no filter, since nothing can
    then be compared. This is still a heuristic, not proof the model followed
    the definition; the submitted SQL is generated, not the saved analysis.
``executed``
    The submitted SQL, which is what ran, is the analysis's SQL, or contains it
    whole as a subquery or as a CTE the outer query reads (an unreferenced CTE
    never ran): the same statement once parsed and normalised,
    so whitespace, keyword case and a trailing semicolon do not matter. A
    repaired or reinterpreted query is never reported as executed. Saved SQL
    that uses semantic names cannot run as it is, so it tops out at
    ``reference``.

Each entry carries the analysis ``id`` and a ``fingerprint`` (SHA-256 of the
normalised SQL) that identifies the exact definition the stage refers to. An
empty list means no analysis was retrieved in this episode.

Every ReAct outcome carries the list: a submitted query, an information
answer, a prediction and an unanswered episode (the last two cannot go past
``retrieved``). Only the standalone KumoRFM predictor, which never retrieves
saved analyses, returns ``[]`` unconditionally, and there it means "not
applicable". The stage is derived from the submitted SQL text, not from an
execution record.
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


def _conjuncts(condition: exp.Expression) -> list[exp.Expression]:
    if isinstance(condition, exp.Paren):
        return _conjuncts(condition.this)
    if isinstance(condition, exp.And):
        return _conjuncts(condition.left) + _conjuncts(condition.right)
    return [condition]


def _filters(sql: str | None, dialect: str | None) -> set[str]:
    """The WHERE/HAVING conditions of *sql*, one entry per AND-ed condition."""
    out: set[str] = set()
    for statement in _parse(sql, dialect):
        for clause in statement.find_all(exp.Where, exp.Having):
            for part in _conjuncts(clause.this):
                out.add(_bare_columns(part).sql(dialect=dialect, normalize=True))
    return out


def _referenced(select: exp.Expression, root: exp.Expression) -> bool:
    """Whether every CTE enclosing *select* is read somewhere in *root*."""
    read = {t.name.lower() for t in root.find_all(exp.Table)}
    node = select.parent
    while node is not None:
        if isinstance(node, exp.CTE) and node.alias_or_name.lower() not in read:
            return False
        node = node.parent
    return True


def _runs_whole(
    normalised: str | None, submitted_sql: str | None, dialect: str | None
) -> bool:
    """Whether the submission is, or embeds as a read CTE or subquery, the saved SQL."""
    if normalised is None:
        return False
    statements = _parse(submitted_sql, dialect)
    if len(statements) != 1:
        return False
    root = statements[0]
    return any(
        select.sql(dialect=dialect, normalize=True) == normalised
        and _referenced(select, root)
        for select in root.find_all(exp.Select, exp.SetOperation)
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
    submitted_filters = _filters(submitted_sql, dialect)
    entries: list[dict[str, str]] = []
    for analysis in ctx.shown_analyses.values():
        sql = analysis.get("sql") or ""
        normalised = _normalise(sql, dialect)
        stage = RETRIEVED
        if _runs_whole(normalised, submitted_sql, dialect):
            stage = EXECUTED
        elif submitted_tables & _tables(sql, dialect) and (
            (saved_filters := _filters(sql, dialect))
            and saved_filters <= submitted_filters
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
