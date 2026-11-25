# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provenance of saved custom analyses in a ReAct SQL answer.

``custom_analyses_used`` lists every saved analysis the episode's ``retrieve``
showed the model, each with the furthest stage that can be established:

``retrieved``
    The definition was put in front of the model. Nothing shows it was used.
``reference``
    Retrieved, and the submitted SQL reads at least one table the analysis's
    SQL reads. This is a heuristic about overlap, not proof the model followed
    the definition; the submitted SQL is generated, not the saved analysis.
``executed``
    The submitted SQL, which is what ran, is the analysis's SQL: the same
    statement once parsed and normalised, so whitespace, keyword case and a
    trailing semicolon do not matter. A repaired or reinterpreted query is
    never reported as executed.

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


def _tables(sql: str | None, dialect: str | None) -> set[str]:
    if not sql:
        return set()
    try:
        statements = sqlglot.parse(sql, read=dialect)
    except sqlglot.errors.SqlglotError:
        return set()
    return {
        table.name.lower()
        for statement in statements
        if statement is not None
        for table in statement.find_all(exp.Table)
        if table.name
    }


def _fingerprint(sql: str, normalised: str | None) -> str:
    return hashlib.sha256((normalised or " ".join(sql.split())).encode()).hexdigest()


def custom_analyses_used(
    ctx: ToolContext, submitted_sql: str | None, dialect: str | None = None
) -> list[dict[str, str]]:
    """Provenance entries for every analysis shown in this episode.

    *submitted_sql* is the query that ran, or ``None`` when no SQL was
    submitted, in which case nothing can be past ``retrieved``.
    """
    submitted = _normalise(submitted_sql, dialect)
    submitted_tables = _tables(submitted_sql, dialect)
    entries: list[dict[str, str]] = []
    for analysis in ctx.shown_analyses.values():
        sql = analysis.get("sql") or ""
        normalised = _normalise(sql, dialect)
        stage = RETRIEVED
        if submitted is not None and normalised == submitted:
            stage = EXECUTED
        elif submitted_tables & _tables(sql, dialect):
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
