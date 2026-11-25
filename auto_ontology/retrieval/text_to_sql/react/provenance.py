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
    keeps every WHERE/HAVING condition of the analysis. Conditions are compared
    by the table each column belongs to (aliases resolved, ``table__`` semantic
    prefixes removed), so the same column name on another table does not match,
    and only those of the answer count, not of a CTE nothing reads. Aggregates,
    joins and table overlap do not count, and a changed or opposite filter
    leaves the analysis at ``retrieved``; so does an analysis with no filter,
    since nothing can then be compared. This is still a heuristic, not proof the model followed
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
from sqlglot.optimizer.normalize_identifiers import normalize_identifiers

from auto_ontology.retrieval.text_to_sql.react.context import ToolContext

RETRIEVED = "retrieved"
REFERENCE = "reference"
EXECUTED = "executed"


def _normalise(sql: str | None, dialect: str | None) -> str | None:
    """*sql* as one canonical statement, or ``None`` if it is not exactly one."""
    if not sql or not sql.strip():
        return None
    statements = _parse(sql, dialect)
    if len(statements) != 1:
        return None
    return statements[0].sql(dialect=dialect, normalize=True)


def _parse(sql: str | None, dialect: str | None) -> list[exp.Expression]:
    """*sql* as statements whose identifiers follow the dialect's case rules.

    Unquoted names are folded the way the database folds them and a quoted name
    keeps its case where the database does, so equal strings are the same name.
    """
    if not sql:
        return []
    try:
        return [
            normalize_identifiers(e, dialect)
            for e in sqlglot.parse(sql, read=dialect)
            if e is not None
        ]
    except sqlglot.errors.SqlglotError:
        return []


def _enclosing_ctes(node: exp.Expression) -> list[exp.CTE]:
    """Every CTE definition *node* sits inside, innermost first."""
    found: list[exp.CTE] = []
    parent = node.parent
    while parent is not None:
        if isinstance(parent, exp.CTE):
            found.append(parent)
        parent = parent.parent
    return found


def _cte_read_by(table: exp.Table) -> exp.CTE | None:
    """The CTE definition *table* names, or ``None`` for a physical table.

    The nearest enclosing WITH that defines the name and is visible from the
    read, so two CTEs that share a name in different scopes stay two definitions.
    A schema- or catalog-qualified name is always a physical table. In a
    WITH that is not recursive, a CTE sees only the ones defined before it, so
    ``WITH orders AS (SELECT ... FROM orders)`` reads the physical table.
    """
    if table.args.get("db") or table.args.get("catalog"):
        return None
    name = table.name
    node = table.parent
    while node is not None:
        clause = node.args.get("with_") or node.args.get("with")
        if clause is not None:
            definitions = list(clause.expressions)
            inside = next(
                (c for c in _enclosing_ctes(table) if c.parent is clause), None
            )
            if inside is not None and not clause.args.get("recursive"):
                definitions = definitions[: definitions.index(inside)]
            for cte in definitions:
                if cte.alias_or_name == name:
                    return cte
        node = node.parent
    return None


def _live_ctes(root: exp.Expression) -> set[int]:
    """The CTE definitions that contribute to the answer of *root*, by ``id``.

    A definition is live when the answer query reads it, or a live CTE does. One
    that nothing live reads, or only another unused CTE does, does not contribute
    to the answer, so what it filters or reads says nothing about the answer.
    """
    live: set[int] = set()
    while True:
        new = {
            id(cte)
            for table in root.find_all(exp.Table)
            if table.name
            and all(id(c) in live for c in _enclosing_ctes(table))
            and (cte := _cte_read_by(table)) is not None
        } - live
        if not new:
            return live
        live |= new


def _is_live(node: exp.Expression, live: set[int]) -> bool:
    return all(id(cte) in live for cte in _enclosing_ctes(node))


def _tables(sql: str | None, dialect: str | None) -> set[str]:
    """Physical tables the answer reads, leaving out CTE names and unused CTEs."""
    names: set[str] = set()
    for statement in _parse(sql, dialect):
        live = _live_ctes(statement)
        names |= {
            t.name
            for t in statement.find_all(exp.Table)
            if t.name and _cte_read_by(t) is None and _is_live(t, live)
        }
    return names


def _sources(select: exp.Expression) -> dict[str, str]:
    """What each qualifier in *select*'s own FROM and JOINs stands for.

    A table alias or name maps to the table; a derived table's alias to itself.
    """
    out: dict[str, str] = {}
    for table in select.find_all(exp.Table):
        if table.name and table.find_ancestor(exp.Select) is select:
            # Once aliased, the table's own name is not a qualifier: another
            # table may carry it as its alias.
            out[table.alias_or_name] = table.name
    for subquery in select.find_all(exp.Subquery):
        if subquery.alias and subquery.find_ancestor(exp.Select) is select:
            out[subquery.alias] = subquery.alias
    return out


def _owner(column: exp.Column) -> str | None:
    """The table *column* belongs to, as far as the SQL says.

    The table its qualifier names, looked up from the nearest SELECT outwards (a
    correlated subquery uses the outer query's aliases); else the one source of
    the nearest SELECT. A ``table__column`` prefix is not used: it names the
    concept (``order_line``), not necessarily the table (``order_lines``).
    """
    qualifier = column.table or ""
    select = column.find_ancestor(exp.Select)
    if qualifier:
        while select is not None:
            if qualifier in (sources := _sources(select)):
                return sources[qualifier]
            select = select.find_ancestor(exp.Select)
        return qualifier
    if select is not None and len(owners := set(_sources(select).values())) == 1:
        return next(iter(owners))
    return None


def _with_owned_columns(statement: exp.Expression) -> exp.Expression:
    """A copy of *statement* whose columns read ``owner.column``.

    Aliases become the table they stand for and ``table__column`` semantic names
    become ``column``, so a filter compares equal to the same filter written with
    another alias or the physical name, and not to one on another table's column
    of the same name.
    """
    statement = statement.copy()
    for column in list(statement.find_all(exp.Column)):
        owner = _owner(column)
        name = column.name.rsplit("__", 1)[-1]
        column.set("this", exp.to_identifier(name))
        column.set("table", exp.to_identifier(owner) if owner else None)
        column.set("db", None)
        column.set("catalog", None)
    return statement


def _conjuncts(condition: exp.Expression) -> list[exp.Expression]:
    if isinstance(condition, exp.Paren):
        return _conjuncts(condition.this)
    if isinstance(condition, exp.And):
        return _conjuncts(condition.left) + _conjuncts(condition.right)
    return [condition]


def _filters(sql: str | None, dialect: str | None) -> set[str]:
    """The WHERE/HAVING conditions of *sql*'s answer, one per AND-ed condition.

    Conditions of a CTE nothing reads are left out. Each names the table its
    columns belong to.
    """
    out: set[str] = set()
    for statement in _parse(sql, dialect):
        owned = _with_owned_columns(statement)
        live = _live_ctes(owned)
        for clause in owned.find_all(exp.Where, exp.Having):
            if not _is_live(clause, live):
                continue
            for part in _conjuncts(clause.this):
                out.add(part.sql(dialect=dialect, normalize=True))
    return out


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
    live = _live_ctes(root)
    return any(
        select.sql(dialect=dialect, normalize=True) == normalised
        and _is_live(select, live)
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
