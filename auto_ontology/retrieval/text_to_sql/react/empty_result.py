# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hold back a 0-row answer from a filtered query until its filters are checked.

A filter literal that does not match how the value is stored returns 0 rows
just like a value that is absent: ``site_id = '1'`` on a SQLite column holding
the integer 1 matches nothing, and "no orders" reads as a fact. ``run_sql``
answers an empty result with a hint to check the literals; this gives
``submit`` the same check, with the evidence gathered for the model.

When a final query with a WHERE or HAVING clause returns no rows, each filter
that compares a table's column with a literal is run alone against that table
through :class:`ProbeExecutor`. One that matches nothing gets its stored
values: those equal to the literal as text ignoring case, or a sample, or the
stored range. The findings go back to the model as a rejected submission.

It happens once per query. An empty result can be the right answer, and the
model that has seen the evidence and submits the same query again is
believed; a different query is checked on its own. When the
evidence cannot be gathered, the rejection says so instead of passing the
empty result as checked. Where queries are charged nothing is probed, since
the model decides what to spend. The episode's last answer attempt is accepted
without probing because the model would get no turn to use the evidence.
"""

from __future__ import annotations

import numbers
from dataclasses import dataclass
from typing import Any

from sqlglot import exp

from auto_ontology.connectors.base import SQLDatabase
from auto_ontology.infra.feature_flags import get_sql_query_timeout_seconds
from auto_ontology.retrieval.text_to_sql.agents.sql_execution import QueryResponse
from auto_ontology.retrieval.text_to_sql.db_probe import ProbeExecutor
from auto_ontology.retrieval.text_to_sql.react.context import ToolContext
from auto_ontology.retrieval.text_to_sql.react.run_sql import (
    CatalogNames,
    catalog_for,
    catalog_table,
)

_MAX_FILTERS = 4
# Each filter costs at most three probes: alone, values equal to it as text,
# then a sample or the range of its stored values.
_MAX_PROBES = 3 * _MAX_FILTERS
_SAMPLE_VALUES = 10
_MAX_VALUE_CHARS = 60

_EQUALITY = (exp.EQ, exp.In)
_RANGE = (exp.GT, exp.GTE, exp.LT, exp.LTE, exp.Between)
_PATTERN = (exp.Like, exp.ILike)
_NOT_PROBED_ALONE = (exp.AggFunc, exp.Window, exp.Subquery, exp.Select)


@dataclass
class _Filter:
    """A comparison of one table's columns with literal values."""

    predicate: exp.Expression
    subject: exp.Expression
    literals: list[exp.Expression]
    table: exp.Table | None


def is_empty_result(db_response: QueryResponse) -> bool:
    """Whether a query that ran returned no rows."""
    return bool(db_response.result) and all(
        (part or "").strip() in ("", "[]") for part in db_response.result
    )


def empty_result_rejection(
    ctx: ToolContext,
    connector: SQLDatabase,
    statement: exp.Expression | None,
    db_response: QueryResponse,
    resubmit: str = "submit this version",
) -> str | None:
    """Why an empty result from a filtered query is not accepted yet, or ``None``.

    ``None`` on the last answer attempt, for a result with rows, a query without
    WHERE or HAVING, and a query this already rejected once.
    """
    if ctx.last_answer_attempt:
        return None
    if (
        statement is None
        or not is_empty_result(db_response)
        or not any(statement.find_all(exp.Where, exp.Having))
    ):
        return None
    query = statement.sql()
    if query in ctx.empty_result_checked:
        return None
    ctx.empty_result_checked.add(query)
    again = f"If 0 rows is the right answer, {resubmit} again."
    if ctx.cost_status is not None or ctx.can_run_queries is not None:
        return _unchecked("queries here are charged, so none were run", again)
    dialect = getattr(connector, "dialect", None) or None
    filters = literal_filters(statement, catalog_for(ctx, connector))
    probed = [f for f in filters if f.table is not None][:_MAX_FILTERS]
    not_checked = [
        f.predicate.sql(dialect=dialect)
        for f in filters
        if not any(f is p for p in probed)
    ]
    if not probed:
        reason = (
            "each filter reads a CTE, a subquery, an aggregate or more than one "
            f"table: {'; '.join(not_checked)}"
            if not_checked
            else "no filter compares a column with a value"
        )
        return _unchecked(reason, again)
    if getattr(connector, "supports_statement_timeout", False) is not True:
        return _unchecked(
            "the connector cannot cap probe statements in time",
            again,
        )
    executor = ProbeExecutor(
        connector,
        max_calls=_MAX_PROBES,
        max_rows=_SAMPLE_VALUES + 1,
        total_timeout_s=get_sql_query_timeout_seconds(),
    )
    lines = [_probe(executor, f, f.table, dialect) for f in probed if f.table]
    if not any(entry["ok"] for entry in executor.log):
        reason = next((e["error"] for e in executor.log if e["error"]), "no probe ran")
        return _unchecked(f"the probes failed: {reason}", again)
    if not_checked:
        lines.append(f"- Not checked: {'; '.join(not_checked)}")
    return (
        "Not accepted yet. The query returned 0 rows. A filter value that does "
        "not match how the data stores it also returns 0 rows, so each filter "
        "was run alone against its table:\n"
        + "\n".join(lines)
        + f"\nFix the query if a filter is wrong. {again}"
    )


def _unchecked(reason: str, again: str) -> str:
    return (
        "Not accepted yet. The query returned 0 rows, and its filter values "
        f"could not be checked against the stored values ({reason}). Check them "
        f"with run_sql (SELECT DISTINCT <column> ...). {again}"
    )


def literal_filters(statement: exp.Expression, catalog: CatalogNames) -> list[_Filter]:
    """The WHERE and HAVING comparisons of a column expression with literals.

    ``table`` is the one catalog table the column side reads, or ``None`` when
    it cannot be probed alone: it aggregates, reads a CTE or subquery, or its
    columns are not all in one table.
    """
    derived = {cte.alias_or_name.lower() for cte in statement.find_all(exp.CTE)}
    seen: set[int] = set()
    filters: list[_Filter] = []
    for clause in statement.find_all(exp.Where, exp.Having):
        for node in clause.find_all(*_EQUALITY, *_RANGE, *_PATTERN, bfs=False):
            if id(node) in seen or _negated(node):
                continue
            seen.add(id(node))
            sides = _split(node)
            if sides is None:
                continue
            subject, literals = sides
            table = None
            if not subject.find(*_NOT_PROBED_ALONE) and not _reads_output_alias(
                subject, node
            ):
                table = _table_of(subject, node, derived, catalog)
            filters.append(_Filter(node, subject, literals, table))
    return filters


def _negated(node: exp.Expression) -> bool:
    """Whether *node* is negated, itself or by a NOT around it in its clause."""
    if node.args.get("negate"):
        return True
    parent = node.parent
    while parent is not None and not isinstance(parent, (exp.Where, exp.Having)):
        if isinstance(parent, exp.Not):
            return True
        parent = parent.parent
    return False


def _reads_output_alias(subject: exp.Expression, predicate: exp.Expression) -> bool:
    """Whether *subject* names an output of its SELECT, as ``HAVING cnt > 5`` does.

    Such a name is not a column of any table, or not the same value as one, so
    it cannot be probed alone. ``status AS status`` is still the column.
    """
    select = predicate.find_ancestor(exp.Select)
    if select is None:
        return False
    outputs = {
        e.alias.lower()
        for e in select.expressions
        if e.alias
        and not (
            isinstance(e.this, exp.Column) and e.this.name.lower() == e.alias.lower()
        )
    }
    return any(
        not column.table and column.name.lower() in outputs
        for column in subject.find_all(exp.Column)
    )


def _is_value(node: exp.Expression | None) -> bool:
    """A side that reads nothing: a literal, or an expression of literals."""
    return node is not None and not node.find(exp.Column, *_NOT_PROBED_ALONE)


def _split(node: exp.Expression) -> tuple[exp.Expression, list[exp.Expression]] | None:
    """``(column side, literal side)`` of a comparison, or ``None``."""
    if isinstance(node, (exp.In, exp.Between)) and _is_value(node.this):
        return None
    if isinstance(node, exp.In):
        values = list(node.expressions or [])
        if node.args.get("query") or not values or not all(map(_is_value, values)):
            return None
        return node.this, values
    if isinstance(node, exp.Between):
        values = [node.args.get("low"), node.args.get("high")]
        return (node.this, values) if all(map(_is_value, values)) else None
    left, right = node.this, node.expression
    if _is_value(right) and not _is_value(left):
        return left, [right]
    if _is_value(left) and right is not None and not _is_value(right):
        return right, [left]
    return None


def _scope_tables(select: exp.Select | None) -> list[exp.Expression]:
    """The FROM and JOIN sources of *select*, not of its subqueries."""
    if select is None:
        return []
    from_ = select.args.get("from_") or select.args.get("from")
    sources = [from_.this] if from_ is not None else []
    return sources + [join.this for join in select.args.get("joins") or []]


def _table_of(
    subject: exp.Expression,
    predicate: exp.Expression,
    derived: set[str],
    catalog: CatalogNames,
) -> exp.Table | None:
    sources = _scope_tables(predicate.find_ancestor(exp.Select))
    tables = [
        s
        for s in sources
        if isinstance(s, exp.Table) and s.name and s.name.lower() not in derived
    ]
    by_name = {t.alias_or_name.lower(): t for t in tables}
    owners: set[int] = set()
    owner: exp.Table | None = None
    for column in subject.find_all(exp.Column):
        if column.table:
            found = by_name.get(column.table.lower())
        elif len(sources) == 1 and len(tables) == 1:
            found = tables[0]
        else:
            having = [t for t in tables if _has_column(t, column.name, catalog)]
            found = having[0] if len(having) == 1 else None
        if found is None:
            return None
        owners.add(id(found))
        owner = found
    return owner if len(owners) == 1 else None


def _has_column(table: exp.Table, column: str, catalog: CatalogNames) -> bool:
    schema = table.db or None
    columns = catalog.columns(catalog_table(table.name, schema, catalog), schema)
    return any(c.lower() == column.lower() for c in columns or [])


def _query(
    projection: list[exp.Expression],
    table: exp.Table,
    where: exp.Expression,
    limit: int,
    dialect: str | None,
    distinct: bool = False,
) -> str:
    select = exp.select(*projection).from_(table.copy()).where(where).limit(limit)
    if distinct:
        select = select.distinct()
    return select.sql(dialect=dialect)


def _probe(
    executor: ProbeExecutor, f: _Filter, table: exp.Table, dialect: str | None
) -> str:
    """One line on what *f* matches in *table* by itself; stored values if nothing."""
    shown = f.predicate.sql(dialect=dialect)
    alone = executor.run(
        _query([exp.Literal.number(1)], table, f.predicate.copy(), 1, dialect),
        purpose="empty_result_filter_alone",
    )
    if not alone["ok"]:
        return f"- {shown}: could not run it alone ({_first_line(alone['error'])})."
    if alone["rows"]:
        return (
            f"- {shown}: matches rows on its own, so the rows drop out where it "
            "meets the other filters or the joins."
        )
    head = f"- {shown}: matches no row of {table.name} on its own."
    subject = f.subject.copy()
    plain = all(isinstance(v, exp.Literal) for v in f.literals)
    if isinstance(f.predicate, _EQUALITY) and plain:
        texts = [exp.Literal.string(str(v.this).strip().lower()) for v in f.literals]
        as_text = exp.Lower(this=exp.Trim(this=exp.cast(subject.copy(), "TEXT")))
        found = executor.run(
            _query(
                [subject.as_("value")],
                table,
                exp.In(this=as_text, expressions=texts),
                _SAMPLE_VALUES + 1,
                dialect,
                distinct=True,
            ),
            purpose="empty_result_values_as_text",
        )
        values = _values(found)
        if values:
            return (
                f"{head} Stored values equal to it as text, ignoring case: "
                f"{_render(values)}.{_type_note(f.literals, values)}"
            )
    if isinstance(f.predicate, _RANGE):
        entry = executor.run(
            _query(
                [
                    exp.Min(this=subject.copy()).as_("low"),
                    exp.Max(this=subject).as_("high"),
                ],
                table,
                exp.Not(this=exp.Is(this=f.subject.copy(), expression=exp.Null())),
                1,
                dialect,
            ),
            purpose="empty_result_range",
        )
        if not entry["ok"]:
            return f"{head} Its stored range could not be read."
        # By position: an engine may return the aliases in upper case.
        low, high = [*(entry["rows"] or [{}])[0].values(), None, None][:2]
        if low is None:
            return f"{head} It has no stored values."
        note = _type_note(f.literals, [low, high])
        return f"{head} Stored range: {_value(low)} to {_value(high)}.{note}"
    entry = executor.run(
        _query(
            [subject.as_("value")],
            table,
            exp.Not(this=exp.Is(this=f.subject.copy(), expression=exp.Null())),
            _SAMPLE_VALUES + 1,
            dialect,
            distinct=True,
        ),
        purpose="empty_result_sample",
    )
    if not entry["ok"]:
        return f"{head} Its stored values could not be read."
    values = _values(entry)
    if not values:
        return f"{head} It has no stored values."
    return f"{head} Stored values: {_render(values)}.{_type_note(f.literals, values)}"


def _values(entry: dict[str, Any]) -> list[Any]:
    if not entry["ok"]:
        return []
    return [next(iter(row.values()), None) for row in entry["rows"] or []]


def _first_line(text: str | None) -> str:
    return (text or "unknown error").strip().splitlines()[0][:200]


def _value(value: Any) -> str:
    if isinstance(value, str):
        text = "'" + value.replace("'", "''") + "'"
    else:
        text = "NULL" if value is None else str(value)
    return text if len(text) <= _MAX_VALUE_CHARS else text[: _MAX_VALUE_CHARS - 1] + "…"


def _render(values: list[Any]) -> str:
    shown = ", ".join(_value(v) for v in values[:_SAMPLE_VALUES])
    return shown + (", …" if len(values) > _SAMPLE_VALUES else "")


def _is_number(value: Any) -> bool:
    return isinstance(value, numbers.Number) and not isinstance(value, bool)


def _type_note(literals: list[exp.Expression], values: list[Any]) -> str:
    """A hint when the literals and the stored values differ in kind."""
    plain = [v for v in literals if isinstance(v, exp.Literal)]
    if not values or len(plain) != len(literals):
        return ""
    if all(v.is_string for v in plain) and all(map(_is_number, values)):
        return " They are stored as numbers: compare without quotes."
    if not any(v.is_string for v in plain) and all(isinstance(v, str) for v in values):
        return " They are stored as text: quote the value."
    return ""
