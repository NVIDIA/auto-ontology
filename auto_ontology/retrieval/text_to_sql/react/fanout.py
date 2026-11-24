# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Aggregates a join repeats: the ``submit`` check for join fanout.

``SUM(s.rate) FROM suppliers s JOIN asn a ON a.supplier_id = s.id`` adds each
supplier's rate once per ASN, and nothing in the result says so: the query
runs and the totals look plausible. This module finds that shape in one
SELECT at a time. An aggregate is reported when the query also joins a table
that is not among the ones the aggregate reads, and that table's only join
column has values that repeat, so every row it is joined to is counted once
per match.

It only reports what the catalog proves. "Repeats" means profiling saw a
value twice in the column's sample (``distinct_ratio`` below 1.0) and the
column is not a declared single-column key. Everything else leaves the query
alone: a join column with no measured ratio, a join on more than one column,
an equality filter on the joined table (it may pick one row per key),
derived tables, ``COUNT(DISTINCT ...)``, ``COUNT(*)``, window aggregates, and
a ``GROUP BY`` that holds a unique key of the joined table. An aggregate over
a subquery or CTE that already aggregated the detail is not affected, because
that inner SELECT is checked on its own and the outer one joins no detail
rows.

A false report costs the model a turn, not the answer: ``submit`` refuses a
given SQL text once, and accepts it unchanged the second time, for the
questions that do want one value per detail row.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from sqlglot import exp
from sqlglot.optimizer.scope import Scope, traverse_scope

from auto_ontology.dal.datasources import fetch_table_key_profile
from auto_ontology.retrieval.text_to_sql.react.context import ToolContext

logger = logging.getLogger(__name__)

_MEASURES = (exp.Sum, exp.Avg, exp.Count)
_MAX_REPORTED = 5


@dataclass(frozen=True)
class KeyProfile:
    """What the catalog proves about one table's columns, names lowercased.

    ``unique`` holds a declared single-column primary key and columns profiled
    unique with no repeat in their sample. ``repeats`` holds columns whose
    sample had a value twice, which no unique column can.
    """

    pk: tuple[str, ...] = ()
    unique: frozenset[str] = frozenset()
    repeats: frozenset[str] = frozenset()

    @classmethod
    def from_catalog(cls, row: dict[str, Any]) -> KeyProfile:
        """From :func:`fetch_table_key_profile`'s shape."""
        pk = tuple(str(c).lower() for c in row.get("pk") or ())
        unique = set(pk) if len(pk) == 1 else set()
        repeats = set()
        for name, (is_unique, ratio) in (row.get("columns") or {}).items():
            if ratio is not None and ratio < 1.0:
                repeats.add(name.lower())
            elif is_unique:
                unique.add(name.lower())
        return cls(pk, frozenset(unique), frozenset(repeats - unique))


# (table name, schema or None) -> the table's profile, None when unknown.
ProfileLookup = Callable[[str, str | None], KeyProfile | None]


class _Columns(Protocol):
    """What the fanout check needs from the catalog: a table's column names."""

    def columns(self, table: str, schema: str | None) -> list[str] | None:
        """Column names of *table*, or ``None`` when the catalog has no such table."""


@dataclass(frozen=True)
class Fanout:
    """One aggregate that a joined table repeats."""

    aggregate: str
    measured: tuple[str, ...]
    detail: str
    detail_alias: str
    key: str

    def describe(self) -> str:
        read = ", ".join(self.measured)
        key = f"{self.detail_alias}.{self.key}"
        return (
            f"{self.aggregate} reads {read} rows, but the join to {self.detail} on "
            f"{key} repeats each of them once per matching {self.detail} row "
            f"({key} has repeated values). Aggregate {self.detail} in a subquery "
            f"or CTE grouped by {self.key} before joining it, or compute each "
            "measure in its own subquery at the grain it belongs to and join "
            "those results; COUNT(DISTINCT ...) counts each row once."
        )


@dataclass
class _Source:
    name: str
    schema: str | None
    physical: bool
    alias: str


def find_fanouts(
    statement: exp.Expression, catalog: _Columns, profile: ProfileLookup
) -> list[Fanout]:
    """Every aggregate in *statement* that a join repeats; none when unsure."""
    try:
        scopes = traverse_scope(statement)
    except Exception:  # noqa: BLE001 — a shape sqlglot cannot scope is not a finding
        logger.debug("fanout: could not scope the query", exc_info=True)
        return []
    cache: dict[tuple[str, str | None], KeyProfile | None] = {}

    def cached(name: str, schema: str | None) -> KeyProfile | None:
        key = (name.lower(), (schema or "").lower() or None)
        if key not in cache:
            cache[key] = profile(name, schema)
        return cache[key]

    found: list[Fanout] = []
    for scope in scopes:
        if isinstance(scope.expression, exp.Select):
            found += _ScopeCheck(scope, catalog, cached).fanouts()
    return list(dict.fromkeys(found))


class _ScopeCheck:
    """The fanout check of one SELECT, without the queries nested in it."""

    def __init__(self, scope: Scope, catalog: _Columns, profile: ProfileLookup):
        self.select = scope.expression
        self.scope = scope
        self.catalog = catalog
        self.profile = profile
        self.sources: dict[str, _Source] = {}
        for alias, (_node, source) in scope.selected_sources.items():
            physical = isinstance(source, exp.Table) and bool(source.name)
            self.sources[alias.lower()] = _Source(
                source.name if physical else alias,
                (source.db or None) if physical else None,
                physical,
                alias,
            )
        # Join columns per source: lowercased name -> the name as written.
        self.keys: dict[str, dict[str, str]] = {a: {} for a in self.sources}
        self.uncertain: set[str] = set()
        self.grouped: dict[str, set[str]] = {a: set() for a in self.sources}

    def fanouts(self) -> list[Fanout]:
        aggregates = [a for a in self.scope.find_all(*_MEASURES) if _counts_rows(a)]
        if not aggregates or len(self.sources) < 2:
            return []
        self._read_joins()
        self._read_group_by()
        found = []
        for aggregate in aggregates:
            measured = self._measured(aggregate)
            if not measured:
                continue
            for alias, source in self.sources.items():
                if alias in measured:
                    continue
                key = self._repeating_key(alias, source)
                if key:
                    found.append(
                        Fanout(
                            aggregate.sql(),
                            tuple(self.sources[a].name for a in sorted(measured)),
                            source.name,
                            source.alias,
                            key,
                        )
                    )
        return found

    # -- columns ---------------------------------------------------------

    def _local(self, column: exp.Column) -> bool:
        return column.find_ancestor(exp.Select) is self.select

    def _alias_of(self, column: exp.Column) -> str | None:
        """The source of this SELECT that *column* reads, when certain."""
        qualifier = (column.table or "").lower()
        if qualifier:
            return qualifier if qualifier in self.sources else None
        name = column.name.lower()
        owners = [
            alias
            for alias, source in self.sources.items()
            if source.physical
            and name
            in {
                c.lower()
                for c in self.catalog.columns(source.name, source.schema) or []
            }
        ]
        return owners[0] if len(owners) == 1 else None

    def _measured(self, aggregate: exp.Expression) -> set[str] | None:
        """The sources *aggregate*'s argument reads; None when any is unknown."""
        columns = [c for c in aggregate.this.find_all(exp.Column) if self._local(c)]
        aliases = {self._alias_of(c) for c in columns}
        if not aliases or None in aliases:
            return None
        return aliases  # type: ignore[return-value]

    def _referenced(self, node: exp.Expression) -> set[str]:
        """Sources *node* touches, including from subqueries correlated to them."""
        aliases = set()
        for column in node.find_all(exp.Column):
            if self._local(column):
                alias = self._alias_of(column)
            else:
                qualifier = (column.table or "").lower()
                alias = qualifier if qualifier in self.sources else None
            if alias:
                aliases.add(alias)
        return aliases

    # -- joins -------------------------------------------------------------

    def _read_joins(self) -> None:
        """Equi-join columns per source, and sources whose joins are unclear."""
        conditions: list[exp.Expression] = []
        for join in self.select.args.get("joins") or []:
            right = join.this.alias_or_name.lower()
            if join.args.get("method"):  # NATURAL: the columns are implicit
                self.uncertain.add(right)
            if on := join.args.get("on"):
                conditions.append(on)
            for using in join.args.get("using") or []:
                self._read_using(right, using.name)
        if where := self.select.args.get("where"):
            conditions.append(where.this)
        for condition in conditions:
            for conjunct in _conjuncts(condition):
                self._read_conjunct(conjunct)

    def _read_using(self, right: str, using: str) -> None:
        name = using.lower()
        if right not in self.sources:
            return
        left = [
            alias
            for alias, source in self.sources.items()
            if alias != right
            and source.physical
            and name
            in {
                c.lower()
                for c in self.catalog.columns(source.name, source.schema) or []
            }
        ]
        if len(left) != 1:
            self.uncertain.add(right)
            return
        self.keys[right].setdefault(name, using)
        self.keys[left[0]].setdefault(name, using)

    def _read_conjunct(self, conjunct: exp.Expression) -> None:
        if isinstance(conjunct, exp.EQ) and not conjunct.find(exp.Select):
            left, right = _bare(conjunct.this), _bare(conjunct.expression)
            if isinstance(left, exp.Column) and isinstance(right, exp.Column):
                a, b = self._alias_of(left), self._alias_of(right)
                if a and b and a != b and self._local(left) and self._local(right):
                    self.keys[a].setdefault(left.name.lower(), left.name)
                    self.keys[b].setdefault(right.name.lower(), right.name)
                    return
        referenced = self._referenced(conjunct)
        # An equality can pick one row per key (``currency = 'USD'``); a
        # condition across sources or through a subquery can do the same. A
        # range or pattern filter on one source cannot make its rows unique.
        if (
            isinstance(conjunct, (exp.EQ, exp.NullSafeEQ, exp.In))
            or len(referenced) > 1
            or conjunct.find(exp.Select)
        ):
            self.uncertain |= referenced

    def _read_group_by(self) -> None:
        group = self.select.args.get("group")
        if not group:
            return
        outputs = {
            e.alias.lower(): e.unalias() for e in self.select.expressions if e.alias
        }
        for item in group.expressions:
            item = _bare(item)
            if isinstance(item, exp.Literal) and item.is_int:
                position = int(item.this) - 1
                if 0 <= position < len(self.select.expressions):
                    item = _bare(self.select.expressions[position].unalias())
            elif (
                isinstance(item, exp.Column)
                and not item.table
                and self._alias_of(item) is None
                and item.name.lower() in outputs
            ):
                item = _bare(outputs[item.name.lower()])
            if isinstance(item, exp.Column) and (alias := self._alias_of(item)):
                self.grouped[alias].add(item.name.lower())

    # -- the finding -------------------------------------------------------

    def _repeating_key(self, alias: str, source: _Source) -> str | None:
        """The join column through which *alias* repeats the other rows, if proven."""
        keys = self.keys.get(alias) or {}
        if not source.physical or alias in self.uncertain or len(keys) != 1:
            return None
        profile = self.profile(source.name, source.schema)
        if profile is None:
            return None
        ((key, written),) = keys.items()
        if key in profile.unique or key not in profile.repeats:
            return None
        grouped = self.grouped.get(alias) or set()
        if grouped & profile.unique or (profile.pk and set(profile.pk) <= grouped):
            # One row of it per group: grouping by its key undoes the repeat.
            return None
        return written


def _counts_rows(aggregate: exp.Expression) -> bool:
    """Whether *aggregate* adds up rows, so repeating a row changes it."""
    if isinstance(aggregate.find_ancestor(exp.Window, exp.Select), exp.Window):
        return False
    argument = aggregate.this
    return argument is not None and not isinstance(argument, (exp.Distinct, exp.Star))


def _bare(node: exp.Expression) -> exp.Expression:
    while isinstance(node, (exp.Paren, exp.Cast, exp.TryCast)):
        node = node.this
    return node


def _conjuncts(condition: exp.Expression) -> list[exp.Expression]:
    condition = _bare(condition) if isinstance(condition, exp.Paren) else condition
    if isinstance(condition, exp.And):
        return _conjuncts(condition.this) + _conjuncts(condition.expression)
    return [condition]


# -- the catalog -------------------------------------------------------------


class CatalogKeyProfiles:
    """:data:`ProfileLookup` over the catalog the episode's SQL checks loaded.

    A table name that more than one schema has, and the query does not
    qualify, is unknown rather than guessed: the database's search path picks.
    """

    def __init__(self, schemas: dict[str, Any]):
        self._schemas = schemas

    def __call__(self, table: str, schema: str | None) -> KeyProfile | None:
        ids = []
        for key, owner in self._schemas.items():
            if schema and (key or "").lower() != schema.lower():
                continue
            frame = getattr(owner, "tables_df", None)
            if frame is None or "id" not in frame or "table_name" not in frame:
                continue
            rows = frame.loc[frame["table_name"].str.lower() == table.lower(), "id"]
            ids += [str(i) for i in rows if i]
        if len(ids) != 1:
            return None
        row = fetch_table_key_profile(ids[0])
        return KeyProfile.from_catalog(row) if row else None


def key_profiles(ctx: ToolContext, database_name: str | None) -> ProfileLookup:
    """The profiles of *database_name*'s tables, from the catalog loaded for it."""
    return CatalogKeyProfiles(ctx.schemas.get(database_name) or {})


def safe_find_fanouts(
    statement: exp.Expression,
    catalog: _Columns,
    ctx: ToolContext,
    database_name: str | None,
) -> list[Fanout]:
    """:func:`find_fanouts`, where a failing lookup finds nothing.

    The check only advises against a query that already passed the others, so
    an error in it must not keep the query from running.
    """
    try:
        return find_fanouts(statement, catalog, key_profiles(ctx, database_name))
    except Exception:  # noqa: BLE001
        logger.warning("fanout check failed; not applied", exc_info=True)
        return []


def _listed(fanouts: list[Fanout]) -> str:
    return "\n- ".join(f.describe() for f in fanouts[:_MAX_REPORTED])


def submit_refusal(
    ctx: ToolContext,
    fanouts: list[Fanout],
    sql: str,
    resubmit: str = "submit this version",
) -> str | None:
    """Why ``submit`` refuses *sql*, the first time only; ``None`` to accept it.

    *resubmit* says how to send the same query again, as in ``submit``.
    """
    if not fanouts or sql in ctx.fanout_refused:
        return None
    ctx.fanout_refused.add(sql)
    return (
        "Not accepted. A join repeats the rows an aggregate adds up:\n- "
        f"{_listed(fanouts)}\nIf the question does want one value per row of the "
        f"joined table, {resubmit} again unchanged."
    )


def run_sql_hint(fanouts: list[Fanout]) -> str:
    """A note for ``run_sql``'s result; the query ran, so it only warns."""
    if not fanouts:
        return ""
    return (
        "\n\nNote: a join repeats the rows an aggregate adds up; submit will "
        f"not accept this query as it is:\n- {_listed(fanouts)}"
    )
