# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checks that ask whether a *simpler* query would answer the question.

Every other check in the pipeline asks whether a query is valid. These ask
whether it is more elaborate than the question called for, which is a
different and more expensive failure: a query can be valid, execute, return
plausible rows, and still answer a question nobody asked. Three shapes
account for it, and all three are something the model *added* to an
otherwise correct query:

- an aggregate that collapsed several matching rows into one answer,
- a join through tables whose columns the answer never reads,
- a strict ``>`` where the row on the boundary belonged in the result.

Measured against the single-call baseline on a paired slice, those three
signatures fired on 4% of questions and every firing was a regression — the
elaboration never once helped.

A fourth shape is the mirror image and sits at the end of this module: a
relationship tested with ``EXISTS`` or ``IN`` where the question wanted it
joined. It is here because the plumbing is identical — build the other
query, run both, speak only if they disagree — though its cause is not
elaboration but the opposite, a query kept flat because flat queries probe
cleanly.

All four are narrow by construction: each one looks
for its exact shape, builds the less elaborate query, and only speaks up
when the two actually disagree. A query that reads the same either way is
left alone.

Why this cannot be a prompt rule: it already was one, and the trajectories
show the model stating the rule and then reasoning its way around it
("several rows match, so the question must be asking for the total"; "let me
check whether ``>`` or ``>=`` is right" followed by no such check). A
constraint the model can argue with is not a constraint, so these run in the
submission gate, where declining is not an option.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Optional

import sqlglot
from sqlglot import exp

logger = logging.getLogger(__name__)

# Words that license collapsing several rows into one number. If the question
# contains none of them, an aggregate is the model's idea rather than the
# question's.
_TOTAL_WORDS = re.compile(
    r"\b(total|totals|totalling|sum|summed|combined|altogether|overall|"
    r"aggregate|average|averages|mean|across all|in all|put together)\b",
    re.IGNORECASE,
)

# Only SUM and AVG. COUNT is the natural reading of "how many" and MAX/MIN of
# "the highest", so flagging those would fire constantly for no gain; SUM and
# AVG are the two that silently turn a list of answers into one answer.
_COLLAPSING_AGGREGATES = (exp.Sum, exp.Avg)

_STRICTER = {exp.GT: exp.GTE, exp.LT: exp.LTE}
_OP_TEXT = {exp.GT: (">", ">="), exp.LT: ("<", "<=")}

# Words that license counting distinct values rather than matching rows.
_DISTINCT_WORDS = re.compile(
    r"\b(distinct|different|differing|unique|uniquely|separate|individual|"
    r"how many kinds|how many types)\b",
    re.IGNORECASE,
)


@dataclass
class SimplerVariant:
    """A less elaborate reading of a submitted query, and what it would mean.

    ``message`` is only ever shown when running ``sql`` disagrees with the
    submission, so it is written as an observation about the difference
    rather than as a verdict about which one is right — the model is the one
    holding the question and the evidence, and sometimes the more elaborate
    reading is the correct one, so it has to stay free to say so.
    """

    label: str
    sql: str
    message: str


def _parse(sql: str, dialect: Optional[str]) -> Optional[exp.Expression]:
    try:
        return sqlglot.parse_one(sql, read=dialect or None)
    except Exception:  # noqa: BLE001 — an unparseable draft is the static check's problem
        return None


def _render(tree: exp.Expression, dialect: Optional[str]) -> str:
    return tree.sql(dialect=dialect or None)


# ----------------------------------------------------------------------
# 1. An aggregate the question never asked for
# ----------------------------------------------------------------------


def find_gratuitous_aggregation(
    sql: str, question: str, dialect: Optional[str]
) -> Optional[SimplerVariant]:
    """A ``SUM``/``AVG`` over a bare column that the question did not request.

    Requires no ``GROUP BY``: a grouped aggregate is reporting per-group
    figures, which is a different and usually deliberate shape. The
    un-aggregated variant is what the question's own wording points at when
    it names a column rather than a total.
    """
    tree = _parse(sql, dialect)
    if tree is None or not isinstance(tree, exp.Select):
        return None
    if tree.args.get("group") or _TOTAL_WORDS.search(question or ""):
        return None

    variant = tree.copy()
    collapsed: list[str] = []
    for node in list(variant.find_all(*_COLLAPSING_AGGREGATES)):
        inner = node.this
        if not isinstance(inner, exp.Column):
            continue
        # Only projections: an aggregate in HAVING or ORDER BY is doing
        # something the SELECT list is not, and stripping it changes the
        # question rather than simplifying the answer.
        if not any(node is p or node in list(p.find_all(type(node))) for p in variant.expressions):
            continue
        collapsed.append(f"{node.sql_name()}({inner.sql(dialect=dialect or None)})")
        node.replace(inner.copy())

    if not collapsed:
        return None

    return SimplerVariant(
        label="without the aggregate",
        sql=_render(variant, dialect),
        message=(
            f"You wrapped a column in {', '.join(collapsed)}, and that changed the "
            "answer: without it the query returns the rows below, with it they "
            "become a single number.\n\n"
            "The question does not say total, combined, altogether or average. "
            "Several matching rows are several answers, not one answer waiting "
            "to be summed — and a question phrased in the singular does not "
            "make them one; it just means whoever wrote it expected one match. "
            "Return the rows unless the question genuinely asks for a figure "
            "no single row holds.\n\n"
            "Return only the column the question asks about. Do not add a name "
            "or id column to label which row is which: that is a different "
            "answer from the one asked for, in the same way the aggregate was."
        ),
    )


# ----------------------------------------------------------------------
# 1b. Deduplicating a count that nothing asked to deduplicate
# ----------------------------------------------------------------------


def find_gratuitous_distinct_count(
    sql: str, question: str, dialect: Optional[str]
) -> Optional[SimplerVariant]:
    """``COUNT(DISTINCT x)`` in a query with one table and no joins.

    Deduplicating a count is how you undo the row multiplication a join
    causes. With no join there is no multiplication to undo, so the
    ``DISTINCT`` is not correcting an artefact — it is changing what is being
    counted, from "rows that matched the filter" to "values that appear among
    them". Those are different questions, and only one of them was asked.

    Left alone when the question says distinct, different or unique, which is
    the case where counting values really is the point.
    """
    tree = _parse(sql, dialect)
    if tree is None or not isinstance(tree, exp.Select):
        return None
    if list(tree.find_all(exp.Join)) or len(list(tree.find_all(exp.Table))) != 1:
        return None
    if _DISTINCT_WORDS.search(question or ""):
        return None

    variant = tree.copy()
    deduped: list[str] = []
    for projection in list(variant.expressions):
        for node in list(projection.walk()):
            if not isinstance(node, exp.Count) or not isinstance(
                node.this, exp.Distinct
            ):
                continue
            inner = node.this.expressions
            if len(inner) != 1 or not isinstance(inner[0], exp.Column):
                continue
            deduped.append(inner[0].sql(dialect=dialect or None))
            node.set("this", inner[0].copy())

    if not deduped:
        return None

    return SimplerVariant(
        label="counting matching rows instead of distinct values",
        sql=_render(variant, dialect),
        message=(
            f"You counted distinct values of {', '.join(deduped)}, and that "
            "changed the answer. There is no join here, so nothing was "
            "multiplying the rows and there is no duplication to undo: every "
            "row in this table is a record in its own right, and the filter "
            "already decided which ones match.\n\n"
            "`COUNT(DISTINCT ...)` answers how many different values appear, "
            "which is a different question from how many records match. The "
            "question does not say distinct, different or unique, so count "
            "the matching rows unless you can point at the words that ask "
            "you not to."
        ),
    )


# ----------------------------------------------------------------------
# 1c. Identical rows that nothing asked to repeat
# ----------------------------------------------------------------------


def find_undeduplicated_rows(
    sql: str, dialect: Optional[str], result: dict[str, Any]
) -> Optional[str]:
    """Identical multi-column rows in the submitted query's own result.

    Restricted to results with two or more columns, which is what makes this
    safe. One column repeating is very often real — two people can share a
    surname, and the same badge can be earned twice — so deduplicating it
    would destroy the answer. A whole row repeating across several columns is
    a different matter: distinct things do not agree on every field at once,
    so the repeat is the same entity arriving more than once through a
    one-to-many join.

    Takes the result the gate already had to execute, so it costs nothing.
    """
    tree = _parse(sql, dialect)
    if tree is None or not isinstance(tree, exp.Select):
        return None
    if tree.args.get("distinct") or tree.args.get("group"):
        return None
    # An aggregate or a window is already deciding its own grain.
    if any(
        isinstance(node, (exp.AggFunc, exp.Window))
        for projection in tree.expressions
        for node in projection.walk()
    ):
        return None
    if len(tree.expressions) < 2:
        return None
    if not result.get("ok"):
        return None

    rows = result.get("rows") or []
    if len(rows) < 2:
        return None
    keys = [
        tuple(str(v) for v in (row.values() if isinstance(row, dict) else row))
        for row in rows
    ]
    unique = len(set(keys))
    if unique == len(keys):
        return None

    return (
        f"Your query returns {len(keys)} rows, but only {unique} of them are "
        f"different — the rest are exact repeats across every column you "
        "selected.\n\n"
        "Two distinct things do not agree on every field at once, so those "
        "repeats are not separate answers: they are the same thing reached "
        "more than once, because something you joined to has several rows "
        "per match. Add `DISTINCT`, or project the column that actually "
        "tells the repeats apart if the question wants them listed "
        "separately."
    )


# ----------------------------------------------------------------------
# 2. A strict boundary where the boundary row may belong
# ----------------------------------------------------------------------


def find_strict_boundary(
    sql: str, dialect: Optional[str]
) -> Optional[SimplerVariant]:
    """A ``>``/``<`` between two columns, where ``>=``/``<=`` changes the answer.

    Restricted to column-to-column comparisons. ``amount > 3000`` comes
    straight from "greater than 3000" and is right to be strict, but a
    comparison between two of the schema's own date columns is the model's
    rendering of relational English like "after it opened", where whether the
    boundary row counts is exactly what is being guessed.
    """
    tree = _parse(sql, dialect)
    if tree is None:
        return None

    variant = tree.copy()
    relaxed: list[str] = []
    for op_type, inclusive_type in _STRICTER.items():
        for node in list(variant.find_all(op_type)):
            left, right = node.this, node.expression
            if not (isinstance(left, exp.Column) and isinstance(right, exp.Column)):
                continue
            strict_text, inclusive_text = _OP_TEXT[op_type]
            relaxed.append(
                f"{left.sql(dialect=dialect or None)} {strict_text} "
                f"{right.sql(dialect=dialect or None)}"
            )
            node.replace(
                inclusive_type(this=left.copy(), expression=right.copy())
            )

    if not relaxed:
        return None

    first = relaxed[0]
    return SimplerVariant(
        label="with the boundary included",
        sql=_render(variant, dialect),
        message=(
            f"You wrote `{first}`, which excludes the row where the two are "
            "equal, and that exclusion changed the answer — the inclusive "
            "form returns the rows below instead.\n\n"
            "Decide this from the question's wording rather than by habit, and "
            "note that a probe which already contains your own `"
            f"{_OP_TEXT[exp.GT][0]}` can never show you the row you excluded. "
            "If the boundary row belongs in the answer, use the inclusive form."
        ),
    )


# ----------------------------------------------------------------------
# 3. Joins that carry no weight
# ----------------------------------------------------------------------


def _column_tables(tree: exp.Select) -> tuple[set[str], set[str], bool]:
    """Which table aliases are used structurally, which contribute, and whether we can tell.

    A column inside a join condition only holds the query together; a column
    anywhere else is part of the answer. The third return value is False when
    any column is unqualified, because then the attribution is a guess and
    this whole check has to stand down.
    """
    structural: set[str] = set()
    contributing: set[str] = set()
    resolvable = True

    on_columns: set[int] = set()
    for join in tree.find_all(exp.Join):
        condition = join.args.get("on")
        if condition is not None:
            on_columns.update(id(c) for c in condition.find_all(exp.Column))
    # An equality between two columns in WHERE is a join written the old way.
    where = tree.args.get("where")
    if where is not None:
        for eq in where.find_all(exp.EQ):
            if isinstance(eq.this, exp.Column) and isinstance(
                eq.expression, exp.Column
            ):
                on_columns.update(id(c) for c in (eq.this, eq.expression))

    for column in tree.find_all(exp.Column):
        qualifier = column.table
        if not qualifier:
            resolvable = False
            continue
        if id(column) in on_columns:
            structural.add(qualifier.lower())
        else:
            contributing.add(qualifier.lower())

    return structural, contributing, resolvable


def _alias_map(tree: exp.Select) -> dict[str, str]:
    """Alias as written in the SQL to the real table name behind it.

    The join graph is keyed on aliases because that is what column
    qualifiers carry, but the relationship graph and the message both want
    the table's actual name.
    """
    names: dict[str, str] = {}
    for table in tree.find_all(exp.Table):
        alias = (table.alias_or_name or "").lower()
        if alias:
            names[alias] = table.name or alias
    return names


def _join_graph(tree: exp.Select) -> dict[str, set[str]]:
    """Table aliases, edged wherever a join equality connects two of them."""
    graph: dict[str, set[str]] = {}
    for table in tree.find_all(exp.Table):
        graph.setdefault((table.alias_or_name or "").lower(), set())

    pairs: list[tuple[exp.Column, exp.Column]] = []
    for join in tree.find_all(exp.Join):
        condition = join.args.get("on")
        if condition is not None:
            pairs.extend(
                (eq.this, eq.expression)
                for eq in condition.find_all(exp.EQ)
                if isinstance(eq.this, exp.Column)
                and isinstance(eq.expression, exp.Column)
            )
    where = tree.args.get("where")
    if where is not None:
        pairs.extend(
            (eq.this, eq.expression)
            for eq in where.find_all(exp.EQ)
            if isinstance(eq.this, exp.Column)
            and isinstance(eq.expression, exp.Column)
        )

    for left, right in pairs:
        a, b = (left.table or "").lower(), (right.table or "").lower()
        if a and b and a != b:
            graph.setdefault(a, set()).add(b)
            graph.setdefault(b, set()).add(a)
    return graph


def _connected(graph: dict[str, set[str]], nodes: set[str]) -> bool:
    """Whether every node in *nodes* is reachable from any one of them."""
    if len(nodes) <= 1:
        return True
    start = next(iter(nodes))
    seen = {start}
    stack = [start]
    while stack:
        for neighbour in graph.get(stack.pop(), set()):
            if neighbour not in seen:
                seen.add(neighbour)
                stack.append(neighbour)
    return nodes <= seen


def find_inert_joins(
    sql: str,
    dialect: Optional[str],
    has_direct_link: Optional[Callable[[str, str], bool]] = None,
) -> list[str]:
    """Tables the query joins but the answer never reads.

    Two shapes get reported, and both are about a path rather than a table:

    - the table is not even holding the query together, so dropping it leaves
      everything the answer *does* use still connected — typically a join the
      model invented from a word in the question that happened to match a
      table or a column value;
    - the table is on the path, but the tables at either end have a direct
      known relationship, so the query took the long way round to the same
      place and picked up the middle tables' row multiplicity on the way.

    Without the second case a legitimate bridge in a many-to-many would be
    indistinguishable from a detour, which is why ``has_direct_link`` — a
    question to the relationship graph — is what separates them.
    """
    tree = _parse(sql, dialect)
    if tree is None or not isinstance(tree, exp.Select):
        return []
    # A subquery has its own scope and its own join graph; attributing
    # columns across both is not worth the false positives.
    if len(list(tree.find_all(exp.Select))) > 1:
        return []

    # ``COUNT(*)`` names no column, but the joined rows themselves are the
    # answer, so every table in the query contributes and nothing here can be
    # inert. Without this, "how many members are in the College of
    # Engineering" reads as though ``member`` were along for the ride.
    if any(
        isinstance(node, exp.Star)
        for projection in tree.expressions
        for node in projection.walk()
    ):
        return []

    structural, contributing, resolvable = _column_tables(tree)
    if not resolvable or not contributing:
        return []

    graph = _join_graph(tree)
    names = _alias_map(tree)
    inert = sorted((structural | set(graph)) - contributing)
    inert = [t for t in inert if t]
    if not inert:
        return []

    findings: list[str] = []
    # A ``COUNT(DISTINCT ...)`` over a join is the author saying they knew the
    # join multiplies rows and chose to dedupe, which is what an intentional
    # "only count the ones that have one" existence filter looks like. Plain
    # ``SELECT DISTINCT`` is not the same signal — it is also just how you
    # answer a question that asks for *different* values.
    guarded = any(
        isinstance(node, exp.Count) and isinstance(node.this, exp.Distinct)
        for projection in tree.expressions
        for node in projection.walk()
    )
    for table in inert if not guarded else []:
        pruned = {
            node: {n for n in neighbours if n != table}
            for node, neighbours in graph.items()
            if node != table
        }
        if _connected(pruned, contributing - {table}):
            findings.append(
                f"- `{names.get(table, table)}` is joined but no column of it "
                "appears in your answer or in any filter the question asked "
                "for, and removing it would still leave the tables you do use "
                "connected. Either it is doing nothing, or it is quietly "
                "changing which rows survive — a join to a table with several "
                "matching rows duplicates the rows it joins to.\n"
                "  If you joined it on purpose, to require that a matching row "
                "exists, that is legitimate: say so and submit the same query "
                "again. If you cannot say what it is for, take it out."
            )

    if not findings and has_direct_link is not None and len(contributing) >= 2:
        ends = sorted(contributing)
        for i, left in enumerate(ends):
            for right in ends[i + 1 :]:
                if right in graph.get(left, set()):
                    continue
                left_name = names.get(left, left)
                right_name = names.get(right, right)
                try:
                    direct = has_direct_link(left_name, right_name)
                except Exception:  # noqa: BLE001 — a graph miss must not cost the question
                    logger.debug("Direct-link lookup failed", exc_info=True)
                    continue
                if direct:
                    detour = ", ".join(f"`{names.get(t, t)}`" for t in inert)
                    findings.append(
                        f"- Your answer only reads columns from `{left_name}` and "
                        f"`{right_name}`, and those two are directly related — but "
                        f"you connected them through {detour}, which reads "
                        "nothing. The longer path is not equivalent: it drops "
                        "rows that have no match in the middle tables and "
                        "duplicates rows that have several. Join them directly "
                        "unless the question specifically asks about what the "
                        "middle tables hold."
                    )
    return findings


def render_inert_joins(findings: list[str]) -> str:
    return (
        "Some tables in your query carry no weight:\n\n"
        + "\n".join(findings)
        + "\n\nEither use what they hold or take them out."
    )


# ----------------------------------------------------------------------
# 4. A relationship written as a filter instead of a join
# ----------------------------------------------------------------------


def _encloses(node: exp.Expression, ancestor: exp.Expression) -> bool:
    """Whether ``ancestor`` is the nearest SELECT above ``node``."""
    parent = node.parent
    while parent is not None:
        if isinstance(parent, exp.Select):
            return parent is ancestor
        parent = parent.parent
    return False


def _conjuncts(where: Optional[exp.Where]) -> list[exp.Expression]:
    if where is None:
        return []
    condition = where.this
    if isinstance(condition, exp.And):
        return list(condition.flatten())
    return [condition]


def _correlating_equality(
    conjuncts: list[exp.Expression], inner_alias: str, outer_aliases: set[str]
) -> Optional[exp.EQ]:
    """The single equality tying the subquery to the query around it."""
    found: list[exp.EQ] = []
    for conjunct in conjuncts:
        if not isinstance(conjunct, exp.EQ):
            continue
        left, right = conjunct.this, conjunct.expression
        if not (isinstance(left, exp.Column) and isinstance(right, exp.Column)):
            continue
        sides = {left.table.lower(), right.table.lower()}
        if inner_alias in sides and sides & outer_aliases:
            found.append(conjunct)
    return found[0] if len(found) == 1 else None


def _eligible_inner(
    sub: exp.Expression, only_inner: exp.Select, outer_aliases: set[str]
) -> Optional[exp.Table]:
    """The subquery's one source table, when the join form is derivable.

    Anything with its own grouping, ordering, limit or aggregate is making a
    decision of its own that a join would not preserve.
    """
    if not isinstance(sub, exp.Select) or sub is not only_inner:
        return None
    if any(
        sub.args.get(key)
        for key in ("group", "having", "limit", "distinct", "order", "qualify")
    ):
        return None
    if next(sub.find_all(exp.AggFunc), None) is not None:
        return None
    if next(sub.find_all(exp.Join), None) is not None:
        return None
    tables = list(sub.find_all(exp.Table))
    if len(tables) != 1:
        return None
    alias = (tables[0].alias_or_name or "").lower()
    if not alias or alias in outer_aliases:
        return None
    return tables[0]


def _residual_is_local(
    residual: list[exp.Expression], inner_alias: str
) -> bool:
    """Whether every leftover predicate only speaks about the joined table.

    One that reaches back out to the outer query is not a filter on the table
    being joined and would not mean the same thing once moved.
    """
    return not any(
        column.table.lower() not in (inner_alias, "")
        for conjunct in residual
        for column in conjunct.find_all(exp.Column)
    )


def find_relationship_as_filter(
    sql: str, dialect: Optional[str]
) -> Optional[SimplerVariant]:
    """A correlated ``EXISTS`` or ``IN`` standing where a join belongs.

    The two are not equivalent and the difference is row multiplicity: a join
    emits one row per match, ``EXISTS`` emits one row per outer row no matter
    how many times it matched. When the answer counts, sums or lists those
    rows, that is the whole answer.

    This is the only check here aimed at a shape the model reaches for
    *because* it can execute: ``EXISTS`` returns a plausible number on the
    first probe, every validity check passes, and nothing asks whether the
    matches were supposed to be counted more than once. Measured against the
    single-call baseline, the subquery form appeared on 13 paired questions
    and was right on one of them, while the baseline reached for it on none
    of 455 — so it is a shape the loop introduces, not one the model brings.

    Narrow by construction: only the mechanically derivable shape, where the
    subquery draws on one table, correlates through exactly one equality, and
    is the query's only subquery. ``NOT EXISTS`` is an anti-join and is left
    alone, as is anything whose join form would be a guess.
    """
    tree = _parse(sql, dialect)
    if not isinstance(tree, exp.Select):
        return None
    # The outer query and the subquery, and nothing else: a second subquery
    # means the rewrite below would have to reason about interactions it
    # cannot see.
    if len(list(tree.find_all(exp.Select))) != 2:
        return None

    variant = tree.copy()
    inner_selects = [s for s in variant.find_all(exp.Select) if s is not variant]
    if len(inner_selects) != 1:
        return None

    outer_aliases = {
        (table.alias_or_name or "").lower()
        for table in variant.find_all(exp.Table)
        if not _encloses(table, inner_selects[0])
        and table.parent_select is variant
    }
    outer_aliases.discard("")
    if not outer_aliases:
        return None

    for node in variant.find_all(exp.Exists, exp.In):
        if isinstance(node.parent, exp.Not):
            continue
        if not _encloses(node, variant):
            continue

        if isinstance(node, exp.Exists):
            sub = node.this
        else:
            query = node.args.get("query")
            if query is None or node.args.get("expressions"):
                continue
            sub = query.this if isinstance(query, exp.Subquery) else query

        inner = _eligible_inner(sub, inner_selects[0], outer_aliases)
        if inner is None:
            continue
        inner_alias = (inner.alias_or_name or "").lower()
        conjuncts = _conjuncts(sub.args.get("where"))

        if isinstance(node, exp.Exists):
            correlation = _correlating_equality(
                conjuncts, inner_alias, outer_aliases
            )
            if correlation is None:
                continue
            residual = [c for c in conjuncts if c is not correlation]
        else:
            # ``A.k IN (SELECT B.k ...)`` carries its correlation in the
            # comparison itself rather than in the subquery's WHERE.
            outer_column = node.this
            if not (
                isinstance(outer_column, exp.Column)
                and outer_column.table.lower() in outer_aliases
            ):
                continue
            if len(sub.expressions) != 1:
                continue
            inner_column = sub.expressions[0]
            if not isinstance(inner_column, exp.Column):
                continue
            correlation = exp.EQ(
                this=outer_column.copy(), expression=inner_column.copy()
            )
            residual = list(conjuncts)

        if not _residual_is_local(residual, inner_alias):
            continue

        joined = inner.copy()
        variant.args.setdefault("joins", []).append(
            exp.Join(this=joined, on=correlation.copy(), kind="INNER")
        )
        node.replace(
            exp.and_(*[c.copy() for c in residual]) if residual else exp.true()
        )
        tested = "EXISTS" if isinstance(node, exp.Exists) else "IN"
        return SimplerVariant(
            label="with the relationship joined instead of tested",
            sql=_render(variant, dialect),
            message=(
                f"You tested for a match in `{inner.name}` with `{tested}` rather "
                "than joining it, and the two return different answers.\n\n"
                f"`{tested}` asks only whether a match exists, so each row is "
                "counted once however many times it matched. A join emits one "
                "row per match. Neither is more correct in general, but they "
                "answer different questions, and a probe of your own query "
                "cannot show you the difference because it never produced the "
                "duplicate rows.\n\n"
                "If the question is about the matching rows themselves — how "
                "many there are, what they sum to, listing them — the join is "
                "what it asks for. Keep `EXISTS` only if the question is about "
                "which outer rows have a match at all."
            ),
        )

    return None


def rows_differ(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Whether two probe results disagree, by the same set equality that scores us.

    A variant that cannot execute counts as agreement: the submission is then
    the only query that works, and complaining about a comparison we failed
    to make would be noise.
    """
    if not left.get("ok") or not right.get("ok"):
        return False
    if left.get("row_count") != right.get("row_count"):
        return True

    def _key(rows: Any) -> list:
        out = []
        for row in rows or []:
            values = row.values() if isinstance(row, dict) else row
            out.append(tuple(str(v) for v in values))
        return sorted(out)

    return _key(left.get("rows")) != _key(right.get("rows"))


__all__ = [
    "SimplerVariant",
    "find_gratuitous_aggregation",
    "find_gratuitous_distinct_count",
    "find_strict_boundary",
    "find_relationship_as_filter",
    "find_inert_joins",
    "find_undeduplicated_rows",
    "render_inert_joins",
    "rows_differ",
]
