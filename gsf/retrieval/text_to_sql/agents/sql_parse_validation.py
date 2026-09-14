# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
SQL Validation Agent

This agent validates SQL queries before execution.
Checks for logical correctness, not just syntax.

Responsibilities:
- Validate SQL logic (not just syntax)
- Check for common mistakes (self-comparisons, incorrect filters, etc.)
- Handle text-based answers (skip validation)
- Store validation result in path_state

Design Decisions:
- Uses LLM to validate logical correctness
- Sets connection data based on retrieved tables
- Returns decision: "valid_sql" or "invalid_sql"
"""

import logging
import os
from typing import Any, Dict

import sqlglot
from sqlglot import expressions as exp

from gsf.catalog.sql_parse import parse_query_single
from gsf.dal.datasources import find_table_key_columns
from gsf.retrieval.text_to_sql.base import BaseAgent
from gsf.retrieval.text_to_sql.connector_routing import resolve_connector_from_tables
from gsf.retrieval.text_to_sql.state import AgentState
from gsf.retrieval.data_access.custom_analyses import get_custom_analyses_ids
from gsf.retrieval.data_access.graph_schemas import (
    fetch_all_schema_ids,
    get_schemas_by_ids,
)

logger = logging.getLogger(__name__)

# A connector's ``dialect`` goes straight to sqlglot here. Do not reintroduce a
# name map: ``connectors/tests/test_registry_dialects.py`` already guarantees
# every connector reports a name sqlglot knows, so a map can only ever be an
# identity — one that silently yields ``None`` for anything it has not been
# taught, which makes sqlglot re-render in its default dialect and quote
# identifiers with ``"``. Spark reads that as a string literal, so the SQL stops
# parsing. Qualification and dialect are the connector's to decide.
_TRUTHY = {"1", "true", "yes", "on"}


def _vacuous_group_by_check_enabled() -> bool:
    """Whether :func:`detect_vacuous_group_by` is wired in.

    Off by default: the uniqueness inference it relies on (PK/FK metadata,
    observed-data profiling) can misfire on a dataset where that metadata is
    incomplete or misleading, incorrectly rejecting a correct query and
    burning a reconstruction cycle. Opt-in via ``DETECT_VACUOUS_GROUP_BY``.
    """
    return os.environ.get("DETECT_VACUOUS_GROUP_BY", "").strip().lower() in _TRUTHY


def _unwrap_projection(e: exp.Expression) -> exp.Expression:
    """Strip an alias wrapper so ``NULL AS x`` is seen as ``NULL``."""
    return e.this if isinstance(e, exp.Alias) else e


def _is_always_false(cond: exp.Expression | None) -> bool:
    """True for constant-false predicates like ``1=0``, ``0=1``, ``FALSE``."""
    if cond is None:
        return False
    if isinstance(cond, exp.Boolean):
        return cond.this is False
    if isinstance(cond, exp.EQ):
        left, right = cond.left, cond.right
        if (
            isinstance(left, exp.Literal)
            and isinstance(right, exp.Literal)
            and left.is_number
            and right.is_number
        ):
            return left.name != right.name
    return False


def detect_degenerate_sql(sql: str, dialect: str | None = None) -> str:
    """Return a human-readable reason when *sql* is a placeholder/no-op query.

    Flags queries that parse and execute fine but can never answer the question:
    ``SELECT NULL`` / constant-only projections, always-false ``WHERE`` clauses
    (``1=0``), and ``LIMIT 0``. Returns ``""`` when the SQL looks like real work.

    Inspects only the OUTERMOST SELECT, so legitimate ``EXISTS (SELECT 1 ...)``
    subqueries are not flagged.
    """
    if not sql or not sql.strip():
        return "the generated SQL is empty"

    try:
        parsed = sqlglot.parse_one(sql, read=dialect or None)
    except Exception:
        # Unparseable here → let the normal parse validator handle it.
        return ""
    if parsed is None:
        return ""

    select = parsed if isinstance(parsed, exp.Select) else parsed.find(exp.Select)
    if select is None:
        return ""

    projections = list(select.expressions or [])
    if projections and all(
        isinstance(_unwrap_projection(p), (exp.Null, exp.Literal, exp.Boolean))
        for p in projections
    ):
        return (
            "the query only selects constant/NULL values instead of real data "
            "from the tables (e.g. SELECT NULL)"
        )

    where = select.args.get("where")
    if where is not None and _is_always_false(where.this):
        return (
            "the query has an always-false WHERE condition (e.g. 1=0), so it can "
            "never return any rows"
        )

    limit = select.args.get("limit")
    if limit is not None:
        limit_expr = getattr(limit, "expression", None)
        if (
            isinstance(limit_expr, exp.Literal)
            and limit_expr.is_number
            and limit_expr.name == "0"
        ):
            return "the query uses LIMIT 0, so it always returns no rows"

    return ""


def quote_known_mixed_case_identifiers(
    sql: str, relevant_tables: list[dict] | None, dialect: str | None
) -> str:
    """Deterministically quote every table/column reference in *sql* that
    exactly matches a known mixed-case identifier from *relevant_tables*, so
    Postgres (and any other dialect with the same unquoted-lowercasing rule)
    doesn't silently fold e.g. ``WeatherAndStructure`` to
    ``weatherandstructure`` and fail with "relation does not exist".

    The model is shown these exact names in the AVAILABLE TABLES/COLUMNS
    prompt section and told, via the dialect rules, to quote anything with
    an uppercase letter — but that instruction competes against its
    dominant prior from the mostly-lowercase rest of the schema corpus and
    loses often enough to be worth enforcing deterministically, rather than
    relying on compliance or on a failed execution + LLM reconstruction
    round-trip to catch it after the fact (see ``sql_reconstruction.py``,
    which still exists as a fallback for anything this misses — e.g. an
    identifier not in ``relevant_tables`` at all).

    Only ever *adds* quoting to an identifier that already exactly matches a
    known real name — never renames or case-corrects a wrong reference; a
    near-miss like ``extTempc`` (wrong case, not an exact match to the real
    ``extTempC``) is left untouched rather than guessed at, since that's a
    different problem (a genuinely wrong column) this function has no basis
    to fix.

    No-op (returns *sql* unchanged) when *relevant_tables* is empty, when
    none of its table/column names contain an uppercase letter (the common
    case — checked before any parsing), or when *sql* doesn't parse.
    """
    if not relevant_tables or not sql or not sql.strip():
        return sql

    mixed_case_tables: dict[str, str] = {}
    mixed_case_columns: dict[str, str] = {}
    for table in relevant_tables:
        name = table.get("name") if isinstance(table, dict) else None
        if name and any(ch.isupper() for ch in name):
            mixed_case_tables[name.lower()] = name
        columns = table.get("columns") if isinstance(table, dict) else None
        if isinstance(columns, list):
            for col in columns:
                col_name = col.get("name") if isinstance(col, dict) else None
                if col_name and any(ch.isupper() for ch in col_name):
                    mixed_case_columns[col_name.lower()] = col_name
    if not mixed_case_tables and not mixed_case_columns:
        return sql

    try:
        tree = sqlglot.parse_one(sql, read=dialect or None)
    except Exception:
        return sql
    if tree is None:
        return sql

    changed = False
    for table_node in tree.find_all(exp.Table):
        ident = table_node.this
        if not isinstance(ident, exp.Identifier) or ident.args.get("quoted"):
            continue
        real = mixed_case_tables.get((ident.this or "").lower())
        if real and ident.this == real:
            table_node.set("this", exp.to_identifier(real, quoted=True))
            changed = True

    for col_node in tree.find_all(exp.Column):
        ident = col_node.this
        if not isinstance(ident, exp.Identifier) or ident.args.get("quoted"):
            continue
        real = mixed_case_columns.get((ident.this or "").lower())
        if real and ident.this == real:
            col_node.set("this", exp.to_identifier(real, quoted=True))
            changed = True

    if not changed:
        return sql
    return tree.sql(dialect=dialect or None)


def detect_vacuous_group_by(
    sql: str, dialect: str | None, database_name: str | None
) -> str:
    """Return a human-readable reason when *sql* groups/partitions by a column
    that's already unique per row — e.g. ``GROUP BY`` (or a window function's
    ``PARTITION BY``) on the FROM table's own primary key, or on a column
    known unique from observed-data profiling. Since such a column has
    exactly one distinct value per row, every group/partition contains
    exactly one row and any aggregate over it (``AVG``, ``SUM``, a window
    ``COUNT`` etc.) is a silent no-op — it parses and executes fine but never
    computes what "average per <thing>" actually means.

    Scoped to the narrow, always-true case only: a single-table SELECT block
    (no JOIN *within that block*) grouping/partitioning by that same table's
    own key. A JOIN can legitimately re-introduce multiple rows per key (e.g.
    aggregating a child table's rows per parent id), so this intentionally
    does not flag a block once a JOIN is present in it — that needs
    join-cardinality reasoning this check doesn't attempt.

    Checked on *every* SELECT block in the parsed tree — the outermost query,
    every subquery, every CTE — not just the outermost one. A subquery that
    itself vacuously groups a table by its own key is just as much a no-op as
    if it were written at the top level; wrapping it in an outer JOIN back to
    the same table (e.g. ``t JOIN (SELECT k, AVG(x) FROM t GROUP BY k) s ON
    t.k = s.k``) makes the *outer* query's FROM/JOIN shape look fine while
    the actual aggregation inside the subquery is still averaging exactly one
    row per group — a real pattern seen in practice once reconstruction was
    pushed away from the bare single-table form.

    Also catches the same no-op spelled without ``GROUP BY``/``PARTITION BY``
    at all: a correlated (or plain) scalar-subquery aggregate whose ``WHERE``
    filters the table down by its own unique column — e.g. ``(SELECT
    AVG(s.x) FROM t s WHERE s.pk = rp.pk)``. Filtering to an exact match on a
    unique column matches at most one row, so the aggregate is exactly as
    vacuous as ``GROUP BY`` on that column, just spelled differently — seen
    in practice as the next thing reconstruction tried once the GROUP-BY
    form got rejected.

    Off by default — see :func:`_vacuous_group_by_check_enabled`; opt-in via
    ``DETECT_VACUOUS_GROUP_BY``. Both call sites (this module's
    ``SQLValidationAgent`` and ``join_path_check.py``'s self-applied
    bridge-fix guard) already treat ``""`` as "nothing wrong", so disabling
    this check is a no-op change to their control flow.

    Returns ``""`` when nothing looks wrong, including whenever the SQL
    doesn't contain "group by"/"partition by"/an aggregate function call at
    all (checked before any parsing, so the common case costs nothing) or
    isn't parseable (left to the normal parse validator).
    """
    if not _vacuous_group_by_check_enabled():
        return ""

    sql_lower = (sql or "").lower()
    _CLUE_WORDS = ("group by", "partition by", "avg(", "sum(", "count(", "min(", "max(")
    if not any(kw in sql_lower for kw in _CLUE_WORDS):
        return ""

    try:
        parsed = sqlglot.parse_one(sql, read=dialect or None)
    except Exception:
        return ""
    if parsed is None:
        return ""

    for select in parsed.find_all(exp.Select):
        reason = _check_select_block_vacuous(select, database_name)
        if reason:
            return reason
    return ""


def _check_select_block_vacuous(select: exp.Select, database_name: str | None) -> str:
    """Single-SELECT-block half of :func:`detect_vacuous_group_by` — see there
    for the full rationale. Returns ``""`` when this block looks fine."""
    # Only the single-table, no-JOIN case — see docstring.
    if select.args.get("joins"):
        return ""
    # exp.From, not select.args.get("from") — sqlglot's internal arg key for
    # this has changed across versions ("from" vs "from_"); searching by node
    # type is stable regardless. .find() (not find_all/recursion into nested
    # selects) stays scoped to this block's own FROM since a nested SELECT
    # would be inside a subquery, not a sibling of this block's FROM clause.
    from_clause = select.find(exp.From)
    table_expr = from_clause.this if from_clause is not None else None
    if not isinstance(table_expr, exp.Table) or not table_expr.name:
        return ""
    table_name = table_expr.name
    table_alias = table_expr.alias_or_name

    keys = find_table_key_columns(table_name, database_name)
    # ``unique`` entries are each independently unique per row, so any single
    # one of them appearing in the grouping/filter set is already vacuous.
    # ``pk`` is not: a multi-column (composite) primary key is only unique
    # as the *combination* of all its columns — grouping by just one member
    # (e.g. the FK half of a (race, driver, lap) key) still leaves many rows
    # per group and is a perfectly real aggregation, not a no-op. So ``pk``
    # only counts as a hit once every one of its columns is covered by the
    # grouping/filter set (checked separately below), never on a partial
    # overlap the way ``unique`` is.
    unique_cols = {c.lower() for c in keys["unique"]}
    pk_cols = {c.lower() for c in keys["pk"]}
    if not unique_cols and not pk_cols:
        return ""

    group = select.args.get("group")

    # 1. GROUP BY / a window function's PARTITION BY on a unique column.
    grouping_cols: set[str] = set()
    if group is not None:
        for e in group.expressions or []:
            col = e.this if isinstance(e, exp.Ordered) else e
            if isinstance(col, exp.Column):
                grouping_cols.add(col.name.lower())
    for window in select.find_all(exp.Window):
        for col in window.args.get("partition_by") or []:
            if isinstance(col, exp.Column):
                grouping_cols.add(col.name.lower())
    hit = grouping_cols & unique_cols
    if hit:
        return _vacuous_reason(
            next(iter(hit)),
            table_name,
            table_alias,
            verb="groups/partitions by",
            extra="even if it is subsequently joined back to the same table, "
            "since the aggregation was already trivial before that join. ",
        )
    if pk_cols and pk_cols <= grouping_cols:
        pk_verb = (
            "groups/partitions by the full composite primary key"
            if len(pk_cols) > 1
            else "groups/partitions by"
        )
        return _vacuous_reason(
            ", ".join(sorted(pk_cols)),
            table_name,
            table_alias,
            verb=pk_verb,
            extra="even if it is subsequently joined back to the same table, "
            "since the aggregation was already trivial before that join. ",
        )

    # 2. A WHERE-filtered aggregate with no GROUP BY at all — the same no-op,
    # spelled as a (typically correlated) scalar subquery instead, e.g.
    # ``(SELECT AVG(s.x) FROM t s WHERE s.pk = rp.pk)``. Filtering to an
    # exact match on a unique column matches at most one row, so wrapping it
    # in AVG/SUM/etc. is exactly as vacuous as GROUP BY on that column.
    # Scoped to this block's own SELECT list (not select.find(), which would
    # also match an AggFunc buried in a WHERE subquery — e.g. `WHERE pk =
    # (SELECT MAX(pk) FROM t)`, the standard "latest row" idiom, which has no
    # aggregate over the outer row set at all and would otherwise be a false
    # positive here).
    if group is None and any(
        e.find(exp.AggFunc) is not None for e in select.expressions or []
    ):
        where_cols: set[str] = set()
        where = select.args.get("where")
        if where is not None:
            for eq in where.find_all(exp.EQ):
                for side in (eq.left, eq.right):
                    if isinstance(side, exp.Column):
                        where_cols.add(side.name.lower())
        hit = where_cols & unique_cols
        if hit:
            return _vacuous_reason(
                next(iter(hit)),
                table_name,
                table_alias,
                verb='filters by ("WHERE") an equality on',
                extra="",
            )
        if pk_cols and pk_cols <= where_cols:
            pk_verb = (
                'filters by ("WHERE") an equality on the full composite primary key'
                if len(pk_cols) > 1
                else 'filters by ("WHERE") an equality on'
            )
            return _vacuous_reason(
                ", ".join(sorted(pk_cols)),
                table_name,
                table_alias,
                verb=pk_verb,
                extra="",
            )

    return ""


def _vacuous_reason(
    culprit: str, table_name: str, table_alias: str, *, verb: str, extra: str
) -> str:
    return (
        f'the query {verb} "{culprit}", which is already unique '
        f'per row in "{table_name}" (its primary key or a column confirmed '
        f"unique from the data) — with no JOIN bringing in additional rows in "
        f"that part of the query, every group/match has exactly one row, "
        f"so any aggregate over it (AVG, SUM, a window function, etc.) is a "
        f'no-op and does not compute a real "per group" result — {extra}'
        f"Group/aggregate by the actual dimension the question is asking to "
        f"aggregate over instead (e.g. a foreign key or category column "
        f"shared by multiple rows), or remove the grouping/aggregation if "
        f'the question wants one row per "{table_alias or table_name}" record.'
    )


def _select_scope_table_nodes(
    select: exp.Select,
) -> tuple[list[exp.Table], dict[str, exp.Table]]:
    """Table nodes introduced directly in *select*'s own FROM/JOIN.

    Deliberately does not descend into a derived subquery's own FROM (e.g.
    ``JOIN (SELECT ...) s``) — that subquery is a separate SELECT block with
    its own scope, checked independently when :func:`detect_missing_aggregation`
    visits it via ``find_all(exp.Select)``. Mixing the two would misattribute
    an inner query's tables to this block's join graph.
    """
    nodes: list[exp.Table] = []
    # exp.From via .find(), not select.args.get("from") — see
    # _check_select_block_vacuous's comment: sqlglot's internal arg key for
    # this has changed across versions ("from" vs "from_"); searching by node
    # type is stable regardless. .find() (not find_all) stays scoped to this
    # block's own FROM since a nested SELECT lives inside a subquery, not a
    # sibling of this block's FROM clause.
    from_clause = select.find(exp.From)
    if from_clause is not None and isinstance(from_clause.this, exp.Table):
        nodes.append(from_clause.this)
    for join in select.args.get("joins") or []:
        if isinstance(join.this, exp.Table):
            nodes.append(join.this)
    by_key: dict[str, exp.Table] = {}
    for t in nodes:
        if t.name:
            by_key.setdefault(t.name.lower(), t)
        alias = t.alias
        if alias:
            by_key[alias.lower()] = t
    return nodes, by_key


def _resolve_column_table(
    col: exp.Column, by_key: dict[str, exp.Table]
) -> exp.Table | None:
    """The table node *col* belongs to, resolved only from its own qualifier.

    Returns ``None`` for an unqualified column in a multi-table join rather
    than guessing among candidates — misattributing a column to the wrong
    table here would over- or under-flag, so an ambiguous column is simply
    left unchecked.
    """
    qualifier = (col.table or "").lower()
    return by_key.get(qualifier) if qualifier else None


def _many_side_tables(
    select: exp.Select,
    by_key: dict[str, exp.Table],
    database_name: str | None,
) -> set[str]:
    """Lowercased names of tables that are the FK/"many" side of some
    ``JOIN ... ON`` equality in *select*, determined from PK/unique key
    metadata rather than guessed. An edge only counts when exactly one side
    is known unique on its own table and the other is known not to be —
    when neither or both sides are unique (or key metadata is missing for
    one), the edge is ambiguous and contributes nothing, to keep this
    conservative rather than risk flagging a legitimate 1:1 join.
    """
    many: set[str] = set()
    for join in select.args.get("joins") or []:
        on = join.args.get("on")
        if on is None:
            continue
        for eq in on.find_all(exp.EQ):
            left, right = eq.left, eq.right
            if not (isinstance(left, exp.Column) and isinstance(right, exp.Column)):
                continue
            table_left = _resolve_column_table(left, by_key)
            table_right = _resolve_column_table(right, by_key)
            if table_left is None or table_right is None:
                continue
            if table_left.name.lower() == table_right.name.lower():
                continue  # self-join — not this check's job
            keys_left = find_table_key_columns(table_left.name, database_name)
            keys_right = find_table_key_columns(table_right.name, database_name)
            unique_left = left.name.lower() in (keys_left["pk"] + keys_left["unique"])
            unique_right = right.name.lower() in (
                keys_right["pk"] + keys_right["unique"]
            )
            if unique_left and not unique_right:
                many.add(table_right.name.lower())
            elif unique_right and not unique_left:
                many.add(table_left.name.lower())
    return many


def detect_missing_aggregation(
    sql: str, dialect: str | None, database_name: str | None
) -> str:
    """Return a human-readable reason when *sql* groups by one table's key
    but also selects a *different*, many-side-joined table's column raw,
    outside any aggregate function.

    Mirror-image failure to :func:`detect_vacuous_group_by`: instead of an
    aggregate that's a no-op because nothing varies within a group, this is
    a column that's silently *undefined* within a group because a joined
    child table can contribute more than one row per group — e.g.
    ``GROUP BY fan.id`` while also directly selecting
    ``interactions.watch_hrs`` when a fan can have many interaction rows.
    Every SQL engine this pipeline targets (Postgres included) accepts this
    without error whenever it's spelled without an explicit ``GROUP BY`` at
    all; it's only actually an error when a ``GROUP BY`` is present and the
    ungrouped column isn't functionally dependent on it — which is exactly
    the case this checks. It parses and executes fine, silently picking an
    arbitrary row's value instead of the intended per-group summary.

    Scoped to the narrow, always-true case: a SELECT block with an explicit
    ``GROUP BY`` on a bare column, at least one ``JOIN ... ON`` in that same
    block, and a join equality where PK/unique key metadata *confirms* one
    side is the "many" side relative to the other (ambiguous cardinality is
    left unflagged — see :func:`_many_side_tables`). Does not attempt the
    much harder "no GROUP BY at all" case, where the intended output grain
    isn't recoverable from the SQL's own structure — that needs a separate
    signal this check doesn't have. Only the SELECT list is inspected, not
    ORDER BY/HAVING.

    Returns ``""`` when nothing looks wrong, including whenever the SQL has
    no "group by" at all (checked before parsing) or isn't parseable.
    """
    sql_lower = (sql or "").lower()
    if "group by" not in sql_lower:
        return ""

    try:
        parsed = sqlglot.parse_one(sql, read=dialect or None)
    except Exception:
        return ""
    if parsed is None:
        return ""

    for select in parsed.find_all(exp.Select):
        reason = _check_select_block_missing_agg(select, database_name)
        if reason:
            return reason
    return ""


def _group_expr_covers(col: exp.Column, group_exprs: list[exp.Expression]) -> bool:
    """True when *col* lives inside a projection subtree that's the same
    expression (by rendered SQL) as one of the GROUP BY entries — e.g. the
    same ``CASE WHEN ...`` or ``SPLIT_PART(...)`` call is both selected and
    grouped by verbatim. GROUP BY on that expression pins one value of it
    per group, so a raw many-side column inside it is exactly as safe as a
    bare grouped column — the check just needs to recognize the expression
    form too, not only ``GROUP BY table.col``.

    Structural match only (the two sides must render to identical SQL); this
    does not attempt functional-dependency reasoning about two *different*
    expressions that happen to be equivalent.
    """
    if not group_exprs:
        return False
    group_sqls = {g.sql() for g in group_exprs}
    ancestor = col.parent
    while ancestor is not None:
        if ancestor.sql() in group_sqls:
            return True
        ancestor = ancestor.parent
    return False


def _check_select_block_missing_agg(
    select: exp.Select, database_name: str | None
) -> str:
    """Single-SELECT-block half of :func:`detect_missing_aggregation`."""
    group = select.args.get("group")
    if group is None or not select.args.get("joins"):
        return ""

    all_nodes, by_key = _select_scope_table_nodes(select)
    if len(all_nodes) < 2:
        return ""

    grouping_pairs: set[tuple[str, str]] = set()
    grouping_tables: set[str] = set()
    group_exprs: list[exp.Expression] = []
    for e in group.expressions or []:
        ge = e.this if isinstance(e, exp.Ordered) else e
        group_exprs.append(ge)
        if not isinstance(ge, exp.Column):
            continue
        table = _resolve_column_table(ge, by_key)
        if table is None:
            continue
        grouping_pairs.add((table.name.lower(), ge.name.lower()))
        grouping_tables.add(table.name.lower())
    if not grouping_tables:
        # GROUP BY on an expression or an unresolvable column — can't
        # safely establish an anchor, so nothing to check against.
        return ""

    many_tables = _many_side_tables(select, by_key, database_name) - grouping_tables
    if not many_tables:
        return ""

    for proj in select.expressions or []:
        for col in proj.find_all(exp.Column):
            if col.find_ancestor(exp.AggFunc) is not None:
                continue  # aggregated — fine regardless of which table it's from
            table = _resolve_column_table(col, by_key)
            if table is None:
                continue
            table_name = table.name.lower()
            if table_name not in many_tables:
                continue
            if (table_name, col.name.lower()) in grouping_pairs:
                continue
            if _group_expr_covers(col, group_exprs):
                continue
            anchor_desc = ", ".join(f'"{t}"' for t in sorted(grouping_tables))
            culprit = f"{table.alias_or_name}.{col.name}"
            return (
                f'You grouped by {anchor_desc}, but "{table.name}" is joined '
                f'many-to-one against that key and "{culprit}" is selected raw — '
                f"it can have more than one value per group. Wrap it in the "
                f"aggregate that matches what the question asks for that field "
                f"(e.g. AVG, SUM, MAX), or add it to GROUP BY only if it is truly "
                f"one value per group. Change only {culprit}; keep every other "
                f"join, filter, and column exactly as they are."
            )
    return ""


class SQLValidationAgent(BaseAgent):
    """
    Agent that validates SQL queries before execution.

    This agent performs logical validation of SQL queries, checking for
    common mistakes like self-comparisons, incorrect filters, etc.

    Input Requirements:
    - path_state["sql_generation_result"]: SQL response to validate
    - path_state["relevant_tables"]: Relevant tables used

    Output:
    - path_state["sql_response_from_db"]: None (will be set after execution)
    - path_state["sql_columns"]: Column IDs from SQL
    - path_state["custom_analyses_used"]: Semantic entity IDs used
    - decision: "valid_sql" or "invalid_sql"
    """

    def __init__(self):
        super().__init__("sql_validation")

    def validate_input(self, state: AgentState) -> bool:
        """Validate that SQL response is available."""
        path_state = state.get("path_state", {})
        if state.get("decision") == "unconstructable":
            # Skip validation if SQL couldn't be constructed
            return False
        if not path_state.get("sql_generation_result"):
            self.logger.warning("No SQL response found for validation")
            return False
        return True

    def execute(self, state: AgentState) -> Dict[str, Any]:
        """
        Validate SQL query.

        Performs logical validation using LLM and query_validation function.
        Sets connection data and extracts columns from SQL.

        Args:
            state: Current agent state

        Returns:
            Dictionary with:
            - path_state: Contains validation result and extracted data
            - decision: "valid_sql" or "invalid_sql"
        """
        path_state = state.get("path_state", {})
        response = path_state.get("sql_generation_result")
        connectors = state.get("connectors") or []
        dialects = [c.dialect for c in connectors if getattr(c, "dialect", None)]
        connector = resolve_connector_from_tables(
            path_state.get("relevant_tables"), connectors
        )
        schemas_ids = fetch_all_schema_ids()
        schemas = get_schemas_by_ids(schemas_ids)
        degenerate_dialect = getattr(connector, "dialect", None)

        # Every constructed/reconstructed SQL passes through this node before
        # execution (see the graph: construct_sql_from_candidates,
        # construct_sql_not_from_snippets, and reconstruct_sql all route
        # here), so this is the single choke point to deterministically fix
        # up unquoted mixed-case identifiers regardless of which path
        # produced the SQL — see quote_known_mixed_case_identifiers for why
        # this can't just be left to the prompt/model.
        quoted_sql = quote_known_mixed_case_identifiers(
            response.sql_code, path_state.get("relevant_tables"), degenerate_dialect
        )
        if quoted_sql != response.sql_code:
            self.logger.info(
                "Quoted known mixed-case identifier(s) in the generated SQL "
                "before validation"
            )
            response.sql_code = quoted_sql
            path_state["sql_generation_result"] = response

        validation_result = self._sql_parse_validation(
            schemas, response.sql_code, dialects
        )

        if validation_result.get("error"):
            error_msg = validation_result["error"]
            self.logger.info(f"SQL validation failed: {error_msg}")
            path_state["error"] = error_msg
            return {
                "decision": "invalid_sql",
                "path_state": path_state,
            }

        degenerate_reason = detect_degenerate_sql(response.sql_code, degenerate_dialect)

        if degenerate_reason:
            self.logger.info("Degenerate SQL rejected: %s", degenerate_reason)
            path_state["error"] = (
                f"The generated SQL is a placeholder that does not answer the "
                f"question: {degenerate_reason}. Rewrite a real query that selects "
                f"the requested data from the available tables. Do NOT use SELECT "
                f"NULL, constant-only projections, always-false conditions such as "
                f"WHERE 1=0, or LIMIT 0."
            )
            return {
                "decision": "invalid_sql",
                "path_state": path_state,
            }

        vacuous_reason = detect_vacuous_group_by(
            response.sql_code, degenerate_dialect, path_state.get("target_db")
        )
        if vacuous_reason:
            self.logger.info(
                "Vacuous GROUP BY/PARTITION BY rejected: %s", vacuous_reason
            )
            path_state["error"] = (
                f"The generated SQL's aggregation is a no-op: {vacuous_reason}"
            )
            # Deterministic, self-contained diagnosis — never a missing_data
            # situation (no new table/column could fix a query that's grouping
            # by a column already unique per row). Skip reconstruction's LLM
            # error-classification call so it can't misread "join"/"aggregate"
            # in the message and go searching for tables that don't help here.
            path_state["error_known_fixable"] = True
            return {
                "decision": "invalid_sql",
                "path_state": path_state,
            }

        missing_agg_reason = detect_missing_aggregation(
            response.sql_code, degenerate_dialect, path_state.get("target_db")
        )
        if missing_agg_reason:
            self.logger.info("Missing aggregation rejected: %s", missing_agg_reason)
            path_state["error"] = (
                f"The generated SQL's GROUP BY is unsafe: {missing_agg_reason}"
            )
            # Same rationale as the vacuous-GROUP-BY branch above: deterministic
            # and self-contained — no missing table/column would fix this, so
            # skip reconstruction's LLM error-classification call.
            path_state["error_known_fixable"] = True
            return {
                "decision": "invalid_sql",
                "path_state": path_state,
            }
        self.logger.info(
            "SQL passed static checks: parse, degenerate, vacuous-aggregation, "
            "missing-aggregation"
        )

        sql_columns = validation_result.get("sql_columns") or []
        custom_analyses_used = []
        if hasattr(response, "custom_analyses_used"):
            custom_analyses_used = get_custom_analyses_ids(
                response.custom_analyses_used
            )

        # Store connection_data in the format expected by execute_sql_query
        # execute_sql_query expects connections as a list
        updated_path_state = {
            **path_state,
            "sql_response_from_db": None,  # Will be set after execution
            "sql_columns": sql_columns,
            "custom_analyses_used": custom_analyses_used,
            "sql_code": response.sql_code,  # Store SQL code for execution
        }

        self.logger.info(f"SQL validation passed, columns: {len(sql_columns)}")

        return {
            "decision": "valid_sql",
            "path_state": updated_path_state,
        }

    @staticmethod
    def _sql_parse_validation(schemas, sql: str, dialects: list[str]) -> dict:
        result: dict = {}
        try:
            parse_query_single(
                sql=sql,
                schemas=schemas,
                dialects=dialects,
            )
            result["success"] = True
        except Exception as error:
            result.update({"error": str(error), "another_try": 1})
        return result
