# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Literal-vs-database value check for the repair path.

This makes **no assumptions about the incoming database**: it does not read any
declared schema, column type, or catalog metadata. Everything is discovered at
runtime from (1) the generated SQL itself and (2) live read-only probes:

- the table/column a filter refers to is taken from the SQL's own AST
  (``sqlglot``) — the query already executed against the real DB, so those
  identifiers are guaranteed valid;
- whether a column is "categorical" is decided purely by probing
  ``SELECT DISTINCT col`` and seeing if the value set is small — no reliance on
  a declared ``TEXT``/``VARCHAR`` type;
- a literal is only flagged when it is a *near-miss* of a real value (wrong
  case / spelling / spacing). A literal with no close real counterpart is left
  alone, because an empty result can be the correct answer.

A second, independent check targets numeric comparison predicates (``>``,
``>=``, ``<``, ``<=``): a literal that falls far outside the column's real
``MIN``/``MAX`` range, but whose value ``/100`` (or ``*100``) *does* fall
inside it, is flagged as a likely percent-vs-fraction scale mismatch — the
literal usually came from the user phrasing a threshold as "60%" while the
column is stored as a 0-1 fraction (or vice versa). Unlike the string check,
this is a heuristic, not a certainty, so it is reported as evidence (the
real range plus the rescaled candidate) for reconstruction to weigh, not
auto-applied.
"""

from __future__ import annotations

import difflib
import logging
import re
from typing import Any, Optional

import sqlglot
from sqlglot import exp

from gsf.retrieval.text_to_sql.db_probe.config import DB_PROBE_LOW_CARD_THRESHOLD
from gsf.retrieval.text_to_sql.db_probe.executor import ProbeExecutor

logger = logging.getLogger(__name__)


def _first_value(row: dict) -> Any:
    """Return the first value of a single-row result dict."""
    if not isinstance(row, dict) or not row:
        return None
    return next(iter(row.values()))


# How similar a used literal must be to a real value to count as a fixable
# near-miss (difflib ratio, 0..1). Below this we treat the value as legitimately
# absent and do not repair.
_CLOSE_MATCH_CUTOFF = 0.6

_SQLGLOT_DIALECTS = {
    "postgresql": "postgres",
    "postgres": "postgres",
    "sqlite": "sqlite",
    "duckdb": "duckdb",
    "snowflake": "snowflake",
    "mysql": "mysql",
    "bigquery": "bigquery",
}


def _sqlglot_dialect(dialect: Optional[str]) -> Optional[str]:
    return _SQLGLOT_DIALECTS.get((dialect or "").lower())


def _text_cast(column_sql: str, dialect: Optional[str]) -> str:
    """*column_sql* wrapped in an explicit text cast, when the dialect needs one.

    Postgres-only: a Postgres ``ENUM`` column has no ``LOWER()``/``TRIM()``
    overload without an explicit ``::text`` cast first (observed directly:
    ``function pg_catalog.btrim(enum_visa_class) does not exist``). Left
    uncast for every other or unrecognized dialect — this module doesn't
    know the column's real type, so ``::text`` is only added where it's
    known to be a safe no-op (Postgres casts a plain ``text`` column to
    itself) *and* known to fix a real, observed failure.
    """
    if _sqlglot_dialect(dialect) == "postgres":
        return f"{column_sql}::text"
    return column_sql


def _norm(value: Any) -> str:
    return str(value).strip().casefold()


def _string_predicates(tree: exp.Expression) -> list[tuple[exp.Column, list[str]]]:
    """Positive string-equality / ``IN`` predicates as ``(column_node, values)``.

    Only positive membership (``=``, ``IN``) is considered — negations are
    ignored, and ``LIKE`` has its own matching semantics (wildcards) so it's
    handled separately by ``_like_predicates``.
    """
    out: list[tuple[exp.Column, list[str]]] = []

    for eq in tree.find_all(exp.EQ):
        col = _as_column(eq.this) or _as_column(eq.expression)
        lit = _as_string_literal(eq.expression) or _as_string_literal(eq.this)
        if col is not None and lit is not None:
            out.append((col, [lit]))

    for in_expr in tree.find_all(exp.In):
        col = _as_column(in_expr.this)
        if col is None:
            continue
        values = [
            e.this
            for e in (in_expr.expressions or [])
            if isinstance(e, exp.Literal) and e.is_string
        ]
        if values:
            out.append((col, values))

    return out


def _like_predicates(tree: exp.Expression) -> list[tuple[exp.Column, str]]:
    """Positive, case-sensitive ``LIKE`` predicates as ``(column_node, pattern)``.

    ``NOT LIKE`` is skipped (only positive membership matters, same rule as
    ``_string_predicates``), and ``ILIKE`` is skipped too — it's already
    case-insensitive by construction, so there's nothing for this check to
    catch there.
    """
    out: list[tuple[exp.Column, str]] = []
    for like in tree.find_all(exp.Like):
        if isinstance(like.parent, exp.Not):
            continue
        col = _as_column(like.this)
        pattern = _as_string_literal(like.expression)
        if col is not None and pattern is not None:
            out.append((col, pattern))
    return out


def _sql_like_pattern_to_regex(pattern: str) -> str:
    """Translate a SQL ``LIKE`` pattern (``%``/``_`` wildcards) to an anchored regex.

    No escape-character support (``LIKE ... ESCAPE '\\'``) — patterns using it
    are rare enough here that treating ``%``/``_`` as literal wildcards always
    is an acceptable simplification, matching how conservative the rest of
    this module already is.
    """
    out = ["^"]
    for ch in pattern:
        if ch == "%":
            out.append(".*")
        elif ch == "_":
            out.append(".")
        else:
            out.append(re.escape(ch))
    out.append("$")
    return "".join(out)


def _as_column(node: Any) -> Optional[exp.Column]:
    return node if isinstance(node, exp.Column) else None


def _as_string_literal(node: Any) -> Optional[str]:
    if isinstance(node, exp.Literal) and node.is_string:
        return node.this
    return None


def _as_numeric_literal(node: Any) -> Optional[float]:
    if isinstance(node, exp.Literal) and not node.is_string:
        try:
            return float(node.this)
        except (TypeError, ValueError):
            return None
    return None


_COMPARISON_TYPES = (exp.GT, exp.GTE, exp.LT, exp.LTE)


def _numeric_comparison_predicates(
    tree: exp.Expression,
) -> list[tuple[exp.Column, float]]:
    """``(column_node, literal)`` pairs from ``>``/``>=``/``<``/``<=`` predicates.

    Only a bare ``column OP number`` (either side) is matched — no arithmetic
    (``ABS(col) > 0.15``), since the raw column value is what gets probed for
    its real range. Direction/operator doesn't matter here: the check only
    asks whether the literal's *magnitude* is plausible for the column, not
    which side of it the filter keeps.
    """
    out: list[tuple[exp.Column, float]] = []
    for cmp_type in _COMPARISON_TYPES:
        for node in tree.find_all(cmp_type):
            col = _as_column(node.this) or _as_column(node.expression)
            lit = _as_numeric_literal(node.expression) or _as_numeric_literal(node.this)
            if col is not None and lit is not None:
                out.append((col, lit))
    return out


def _table_nodes(tree: exp.Expression) -> tuple[list[exp.Table], dict[str, exp.Table]]:
    """All table refs in the query plus an alias/name → node lookup."""
    nodes = list(tree.find_all(exp.Table))
    by_key: dict[str, exp.Table] = {}
    for t in nodes:
        if t.name:
            by_key.setdefault(t.name.lower(), t)
        alias = t.alias
        if alias:
            by_key[alias.lower()] = t
    return nodes, by_key


def _candidate_tables(
    col: exp.Column, all_nodes: list[exp.Table], by_key: dict[str, exp.Table]
) -> list[exp.Table]:
    """Table nodes a column might belong to, resolved from the SQL alone."""
    qualifier = (col.table or "").lower()
    if qualifier:
        node = by_key.get(qualifier)
        return [node] if node is not None else all_nodes
    return all_nodes


def _fetch_distinct(
    executor: ProbeExecutor,
    table_node: exp.Table,
    col: exp.Column,
    dialect: Optional[str],
) -> Optional[list[str]]:
    """Distinct values for ``col`` in ``table_node`` via a live probe.

    Reproduces the exact identifiers/quoting the model used (both already
    executed against the real DB), so no schema knowledge is needed. Returns
    ``None`` when the column isn't in this table, the probe fails, or the column
    is high-cardinality (so we never judge a free-text/name column).
    """
    d = dialect or None
    # Bare column (drop table qualifier) so it works regardless of alias scoping.
    col_ref = exp.column(col.this).sql(dialect=d)
    table_ref = table_node.sql(dialect=d)
    threshold = DB_PROBE_LOW_CARD_THRESHOLD
    sql = (
        f"SELECT DISTINCT {col_ref} AS value FROM {table_ref} "
        f"WHERE {col_ref} IS NOT NULL LIMIT {threshold + 1}"
    )
    res = executor.run(sql, purpose="value_check_distinct")
    if not res["ok"] or not res["rows"]:
        return None
    values = [_first_value(r) for r in res["rows"]]
    if len(values) > threshold:  # high-cardinality — can't judge a literal here
        return None
    return [str(v) for v in values if v is not None]


def _fetch_range(
    executor: ProbeExecutor,
    table_node: exp.Table,
    col: exp.Column,
    dialect: Optional[str],
) -> Optional[tuple[float, float]]:
    """``(min, max)`` for ``col`` in ``table_node`` via a live probe, or ``None``
    when the column isn't numeric, has no rows, or the probe fails.

    A single-row ``MIN``/``MAX`` aggregate — cheap and portable across every
    dialect this executor supports (unlike ``PERCENTILE_CONT``, which SQLite
    and MySQL don't have).
    """
    d = _sqlglot_dialect(dialect)
    col_ref = exp.column(col.this).sql(dialect=d)
    table_ref = table_node.sql(dialect=d)
    sql = (
        f"SELECT MIN({col_ref}) AS lo, MAX({col_ref}) AS hi FROM {table_ref} "
        f"WHERE {col_ref} IS NOT NULL"
    )
    res = executor.run(sql, purpose="value_check_range")
    if not res["ok"] or not res["rows"]:
        return None
    row = res["rows"][0]
    try:
        lo = float(row.get("lo"))
        hi = float(row.get("hi"))
    except (TypeError, ValueError):
        return None  # non-numeric column (or all-NULL) — can't judge a threshold here
    return (lo, hi)


# Scale factors checked when a literal falls outside the column's real range.
# 100 covers the pattern actually observed (a threshold phrased as "60%" used
# verbatim against a column stored as a 0-1 fraction, or the reverse) — kept
# to this one well-evidenced conversion rather than guessing at arbitrary
# unit pairs, to keep false positives low.
_SCALE_FACTORS = (100.0, 1 / 100.0)


def find_numeric_scale_mismatches(
    executor: ProbeExecutor,
    dialect: Optional[str],
    sql: str,
) -> list[dict[str, Any]]:
    """Return likely percent-vs-fraction scale mismatches in numeric filters.

    Each entry: ``{"kind": "numeric_scale", "table", "column", "used",
    "suggested", "real_min", "real_max"}``. A literal already inside the
    column's real ``[min, max]`` range is left alone; a literal far outside
    it is only flagged when rescaling by 100x brings it back inside — that's
    reported as evidence, not auto-applied, since (unlike the string check)
    this is a heuristic rather than a certainty.
    """
    try:
        tree = sqlglot.parse_one(sql, read=_sqlglot_dialect(dialect))
    except Exception as exc:  # noqa: BLE001 — never break the pipeline on a parse error
        logger.info("numeric_scale_check: could not parse SQL (%s)", exc)
        return []
    if tree is None:
        return []

    all_nodes, by_key = _table_nodes(tree)
    if not all_nodes:
        return []

    mismatches: list[dict[str, Any]] = []
    range_cache: dict[tuple[str, str], Optional[tuple[float, float]]] = {}

    for col, used in _numeric_comparison_predicates(tree):
        candidates = _candidate_tables(col, all_nodes, by_key)
        real_range: Optional[tuple[float, float]] = None
        owning_table: Optional[str] = None
        for node in candidates:
            if not executor.budget_left:
                break
            key = (node.name.lower(), col.name.lower())
            if key not in range_cache:
                range_cache[key] = _fetch_range(executor, node, col, dialect)
            if range_cache[key] is not None:
                real_range = range_cache[key]
                owning_table = node.name
                break
        if real_range is None:  # column not resolvable, non-numeric, or probe failed
            continue

        lo, hi = real_range
        if lo <= used <= hi:  # already plausible — nothing to flag
            continue

        for factor in _SCALE_FACTORS:
            rescaled = used * factor
            if lo <= rescaled <= hi:
                mismatches.append(
                    {
                        "kind": "numeric_scale",
                        "table": owning_table,
                        "column": col.name,
                        "used": used,
                        "suggested": rescaled,
                        "real_min": lo,
                        "real_max": hi,
                    }
                )
                break  # one candidate rescale is enough evidence to surface

    return mismatches


# Postgres integer-family type names — exact match against known column
# types, not a substring check. Precision matters more here than in the
# composite-hit heuristics elsewhere in this codebase: a false match would
# mean flagging (or auto-rewriting) a division that was never a bug.
_INTEGER_TYPES = frozenset(
    {
        "integer",
        "int",
        "int2",
        "int4",
        "int8",
        "bigint",
        "smallint",
        "serial",
        "bigserial",
        "smallserial",
    }
)

# Casting an operand to any of these makes that operand non-integer, so a
# division against it won't truncate regardless of the other side.
_NON_INTEGER_CAST_TYPES = frozenset(
    {
        "numeric",
        "decimal",
        "float",
        "float4",
        "float8",
        "double precision",
        "double",
        "real",
        "money",
    }
)


def _has_non_integer_cast(node: exp.Expression) -> bool:
    """True if *node*'s own subtree already casts to a non-integer numeric
    type anywhere (``find_all`` includes *node* itself, so a bare
    ``col::numeric`` operand is caught too) — safe from truncation
    regardless of the other side of the division, no matter how deep the
    cast sits inside this operand."""
    for cast in node.find_all(exp.Cast):
        target = cast.to.sql(dialect="postgres").lower()
        if any(t in target for t in _NON_INTEGER_CAST_TYPES):
            return True
    return False


def _resolves_to_known_integer(
    node: exp.Expression,
    known_types: dict[tuple[str, str], str],
    all_nodes: list[exp.Table],
    by_key: dict[str, exp.Table],
) -> Optional[bool]:
    """Best-effort: does *node* evaluate to a Postgres integer-family type?

    True/False only when confidently resolvable from ``known_types``;
    ``None`` ("can't tell") for anything else — callers must never flag on
    ``None``, the same "never flag on missing information" rule this
    module follows everywhere else.
    """
    if isinstance(node, exp.Count):
        return True  # COUNT(...) is always bigint in Postgres, regardless of what's counted
    cols = list(node.find_all(exp.Column))
    if len(cols) != 1:
        return None  # no column, or more than one — can't confidently resolve
    col = cols[0]
    for t in _candidate_tables(col, all_nodes, by_key):
        known = known_types.get((t.name.lower(), col.name.lower()))
        if known:
            return known.strip().lower() in _INTEGER_TYPES
    return None


def _confident_integer_divisions(
    tree: exp.Expression, known_types: dict[tuple[str, str], str]
) -> list[exp.Div]:
    """Every ``Div`` node in *tree* that divides two Postgres integer-family
    operands with no cast anywhere in either operand's own subtree.

    Factored out so detection and self-apply share one predicate and can
    never disagree about what counts as a confident case.
    """
    all_nodes, by_key = _table_nodes(tree)
    if not all_nodes:
        return []
    out: list[exp.Div] = []
    for div in tree.find_all(exp.Div):
        numerator, denominator = div.this, div.expression
        if _has_non_integer_cast(numerator) or _has_non_integer_cast(denominator):
            continue  # one side already explicitly non-integer — safe
        num_is_int = _resolves_to_known_integer(
            numerator, known_types, all_nodes, by_key
        )
        den_is_int = _resolves_to_known_integer(
            denominator, known_types, all_nodes, by_key
        )
        if num_is_int is False or den_is_int is False:
            continue  # one side confirmed non-integer — safe regardless of the other
        if num_is_int is not True and den_is_int is not True:
            continue  # neither side confidently integer — can't tell, leave it alone
        out.append(div)
    return out


def find_integer_division_mismatches(
    dialect: Optional[str],
    sql: str,
    known_types: Optional[dict[tuple[str, str], str]] = None,
) -> list[dict[str, Any]]:
    """Return ``/`` divisions between two Postgres integer-family operands
    with no cast anywhere in either operand's own subtree.

    Postgres silently truncates integer/integer division toward zero
    (``5/2 = 2``, not ``2.5``) — a real, observed bug: ``SUM(x) / COUNT(y)``
    style ratio/average computations lose their fraction before an outer
    ``::numeric``/``ROUND(...)`` cast (which only wraps the already-
    truncated *result*) ever gets a chance to matter.

    Postgres-only, and deliberately so for a *behavioral* reason, not just
    a syntax one (contrast ``jsonb_path_check.py``'s Postgres-only gate,
    which is about ``->``/``->>`` syntax not existing elsewhere): several
    other dialects this pipeline supports don't share this truncation at
    all — MySQL's ``/`` always promotes to decimal (it has a separate
    ``DIV`` operator for integer division), and BigQuery/Snowflake's ``/``
    is always a float/decimal division. Flagging this on those dialects
    would be a false positive, not just unsupported syntax. The prompt
    already tells the SQL-generation LLM about this exact Postgres rule
    (see ``_POSTGRES_DIALECT_RULES`` in prompts.py) — this is the
    enforcement backstop for when that instruction doesn't get followed,
    not a replacement for it.

    Only flagged when at least one operand is confidently known (via
    ``known_types``) to be integer-typed and neither operand's own subtree
    already casts to a non-integer numeric type. Every other shape
    (unresolvable columns, multi-column expressions, both sides unknown)
    is left alone rather than guessed at.

    Each entry: ``{"kind": "integer_division", "numerator", "denominator"}``
    — the two operands' own SQL text, for the reconstruction message.
    """
    if _sqlglot_dialect(dialect) != "postgres":
        return []
    known_types = known_types or {}
    try:
        tree = sqlglot.parse_one(sql, read="postgres")
    except Exception as exc:  # noqa: BLE001 — never break the pipeline on a parse error
        logger.info("integer_division_check: could not parse SQL (%s)", exc)
        return []
    if tree is None:
        return []

    return [
        {
            "kind": "integer_division",
            "numerator": div.this.sql(dialect="postgres"),
            "denominator": div.expression.sql(dialect="postgres"),
        }
        for div in _confident_integer_divisions(tree, known_types)
    ]


def try_self_apply_integer_division_fixes(
    mismatches: list[dict[str, Any]],
    sql: str,
    dialect: Optional[str],
    known_types: Optional[dict[tuple[str, str], str]] = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Deterministically fix every ``integer_division`` mismatch by casting
    the numerator to ``numeric`` directly in the AST, then re-serializing
    once.

    Unlike the JSONB dotted-key case (which has one fixable shape and
    several ambiguous ones left to reconstruction), this fix has no
    ambiguous variant: there is exactly one correct rewrite for every
    mismatch this module detects (cast either operand — the numerator is
    chosen arbitrarily, both are equally correct), so every mismatch found
    here gets fixed. Nothing is left for reconstruction unless the SQL
    changed since detection ran (re-derived independently below, not
    matched against the passed-in mismatches by text/position, so this
    always agrees with what ``find_integer_division_mismatches`` reported
    for the same SQL).

    Returns the (possibly rewritten) SQL and the sub-list of mismatches
    that couldn't be fixed this way (non-``integer_division`` entries,
    unchanged, for the caller to route to reconstruction as normal).
    """
    fixable = [m for m in mismatches if m.get("kind") == "integer_division"]
    other = [m for m in mismatches if m.get("kind") != "integer_division"]
    if not fixable or _sqlglot_dialect(dialect) != "postgres":
        return sql, mismatches

    try:
        tree = sqlglot.parse_one(sql, read="postgres")
    except Exception as exc:  # noqa: BLE001 — never break the pipeline on a parse error
        logger.info("integer_division_check self-apply: could not parse SQL (%s)", exc)
        return sql, mismatches
    if tree is None:
        return sql, mismatches

    divisions = _confident_integer_divisions(tree, known_types or {})
    if not divisions:
        # SQL apparently changed since detection ran — don't silently drop
        # the reported mismatches, let reconstruction see them instead.
        return sql, mismatches

    for div in divisions:
        div.set(
            "this", exp.Cast(this=div.this.copy(), to=exp.DataType.build("numeric"))
        )

    return tree.sql(dialect="postgres"), other


def _resolve_distinct(
    executor: ProbeExecutor,
    dialect: Optional[str],
    col: exp.Column,
    all_nodes: list[exp.Table],
    by_key: dict[str, exp.Table],
    distinct_cache: dict[tuple[str, str], Optional[list[str]]],
) -> tuple[Optional[list[str]], Optional[str]]:
    """``(actual_values, owning_table)`` for *col*, via the shared distinct cache.

    ``(None, None)`` when the column isn't resolvable, is high-cardinality, or
    the probe failed — callers treat that as "nothing to check here."
    """
    for node in _candidate_tables(col, all_nodes, by_key):
        if not executor.budget_left:
            break
        key = (node.name.lower(), col.name.lower())
        if key not in distinct_cache:
            distinct_cache[key] = _fetch_distinct(executor, node, col, dialect)
        if distinct_cache[key] is not None:
            return distinct_cache[key], node.name
    return None, None


def find_literal_mismatches(
    executor: ProbeExecutor,
    dialect: Optional[str],
    sql: str,
) -> list[dict[str, Any]]:
    """Return fixable literal mismatches against real database values.

    Three independent things are flagged, all keyed off the same live
    ``DISTINCT`` probe (see ``_fetch_distinct``):

    - ``kind: "string"`` — the literal has no exact match in the database, but
      a near-miss (case/whitespace/spelling) does. Each entry:
      ``{"table", "column", "used", "suggested", "actual"}``.
    - ``kind: "case_duplicate"`` — the literal *does* exactly match a real
      value, but that same real-world value also appears in the column under
      other casing/whitespace (e.g. a status column storing ``'Certified'``,
      ``'CERTIFIED'``, and ``'certified'`` as three separate strings). An
      exact-match filter only ever catches one of those variants, silently
      undercounting the rest — this is real, present-in-the-data duplication,
      not a hypothetical. Each entry:
      ``{"table", "column", "used", "siblings"}``.
    - ``kind: "case_duplicate_like"`` — the same duplication, but for a
      case-sensitive ``LIKE`` pattern instead of ``=``/``IN``: at least one
      real value would match the pattern case-insensitively but not as
      written, so the query silently drops it. Each entry:
      ``{"table", "column", "pattern", "missed"}``.

    Empty list means nothing to repair: every literal/pattern is valid with no
    case-variant siblings, or a missing value has no close real counterpart
    and the empty result is legitimate.
    """
    try:
        tree = sqlglot.parse_one(sql, read=dialect or None)
    except Exception as exc:  # noqa: BLE001 — never break the pipeline on a parse error
        logger.info("literal_check: could not parse SQL (%s)", exc)
        return []
    if tree is None:
        return []

    all_nodes, by_key = _table_nodes(tree)
    if not all_nodes:
        return []

    mismatches: list[dict[str, Any]] = []
    distinct_cache: dict[tuple[str, str], Optional[list[str]]] = {}

    for col, values in _string_predicates(tree):
        actual, owning_table = _resolve_distinct(
            executor, dialect, col, all_nodes, by_key, distinct_cache
        )
        if not actual:  # column not resolvable, high-cardinality, or probe failed
            continue

        # Group the already-fetched distinct values by their normalized form
        # so a used literal can be checked against *all* siblings that share
        # its casefolded/trimmed identity, not just the first one seen.
        actual_by_norm: dict[str, list[str]] = {}
        for v in actual:
            actual_by_norm.setdefault(_norm(v), []).append(v)

        for used in values:
            norm = _norm(used)
            siblings = actual_by_norm.get(norm, [])

            if used in actual:
                # Exact match — the literal itself is fine, but if the same
                # real-world value is also stored under other casing or
                # whitespace, an exact-match filter still silently drops
                # those rows.
                other_siblings = [v for v in siblings if v != used]
                if other_siblings:
                    mismatches.append(
                        {
                            "kind": "case_duplicate",
                            "table": owning_table,
                            "column": col.name,
                            "used": used,
                            "siblings": sorted(siblings),
                            "dialect": dialect,
                        }
                    )
                continue

            if siblings:  # case / whitespace mismatch
                suggested = siblings
            else:
                suggested = difflib.get_close_matches(
                    used, actual, n=3, cutoff=_CLOSE_MATCH_CUTOFF
                )
            if not suggested:  # no near-miss → value legitimately absent, leave it
                continue
            mismatches.append(
                {
                    "kind": "string",
                    "table": owning_table,
                    "column": col.name,
                    "used": used,
                    "suggested": suggested,
                    "actual": actual,
                }
            )

    for col, pattern in _like_predicates(tree):
        actual, owning_table = _resolve_distinct(
            executor, dialect, col, all_nodes, by_key, distinct_cache
        )
        if not actual:
            continue

        case_sensitive_re = re.compile(_sql_like_pattern_to_regex(pattern))
        case_insensitive_re = re.compile(
            _sql_like_pattern_to_regex(pattern), re.IGNORECASE
        )
        missed = [
            v
            for v in actual
            if case_insensitive_re.match(v) and not case_sensitive_re.match(v)
        ]
        if missed:
            mismatches.append(
                {
                    "kind": "case_duplicate_like",
                    "table": owning_table,
                    "column": col.name,
                    "pattern": pattern,
                    "missed": sorted(missed),
                    "dialect": dialect,
                }
            )

    return mismatches


def _render_string_mismatch(m: dict[str, Any]) -> str:
    actual_preview = ", ".join(str(v) for v in m["actual"][:20])
    suggested = ", ".join(str(v) for v in m["suggested"])
    return (
        f"- Column {m['table']}.{m['column']}: your filter used '{m['used']}', "
        f"which does not exist in the database and returns no rows. "
        f"Actual values are [{actual_preview}]. Closest real value(s): {suggested}."
    )


def _render_case_duplicate_mismatch(m: dict[str, Any]) -> str:
    variants = ", ".join(f"'{v}'" for v in m["siblings"] if v != m["used"])
    cast_col = _text_cast(m["column"], m.get("dialect"))
    return (
        f"- Column {m['table']}.{m['column']}: your filter used '{m['used']}', "
        f"which exists, but the same real-world value is also stored as "
        f"{variants} in this column. An exact-match filter only catches one "
        f"of these variants — use LOWER(TRIM({cast_col})) = "
        f"LOWER(TRIM('{m['used']}')) instead."
    )


def _render_case_duplicate_like_mismatch(m: dict[str, Any]) -> str:
    missed = ", ".join(f"'{v}'" for v in m["missed"])
    return (
        f"- Column {m['table']}.{m['column']}: your filter used "
        f"LIKE '{m['pattern']}', which is case-sensitive, so it misses real "
        f"values that only match once case is ignored: {missed}."
    )


def _render_numeric_scale_mismatch(m: dict[str, Any]) -> str:
    return (
        f"- Column {m['table']}.{m['column']}: your filter used {m['used']:g}, "
        f"but real values in this column range from {m['real_min']:g} to "
        f"{m['real_max']:g} — {m['used']:g} is implausible against that range. "
        f"{m['suggested']:g} (a 100x rescale of your literal) does fall inside "
        f"it, which suggests a percent-vs-fraction unit mismatch."
    )


def _render_integer_division_mismatch(m: dict[str, Any]) -> str:
    return (
        f"- `{m['numerator']} / {m['denominator']}`: both sides are "
        f"integer-typed, so Postgres computes this as integer division and "
        f"silently truncates toward zero (e.g. 5/2 = 2, not 2.5) — any "
        f"ratio or average computed this way is wrong."
    )


def build_value_repair_error(mismatches: list[dict[str, Any]]) -> str:
    """Render mismatches (string near-misses and/or numeric scale issues)
    into a targeted reconstruction instruction."""
    string_mismatches = [m for m in mismatches if m.get("kind", "string") == "string"]
    case_duplicate_mismatches = [
        m for m in mismatches if m.get("kind") == "case_duplicate"
    ]
    case_duplicate_like_mismatches = [
        m for m in mismatches if m.get("kind") == "case_duplicate_like"
    ]
    numeric_mismatches = [m for m in mismatches if m.get("kind") == "numeric_scale"]
    # Reaching here at all is the rare fallback: try_self_apply_integer_
    # division_fixes fixes every integer_division mismatch it's given
    # deterministically, so this section only fires if self-apply couldn't
    # run (e.g. the SQL changed between detection and self-apply).
    integer_division_mismatches = [
        m for m in mismatches if m.get("kind") == "integer_division"
    ]

    sections = []
    if string_mismatches:
        body = "\n".join(_render_string_mismatch(m) for m in string_mismatches)
        sections.append(
            "One or more filter literals do not match any value stored in the "
            "database, so the query returns no matching rows:\n"
            f"{body}\n\n"
            "Rewrite the SQL using the exact value(s) that exist in the database "
            "(choose the one that matches the question's intent). Change ONLY the "
            "mismatched literal(s); keep all joins, columns, grouping, and other "
            "filters exactly as they are."
        )
    if case_duplicate_mismatches:
        body = "\n".join(
            _render_case_duplicate_mismatch(m) for m in case_duplicate_mismatches
        )
        sections.append(
            "One or more filter literals exactly match a real value, but that "
            "same real-world value is also stored under different "
            "casing/whitespace elsewhere in the column, so an exact-match "
            "filter silently misses those rows:\n"
            f"{body}\n\n"
            "Use the exact LOWER(TRIM(...)) comparison given for each column "
            "above so every case/whitespace variant of the intended value is "
            "matched. Change ONLY the affected comparison(s); keep all joins, "
            "columns, grouping, and other filters exactly as they are."
        )
    if case_duplicate_like_mismatches:
        body = "\n".join(
            _render_case_duplicate_like_mismatch(m)
            for m in case_duplicate_like_mismatches
        )
        # NOTE: this always tells reconstruction to switch to ILIKE, which is
        # Postgres-only (e.g. banned outright by the SQLite dialect rules in
        # prompts.py) — potentially wrong advice on non-Postgres connectors.
        # flagging for future generalization.
        sections.append(
            "One or more filter patterns use a case-sensitive LIKE, but the "
            "column stores case/whitespace variants of the same real-world "
            "value that a case-sensitive pattern silently drops:\n"
            f"{body}\n\n"
            "Change LIKE to ILIKE for the affected condition(s) so every "
            "case variant matches. Change ONLY the affected condition(s); "
            "keep all joins, columns, grouping, and other filters exactly "
            "as they are."
        )
    if numeric_mismatches:
        body = "\n".join(_render_numeric_scale_mismatch(m) for m in numeric_mismatches)
        sections.append(
            "One or more numeric filter thresholds look implausible against the "
            "column's real data — evidence, not a certain diagnosis:\n"
            f"{body}\n\n"
            "Check whether the threshold should be rescaled to match the "
            "column's real range (e.g. a percent phrased in the question "
            "applied to a column stored as a 0-1 fraction, or the reverse). "
            "Only change a threshold if the evidence supports it; keep "
            "everything else — joins, columns, grouping, other filters — "
            "exactly as they are."
        )
    if integer_division_mismatches:
        body = "\n".join(
            _render_integer_division_mismatch(m) for m in integer_division_mismatches
        )
        sections.append(
            "One or more divisions use two integer-typed operands, which "
            "Postgres silently truncates toward zero instead of producing a "
            "fractional result:\n"
            f"{body}\n\n"
            "Cast the numerator (or denominator) to ::numeric AT THE "
            "DIVISION ITSELF — casting only the final/outer result does not "
            "help, the inner division has already truncated by then. Change "
            "ONLY the affected division(s); keep everything else exactly as "
            "it is."
        )
    return "\n\n".join(sections)


__all__ = [
    "find_literal_mismatches",
    "find_numeric_scale_mismatches",
    "find_integer_division_mismatches",
    "try_self_apply_integer_division_fixes",
    "build_value_repair_error",
]
