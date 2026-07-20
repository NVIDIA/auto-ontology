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
"""

from __future__ import annotations

import difflib
import logging
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


def _norm(value: Any) -> str:
    return str(value).strip().casefold()


def _string_predicates(tree: exp.Expression) -> list[tuple[exp.Column, list[str]]]:
    """Positive string-equality / ``IN`` predicates as ``(column_node, values)``.

    Only positive membership (``=``, ``IN``) is considered — negations and LIKE
    are intentionally ignored.
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


def _as_column(node: Any) -> Optional[exp.Column]:
    return node if isinstance(node, exp.Column) else None


def _as_string_literal(node: Any) -> Optional[str]:
    if isinstance(node, exp.Literal) and node.is_string:
        return node.this
    return None


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
    d = _sqlglot_dialect(dialect)
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


def find_literal_mismatches(
    executor: ProbeExecutor,
    dialect: Optional[str],
    sql: str,
) -> list[dict[str, Any]]:
    """Return fixable literal mismatches: used value is a near-miss of a real one.

    Each entry: ``{"table", "column", "used", "suggested", "actual"}``. Empty
    list means nothing to repair (every literal is valid, or the missing value
    has no close real counterpart and the empty result is legitimate).
    """
    try:
        tree = sqlglot.parse_one(sql, read=_sqlglot_dialect(dialect))
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
        candidates = _candidate_tables(col, all_nodes, by_key)
        actual: Optional[list[str]] = None
        owning_table: Optional[str] = None
        for node in candidates:
            if not executor.budget_left:
                break
            key = (node.name.lower(), col.name.lower())
            if key not in distinct_cache:
                distinct_cache[key] = _fetch_distinct(executor, node, col, dialect)
            if distinct_cache[key] is not None:
                actual = distinct_cache[key]
                owning_table = node.name
                break
        if not actual:  # column not resolvable, high-cardinality, or probe failed
            continue

        actual_by_norm = {_norm(v): v for v in actual}
        for used in values:
            if used in actual:  # exact match — fine
                continue
            norm = _norm(used)
            if norm in actual_by_norm:  # case / whitespace mismatch
                suggested = [actual_by_norm[norm]]
            else:
                suggested = difflib.get_close_matches(
                    used, actual, n=3, cutoff=_CLOSE_MATCH_CUTOFF
                )
            if not suggested:  # no near-miss → value legitimately absent, leave it
                continue
            mismatches.append(
                {
                    "table": owning_table,
                    "column": col.name,
                    "used": used,
                    "suggested": suggested,
                    "actual": actual,
                }
            )

    return mismatches


def build_value_repair_error(mismatches: list[dict[str, Any]]) -> str:
    """Render mismatches into a targeted reconstruction instruction."""
    lines = []
    for m in mismatches:
        actual_preview = ", ".join(str(v) for v in m["actual"][:20])
        suggested = ", ".join(str(v) for v in m["suggested"])
        lines.append(
            f"- Column {m['table']}.{m['column']}: your filter used '{m['used']}', "
            f"which does not exist in the database and returns no rows. "
            f"Actual values are [{actual_preview}]. Closest real value(s): {suggested}."
        )
    body = "\n".join(lines)
    return (
        "One or more filter literals do not match any value stored in the "
        "database, so the query returns no matching rows:\n"
        f"{body}\n\n"
        "Rewrite the SQL using the exact value(s) that exist in the database "
        "(choose the one that matches the question's intent). Change ONLY the "
        "mismatched literal(s); keep all joins, columns, grouping, and other "
        "filters exactly as they are."
    )


__all__ = [
    "find_literal_mismatches",
    "build_value_repair_error",
]
