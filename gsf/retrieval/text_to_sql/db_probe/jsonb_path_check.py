# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""JSONB key-path check for the repair path.

Mirrors ``literal_check.py``'s approach but for ``->``/``->>`` JSONB navigation
instead of filter literals: the ingestion-time sample-value hint shown to the
model (see ``sql_from_semantic.py``) is only a suggestion, and the model can
still hallucinate a plausible-looking sibling key (e.g. write
``cost_breakdown->'fees'->>'repair_estimate'`` when the real key lives under
``cost_breakdown->'repair_costs'``). Such a path is syntactically valid SQL but
silently returns ``NULL`` for every row instead of erroring, so it needs its
own live check — a bad *column* name is already caught for free by SQL
execution, but a bad *JSONB key* is not.

Like ``literal_check.py`` this makes no assumptions about the incoming
database: the table/column identifiers are taken from the SQL's own AST
(already valid, since the query passed SQL validation), and the real key set
is discovered at runtime via ``jsonb_object_keys``.

Postgres-only: the ``->``/``->>`` operators and ``jsonb_object_keys`` are a
Postgres-specific surface, so this is a no-op on every other dialect.

Only the shapes already covered by the ingestion-time sample hint are checked:
``col ->> 'key'`` (flat) and ``col -> 'container' ->> 'key'`` (one level of
nesting). Deeper chains are left unchecked rather than guessed at.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import sqlglot
from sqlglot import exp

from gsf.retrieval.text_to_sql.db_probe.executor import ProbeExecutor
from gsf.retrieval.text_to_sql.db_probe.literal_check import (
    _as_column,
    _candidate_tables,
    _first_value,
    _sqlglot_dialect,
    _table_nodes,
)

logger = logging.getLogger(__name__)


def _single_json_key(path: Any) -> Optional[str]:
    """The lone key name of a ``JSONPath`` like ``$.foo``, or ``None``.

    Returns ``None`` for anything more complex (wildcards, array indices,
    multi-segment paths) so those are left unchecked rather than misread.
    """
    if not isinstance(path, exp.JSONPath):
        return None
    parts = [e for e in path.expressions if not isinstance(e, exp.JSONPathRoot)]
    if len(parts) != 1:
        return None
    key_node = parts[0]
    if isinstance(key_node, exp.JSONPathKey):
        return str(key_node.this)
    return None


def _parse_extract(
    extract: exp.JSONExtractScalar,
) -> Optional[tuple[exp.Column, Optional[str], str]]:
    """Decompose a ``->>`` node into ``(column, container_or_None, key)``.

    Only the flat (``col ->> 'key'``) and single-nesting (``col -> 'c' ->>
    'key'``) shapes resolve to a result; anything else (function calls, array
    access, deeper chains) returns ``None`` and is skipped.
    """
    key = _single_json_key(extract.expression)
    if key is None:
        return None

    inner = extract.this
    if isinstance(inner, exp.Column):
        return inner, None, key
    if isinstance(inner, exp.JSONExtract):
        container = _single_json_key(inner.expression)
        col = _as_column(inner.this)
        if container is None or col is None:
            return None
        return col, container, key
    return None


def _fetch_jsonb_keys(
    executor: ProbeExecutor,
    table_node: exp.Table,
    col: exp.Column,
    container: Optional[str],
    dialect: Optional[str],
) -> Optional[list[str]]:
    """Real keys present under ``col`` (or ``col->container``) via a live probe.

    Returns ``None`` when the column isn't in this table, the value there is
    never a JSON object, or the probe fails — i.e. "not resolvable here", not
    "the key is missing". Mirrors ``literal_check._fetch_distinct``.
    """
    d = _sqlglot_dialect(dialect)
    col_ref = exp.column(col.this).sql(dialect=d)
    table_ref = table_node.sql(dialect=d)
    target = f"{col_ref}->'{container}'" if container else col_ref
    sql = (
        f"SELECT DISTINCT jsonb_object_keys({target}) AS value FROM {table_ref} "
        f"WHERE {target} IS NOT NULL AND jsonb_typeof({target}) = 'object' "
    )
    res = executor.run(sql, purpose="jsonb_path_check_keys")
    if not res["ok"] or not res["rows"]:
        return None
    values = [_first_value(r) for r in res["rows"]]
    return [str(v) for v in values if v is not None]


def find_jsonb_path_mismatches(
    executor: ProbeExecutor,
    dialect: Optional[str],
    sql: str,
    known_types: Optional[dict[tuple[str, str], str]] = None,
) -> list[dict[str, Any]]:
    """Return JSONB key paths that don't exist in the live database, or that
    were used on a column known not to be JSONB at all.

    ``known_types`` is an optional ``{(table_name.lower(), col_name.lower()):
    data_type}`` map (e.g. from ``path_state["relevant_tables"]``, already
    fetched for the generation prompt — no extra DB/graph round-trip). It's
    used only as a fallback: when the live probe can't confirm real keys for
    a path (``jsonb_object_keys`` errors or returns nothing, e.g. because the
    column isn't a JSON type), that case was previously silently skipped as
    "not resolvable." If ``known_types`` says the column's real type isn't
    JSON-like, that silence is instead surfaced as its own mismatch kind
    (``"wrong_type"`` set) — the model used ``->``/``->>`` on a column that
    was never a JSONB column, not merely a wrong key.

    Each entry: ``{"table", "column", "container", "used_key",
    "available_keys"}`` for a real key mismatch, or ``{"table", "column",
    "container", "used_key", "wrong_type": <data_type>}`` for the
    non-JSONB-column case. Empty list means every checked path resolved, or
    none of the paths in the SQL were checkable (unsupported dialect/shape,
    probe budget exhausted, etc.) — never treated as "everything is fine"
    beyond that.
    """
    if _sqlglot_dialect(dialect) != "postgres":
        return []

    try:
        tree = sqlglot.parse_one(sql, read="postgres")
    except Exception as exc:  # noqa: BLE001 — never break the pipeline on a parse error
        logger.info("jsonb_path_check: could not parse SQL (%s)", exc)
        return []
    if tree is None:
        return []

    all_nodes, by_key = _table_nodes(tree)
    if not all_nodes:
        return []

    mismatches: list[dict[str, Any]] = []
    keys_cache: dict[tuple[str, str, Optional[str]], Optional[list[str]]] = {}

    for extract in tree.find_all(exp.JSONExtractScalar):
        parsed = _parse_extract(extract)
        if parsed is None:
            continue
        col, container, key = parsed

        candidates = _candidate_tables(col, all_nodes, by_key)
        available: Optional[list[str]] = None
        owning_table: Optional[str] = None
        for node in candidates:
            if not executor.budget_left:
                break
            cache_key = (node.name.lower(), col.name.lower(), container)
            if cache_key not in keys_cache:
                keys_cache[cache_key] = _fetch_jsonb_keys(
                    executor, node, col, container, dialect
                )
            if keys_cache[cache_key] is not None:
                available = keys_cache[cache_key]
                owning_table = node.name
                break
        if available is None:  # column not resolvable, never an object, or probe failed
            # Fallback: if we independently know this column's real type and
            # it's not JSON-like, the probe's silence just means "correctly
            # errored on a non-JSON column" — surface that as its own
            # mismatch kind instead of dropping it.
            if known_types is not None:
                for node in candidates:
                    known_type = known_types.get((node.name.lower(), col.name.lower()))
                    if known_type and "json" not in known_type.lower():
                        mismatches.append(
                            {
                                "table": node.name,
                                "column": col.name,
                                "container": container,
                                "used_key": key,
                                "wrong_type": known_type,
                            }
                        )
                        break
            continue

        if key not in available:
            sibling_containers: Optional[list[str]] = None
            if container is not None:
                # Also surface the column's own top-level sections (siblings
                # of the wrong container) so reconstruction has somewhere to
                # look next, not just proof that the guessed spot was wrong.
                top_cache_key = (owning_table.lower(), col.name.lower(), None)
                if top_cache_key not in keys_cache:
                    node = by_key.get(owning_table.lower())
                    keys_cache[top_cache_key] = (
                        _fetch_jsonb_keys(executor, node, col, None, dialect)
                        if node is not None and executor.budget_left
                        else None
                    )
                sibling_containers = keys_cache[top_cache_key]

            mismatches.append(
                {
                    "table": owning_table,
                    "column": col.name,
                    "container": container,
                    "used_key": key,
                    "available_keys": available,
                    "sibling_containers": sibling_containers,
                }
            )

    return mismatches


def try_self_apply_fixes(
    mismatches: list[dict[str, Any]], sql: str
) -> tuple[str, list[dict[str, Any]]]:
    """Deterministically fix the one JSONB-mismatch shape that's unambiguous
    without an LLM: a flattened dotted key (``col ->> 'container.key'``)
    where ``container`` is a real top-level key of that column. This is a
    typo, not a domain-knowledge question — the model meant nested access
    (``col -> 'container' ->> 'key'``) and wrote the dotted flat form
    instead, so there's exactly one correct rewrite, not several candidates
    to guess between.

    Every other mismatch shape (genuinely wrong key with no dot, a container
    that already exists but the sub-key doesn't, `wrong_type`) is left
    alone — those need picking the right key out of ``available_keys`` or
    restructuring the query, which is a real judgment call, not something to
    guess at mechanically. Those still go to reconstruction exactly as
    before.

    Returns the (possibly rewritten) SQL and the sub-list of mismatches that
    couldn't be fixed this way, for the caller to route to reconstruction as
    normal.
    """
    remaining: list[dict[str, Any]] = []
    for m in mismatches:
        fixed = _self_apply_one(m, sql)
        if fixed is None:
            remaining.append(m)
        else:
            sql = fixed
    return sql, remaining


def _self_apply_one(mismatch: dict[str, Any], sql: str) -> Optional[str]:
    """Return the rewritten SQL if *mismatch* is a fixable flattened-dot-key
    typo, else ``None`` (caller leaves it for reconstruction)."""
    if "wrong_type" in mismatch or mismatch.get("container") is not None:
        return None
    key = mismatch["used_key"]
    if key.count(".") != 1:
        return None
    container, real_key = key.split(".", 1)
    available = mismatch.get("available_keys") or []
    matched_container = next(
        (k for k in available if k.lower() == container.lower()), None
    )
    if matched_container is None:
        return None
    old = f"->>'{key}'"
    new = f"->'{matched_container}'->>'{real_key}'"
    if old not in sql:
        return None  # exact text not found — bail rather than guess at a partial match
    return sql.replace(old, new)


def build_jsonb_path_repair_error(mismatches: list[dict[str, Any]]) -> str:
    """Render mismatches into a targeted reconstruction instruction."""
    key_mismatches = [m for m in mismatches if "wrong_type" not in m]
    type_mismatches = [m for m in mismatches if "wrong_type" in m]
    sections = []

    if key_mismatches:
        lines = []
        for m in key_mismatches:
            path = (
                f"{m['column']}->'{m['container']}'->>'{m['used_key']}'"
                if m["container"]
                else f"{m['column']}->>'{m['used_key']}'"
            )
            available_preview = ", ".join(m["available_keys"][:20])
            line = (
                f"- {m['table']}.{path}: this key does not exist and returns NULL "
                f"for every row. Actual keys available there: [{available_preview}]."
            )
            siblings = m.get("sibling_containers")
            if siblings:
                sibling_preview = ", ".join(siblings[:20])
                line += (
                    f" Other top-level sections in {m['column']}: [{sibling_preview}] — "
                    "the key you want may live under one of these instead."
                )
            lines.append(line)
        sections.append(
            "One or more JSONB key paths in the generated SQL do not exist in the "
            "database, so they return NULL for every row instead of erroring:\n"
            + "\n".join(lines)
            + "\n\nRewrite the SQL using the correct container/key name from the actual "
            "keys listed above. Change ONLY the mismatched JSONB path(s); keep all "
            "joins, columns, grouping, and other filters exactly as they are."
        )

    if type_mismatches:
        lines = []
        for m in type_mismatches:
            path = (
                f"{m['column']}->'{m['container']}'->>'{m['used_key']}'"
                if m["container"]
                else f"{m['column']}->>'{m['used_key']}'"
            )
            lines.append(
                f"- {m['table']}.{path}: {m['table']}.{m['column']} is a "
                f"{m['wrong_type']} column, not JSONB — this is not a wrong key, "
                "the ->/->> operators don't apply to this column at all."
            )
        sections.append(
            "One or more JSONB navigation operators were used on a column that "
            "isn't JSONB-typed:\n"
            + "\n".join(lines)
            + "\n\nRewrite the SQL to reference the column directly (no ->/->> "
            "operators) instead of treating it as JSONB. Change ONLY the "
            "affected reference(s); keep all joins, columns, grouping, and "
            "other filters exactly as they are."
        )

    return "\n\n".join(sections)


__all__ = [
    "find_jsonb_path_mismatches",
    "build_jsonb_path_repair_error",
    "try_self_apply_fixes",
]
