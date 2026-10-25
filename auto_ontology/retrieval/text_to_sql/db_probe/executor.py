# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hardened, read-only probe executor.

This is the single choke point for every live query the probing layer issues.
It guarantees a probe:
- is a single ``SELECT`` / ``WITH`` statement (no DDL/DML, no multi-statement),
- is ``LIMIT``-bounded (auto-appended when missing),
- counts against a per-question call budget.

Probes run **inline on the calling thread**. The pipeline's connectors are
shared and long-lived, and thread-affine connectors (e.g. SQLite binds each
connection to its creating thread) reuse the calling thread's existing
connection — so inline execution avoids leaking a fresh connection per probe.
Bounding therefore relies on ``LIMIT`` + the call budget rather than a hard
wall-clock timeout; per-dialect statement timeouts (Postgres ``statement_timeout``,
SQLite ``set_progress_handler``) are the right tool once Phase 2 runs arbitrary
model-authored SQL.

The same executor backs both the Phase-1 deterministic probes and any future
Phase-2 LLM-driven exploration loop, so all live access inherits these guards.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import sqlglot
from sqlglot import exp
from sqlglot.errors import ErrorLevel, ParseError

from auto_ontology.connectors.base import SQLDatabase

from auto_ontology.retrieval.text_to_sql.db_probe.config import (
    DB_PROBE_MAX_CALLS,
    DB_PROBE_MAX_ROWS,
)

from auto_ontology.retrieval.text_to_sql.chat_sql import execute_chat_sql

logger = logging.getLogger(__name__)

# SQLGlot expression class names that represent writes, schema changes, commands,
# transaction control, or another side effect. Names keep this compatible across
# SQLGlot releases where less-common dialect nodes may not exist as attributes.
_FORBIDDEN_NODE_TYPES = frozenset(
    {
        "Alter",
        "Analyze",
        "Attach",
        "Cache",
        "Command",
        "Commit",
        "Copy",
        "Create",
        "Delete",
        "Detach",
        "Drop",
        "Execute",
        "Grant",
        "Insert",
        "Into",
        "LoadData",
        "Lock",
        "Merge",
        "Pragma",
        "Replace",
        "Revoke",
        "Rollback",
        "Set",
        "Transaction",
        "TruncateTable",
        "Uncache",
        "Update",
        "Use",
        "Vacuum",
    }
)
_SIDE_EFFECTING_FUNCTIONS = frozenset(
    {
        "benchmark",
        "dblink_exec",
        "get_lock",
        "lo_export",
        "lo_import",
        "nextval",
        "pg_advisory_lock",
        "pg_advisory_xact_lock",
        "pg_cancel_backend",
        "pg_reload_conf",
        "pg_rotate_logfile",
        "pg_sleep",
        "pg_terminate_backend",
        "pg_try_advisory_lock",
        "pg_try_advisory_xact_lock",
        "release_lock",
        "setval",
        "sleep",
    }
)


def _parse_read_only_query(
    sql: str,
    dialect: str | None = None,
) -> tuple[exp.Query | None, str | None]:
    """Parse one query and return a specific fail-closed rejection reason."""
    if not sql:
        return None, "empty SQL"
    try:
        statements = sqlglot.parse(
            sql,
            read=dialect or None,
            error_level=ErrorLevel.RAISE,
        )
    except (ParseError, ValueError, TypeError) as exc:
        return None, f"unparseable SQL: {str(exc).splitlines()[0]}"

    if len(statements) != 1 or statements[0] is None:
        return None, f"expected exactly one SQL statement, found {len(statements)}"

    tree = statements[0]
    if not isinstance(tree, exp.Query):
        return None, f"statement type {type(tree).__name__} is not a read-only query"

    for node in tree.walk():
        node_type = type(node).__name__
        if node_type in _FORBIDDEN_NODE_TYPES:
            if node_type == "Into":
                return None, "SELECT INTO is not read-only"
            return None, f"forbidden {node_type} construct"
        if isinstance(node, exp.Anonymous) and node.name.casefold() in (
            _SIDE_EFFECTING_FUNCTIONS
        ):
            return None, f"side-effecting function {node.name} is forbidden"

    if tree.find(exp.Select) is None:
        return None, "query contains no SELECT"

    return tree, None


def is_read_only_select(sql: str, dialect: str | None = None) -> bool:
    """Return ``True`` only for one parsed, side-effect-free query."""
    tree, _reason = _parse_read_only_query(sql, dialect)
    return tree is not None


def _ensure_limit(
    sql: str,
    max_rows: int,
    dialect: str | None = None,
    *,
    tree: exp.Query | None = None,
) -> str:
    """Preserve a top-level LIMIT or add one to the parsed query expression."""
    stripped = sql.strip().rstrip(";").strip()
    parsed = tree
    if parsed is None:
        parsed, reason = _parse_read_only_query(stripped, dialect)
        if parsed is None:
            raise ValueError(reason or "query could not be validated")
    if parsed.args.get("limit") is not None:
        return stripped
    return parsed.limit(max_rows, copy=True).sql(dialect=dialect or None)


@dataclass
class ProbeExecutor:
    """Runs read-only probes against a single connector under strict limits.

    Supports the context-manager protocol for symmetry with future
    resource-holding backends; the current inline implementation holds nothing::

        with ProbeExecutor(connector) as ex:
            ex.run("SELECT ...")

    Parameters
    ----------
    connector:
        The resolved :class:`SQLDatabase` to probe (may be ``None`` — every
        ``run`` then fails gracefully).
    """

    connector: Optional[SQLDatabase]
    max_calls: int = DB_PROBE_MAX_CALLS
    max_rows: int = DB_PROBE_MAX_ROWS
    calls: int = field(default=0, init=False)
    log: list[dict[str, Any]] = field(default_factory=list, init=False)

    @property
    def budget_left(self) -> int:
        return max(0, self.max_calls - self.calls)

    def close(self) -> None:
        """No-op — inline execution holds no resources. Present for API symmetry."""

    def __enter__(self) -> "ProbeExecutor":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def run(self, sql: str, purpose: str = "") -> dict[str, Any]:
        """Execute one probe, returning a structured result entry.

        The entry is always appended to ``self.log`` and returned; it never
        raises. Shape: ``{sql, purpose, ok, rows, row_count, truncated, error,
        elapsed_ms}``.
        """
        entry: dict[str, Any] = {
            "sql": sql,
            "purpose": purpose,
            "ok": False,
            "rows": None,
            "row_count": 0,
            "truncated": False,
            "error": None,
            "elapsed_ms": 0,
        }

        if self.connector is None:
            entry["error"] = "no connector available"
            self.log.append(entry)
            return entry

        if self.calls >= self.max_calls:
            entry["error"] = "probe budget exhausted"
            self.log.append(entry)
            return entry

        dialect = getattr(self.connector, "dialect", None)
        tree, rejection_reason = _parse_read_only_query(sql, dialect)
        if tree is None:
            reason = rejection_reason or "query could not be validated"
            entry["error"] = f"rejected: {reason}"
            logger.warning("Probe rejected (%s): %s", reason, sql)
            self.log.append(entry)
            return entry

        try:
            bounded_sql = _ensure_limit(sql, self.max_rows, dialect, tree=tree)
        except Exception as exc:  # noqa: BLE001 — uncertainty must fail closed
            reason = f"row-limit enforcement failed: {exc}"
            entry["error"] = f"rejected: {reason}"
            logger.warning("Probe rejected (%s): %s", reason, sql)
            self.log.append(entry)
            return entry
        entry["sql"] = bounded_sql
        self.calls += 1

        start = time.perf_counter()
        try:
            df = execute_chat_sql(self.connector, bounded_sql, kind="probe SQL")
        except Exception as exc:  # noqa: BLE001 — probes must never break the pipeline
            entry["error"] = str(exc)
            entry["elapsed_ms"] = int((time.perf_counter() - start) * 1000)
            logger.info("Probe failed: %s :: %s", bounded_sql, exc)
            self.log.append(entry)
            return entry

        entry["elapsed_ms"] = int((time.perf_counter() - start) * 1000)
        try:
            total = len(df)
            records = df.head(self.max_rows).to_dict(orient="records")
        except Exception as exc:  # noqa: BLE001
            entry["error"] = f"result parse failed: {exc}"
            self.log.append(entry)
            return entry

        entry["ok"] = True
        entry["rows"] = records
        entry["row_count"] = total
        entry["truncated"] = total > self.max_rows
        self.log.append(entry)
        return entry


__all__ = ["ProbeExecutor", "is_read_only_select"]
