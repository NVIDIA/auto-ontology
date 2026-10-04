# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hardened, read-only probe executor.

This is the single choke point for every live query the probing layer issues.
It guarantees a probe:
- is a single ``SELECT`` / ``WITH`` statement (no DDL/DML, no multi-statement),
- is ``LIMIT``-bounded even when the submitted query carries a larger limit,
- can be restricted to authorized tables and classified non-PII columns,
- retains no result values in its audit log,
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
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import sqlglot
from sqlglot import exp

from auto_ontology.connectors.base import SQLDatabase
from auto_ontology.dal.pii import is_column_safe_for_data_movement
from auto_ontology.retrieval.text_to_sql.db_probe.config import (
    DB_PROBE_MAX_CALLS,
    DB_PROBE_MAX_RESULT_COLUMNS,
    DB_PROBE_MAX_ROWS,
    DB_PROBE_TIMEOUT_S,
)

from auto_ontology.retrieval.text_to_sql.chat_sql import execute_chat_sql

logger = logging.getLogger(__name__)

# Statement keywords that must never appear in a probe (mutations / side effects).
_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|truncate|grant|revoke|"
    r"attach|detach|pragma|vacuum|reindex|merge|call|exec|execute|copy|into)\b",
    re.IGNORECASE,
)
_LIMIT_RE = re.compile(r"\blimit\b", re.IGNORECASE)


def allowed_table_scope(tables: list[dict[str, Any]]) -> frozenset[str]:
    """Build a fail-closed SQL table allowlist from authorized table contexts."""

    names: dict[str, int] = {}
    for table in tables:
        name = str(table.get("name") or "").casefold()
        if name:
            names[name] = names.get(name, 0) + 1

    allowed: set[str] = set()
    for table in tables:
        name = str(table.get("name") or "").casefold()
        schema = str(table.get("schema_name") or "").casefold()
        if not name:
            continue
        if schema:
            allowed.add(f"{schema}.{name}")
        if not schema or names[name] == 1:
            allowed.add(name)
    return frozenset(allowed)


def is_read_only_select(sql: str) -> bool:
    """Return ``True`` only for a single, side-effect-free ``SELECT``/``WITH`` query."""
    if not sql:
        return False
    stripped = sql.strip().rstrip(";").strip()
    if not stripped:
        return False
    # Reject multi-statement payloads (anything with an interior ';').
    if ";" in stripped:
        return False
    lowered = stripped.lower()
    if not (lowered.startswith("select") or lowered.startswith("with")):
        return False
    if _FORBIDDEN.search(stripped):
        return False
    return True


def _ensure_limit(sql: str, max_rows: int) -> str:
    """Wrap a probe so even an excessive existing LIMIT cannot escape the cap."""

    stripped = sql.strip().rstrip(";").strip()
    if not _LIMIT_RE.search(stripped):
        return f"{stripped} LIMIT {max_rows}"
    return f"SELECT * FROM ({stripped}) AS _bounded_probe LIMIT {max_rows}"


def _table_for_column(
    column: exp.Column,
    tables: list[exp.Table],
    aliases: dict[str, exp.Table],
) -> exp.Table | None:
    qualifier = (column.table or "").casefold()
    if qualifier:
        return aliases.get(qualifier)
    unique = {
        (table.catalog or "", table.db or "", table.name): table for table in tables
    }
    return next(iter(unique.values())) if len(unique) == 1 else None


def _data_policy_error(
    sql: str,
    *,
    dialect: str | None,
    database_name: str,
    allowed_tables: frozenset[str] | None,
) -> str | None:
    """Return why a probe violates catalog policy, or ``None`` when allowed."""

    try:
        tree = sqlglot.parse_one(sql, read=dialect or None)
    except Exception:
        return "rejected: probe policy could not parse SQL"
    if tree is None:
        return "rejected: probe policy could not parse SQL"

    # A bare ``SELECT *`` yields no ``exp.Column`` nodes, so it would otherwise
    # pass without a single column being checked.
    for star in tree.find_all(exp.Star):
        if not isinstance(star.parent, exp.Count):
            return "rejected: wildcard probes are not permitted"

    tables = list(tree.find_all(exp.Table))
    aliases: dict[str, exp.Table] = {}
    for table in tables:
        if table.catalog and table.catalog.casefold() != database_name.casefold():
            return "rejected: table is outside the authorized query scope"
        aliases[table.name.casefold()] = table
        if table.alias:
            aliases[table.alias.casefold()] = table
        if allowed_tables is not None:
            table_name = table.name.casefold()
            scope_key = (
                f"{table.db.casefold()}.{table_name}" if table.db else table_name
            )
            if scope_key not in allowed_tables:
                return "rejected: table is outside the authorized query scope"

    checked: set[tuple[str, str, str | None]] = set()
    for column in tree.find_all(exp.Column):
        if isinstance(column.this, exp.Star):
            return "rejected: wildcard probes are not permitted"
        table = _table_for_column(column, tables, aliases)
        if table is None:
            return "rejected: probe column could not be authorized"
        key = (table.name.casefold(), column.name.casefold(), table.db or None)
        if key in checked:
            continue
        checked.add(key)
        try:
            safe = is_column_safe_for_data_movement(
                database_name=database_name,
                schema_name=table.db or None,
                table_name=table.name,
                column_name=column.name,
            )
        except Exception:
            logger.exception("Probe data-policy lookup failed")
            return "rejected: probe data policy is unavailable"
        if not safe:
            return "rejected: column is PII, unprocessed, or unresolved"
    return None


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
    max_result_columns: int = DB_PROBE_MAX_RESULT_COLUMNS
    timeout_s: float = DB_PROBE_TIMEOUT_S
    enforce_data_policy: bool = False
    database_name: str | None = None
    allowed_tables: frozenset[str] | None = None
    calls: int = field(default=0, init=False)
    log: list[dict[str, Any]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        """Prevent callers from loosening deployment-level data limits."""

        self.max_calls = min(max(self.max_calls, 1), DB_PROBE_MAX_CALLS)
        self.max_rows = min(max(self.max_rows, 1), DB_PROBE_MAX_ROWS)
        self.max_result_columns = min(
            max(self.max_result_columns, 1),
            DB_PROBE_MAX_RESULT_COLUMNS,
        )
        self.timeout_s = min(max(self.timeout_s, 0.1), DB_PROBE_TIMEOUT_S)
        if self.allowed_tables is not None:
            self.allowed_tables = frozenset(
                table.casefold() for table in self.allowed_tables
            )

    @property
    def budget_left(self) -> int:
        return max(0, self.max_calls - self.calls)

    def close(self) -> None:
        """No-op — inline execution holds no resources. Present for API symmetry."""

    def __enter__(self) -> "ProbeExecutor":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def _record_audit(self, entry: dict[str, Any]) -> None:
        """Keep operational metadata without retaining SQL, errors, or values."""

        audit = {
            **entry,
            "sql": None,
            "rows": None,
            "error": "failed" if entry.get("error") else None,
        }
        self.log.append(audit)
        logger.info(
            "Probe audit: database=%s purpose=%s ok=%s rows=%d truncated=%s "
            "elapsed_ms=%d",
            self.database_name
            or str(getattr(self.connector, "database_name", "") or ""),
            entry.get("purpose") or "unspecified",
            bool(entry.get("ok")),
            int(entry.get("row_count") or 0),
            bool(entry.get("truncated")),
            int(entry.get("elapsed_ms") or 0),
        )

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
            self._record_audit(entry)
            return entry

        if self.calls >= self.max_calls:
            entry["error"] = "probe budget exhausted"
            self._record_audit(entry)
            return entry

        if not is_read_only_select(sql):
            entry["error"] = "rejected: not a single read-only SELECT"
            logger.warning("Probe rejected as non-read-only (purpose=%s)", purpose)
            self._record_audit(entry)
            return entry

        if self.enforce_data_policy:
            database_name = self.database_name or str(
                getattr(self.connector, "database_name", "") or ""
            )
            if not database_name:
                entry["error"] = "rejected: database identity is unavailable"
                self._record_audit(entry)
                return entry
            entry["error"] = _data_policy_error(
                sql,
                dialect=getattr(self.connector, "dialect", None),
                database_name=database_name,
                allowed_tables=self.allowed_tables,
            )
            if entry["error"]:
                logger.info("Probe blocked by data policy (purpose=%s)", purpose)
                self._record_audit(entry)
                return entry

        if self.max_rows < 1 or self.max_result_columns < 1:
            entry["error"] = "rejected: invalid probe result limits"
            self._record_audit(entry)
            return entry

        bounded_sql = _ensure_limit(sql, self.max_rows)
        entry["sql"] = bounded_sql
        self.calls += 1

        start = time.perf_counter()
        try:
            df = execute_chat_sql(
                self.connector,
                bounded_sql,
                kind="probe SQL",
                timeout_s=self.timeout_s,
                log_statement=False,
            )
        except Exception as exc:  # noqa: BLE001 — probes must never break the pipeline
            entry["error"] = str(exc)
            entry["elapsed_ms"] = int((time.perf_counter() - start) * 1000)
            logger.info(
                "Probe failed (purpose=%s, error=%s)", purpose, type(exc).__name__
            )
            self._record_audit(entry)
            return entry

        entry["elapsed_ms"] = int((time.perf_counter() - start) * 1000)
        try:
            total = len(df)
            bounded = df.iloc[: self.max_rows, : self.max_result_columns]
            records = bounded.to_dict(orient="records")
        except Exception as exc:  # noqa: BLE001
            entry["error"] = f"result parse failed: {exc}"
            self._record_audit(entry)
            return entry

        entry["ok"] = True
        entry["rows"] = records
        entry["row_count"] = total
        entry["truncated"] = total > self.max_rows or len(df.columns) > len(
            bounded.columns
        )
        self._record_audit(entry)
        return entry


__all__ = ["ProbeExecutor", "allowed_table_scope", "is_read_only_select"]
