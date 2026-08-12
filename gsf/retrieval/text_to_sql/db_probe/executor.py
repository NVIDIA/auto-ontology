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
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from gsf.connectors.base import SQLDatabase

from gsf.retrieval.text_to_sql.db_probe.config import (
    DB_PROBE_MAX_CALLS,
    DB_PROBE_MAX_ROWS,
)

from gsf.retrieval.text_to_sql.chat_sql import execute_chat_sql

logger = logging.getLogger(__name__)

# Statement keywords that must never appear in a probe (mutations / side effects).
_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|truncate|grant|revoke|"
    r"attach|detach|pragma|vacuum|reindex|merge|call|exec|execute|copy|into)\b",
    re.IGNORECASE,
)
_LIMIT_RE = re.compile(r"\blimit\b", re.IGNORECASE)


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
    """Append ``LIMIT max_rows`` when the query has no explicit LIMIT."""
    stripped = sql.strip().rstrip(";").strip()
    if _LIMIT_RE.search(stripped):
        return stripped
    return f"{stripped} LIMIT {max_rows}"


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

        if not is_read_only_select(sql):
            entry["error"] = "rejected: not a single read-only SELECT"
            logger.warning("Probe rejected (not read-only): %s", sql)
            self.log.append(entry)
            return entry

        bounded_sql = _ensure_limit(sql, self.max_rows)
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
