# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run SQL through an injected DB connector and package the result."""

import logging
import re
from typing import Any, Optional

from auto_ontology.retrieval.text_to_sql.chat_sql import execute_chat_sql
from auto_ontology.retrieval.text_to_sql.numeric_types import (
    cast_jsonb_numeric_projections,
    numeric_column_kinds,
    result_values_digest,
)
from auto_ontology.connectors.base import SQLDatabase

logger = logging.getLogger(__name__)


class QueryResponse:
    def __init__(
        self,
        result: list[str],
        sliced: bool,
        error: Optional[str] = None,
        statement: Optional[str] = None,
        numeric_kinds: Optional[list] = None,
        values_digest: Optional[str] = None,
        sql_correction: Optional[str] = None,
    ):
        # Per output column ``"float"`` / ``"decimal"`` / ``None``; lets a caller
        # that needs a predictable numeric type know what this SQL returned.
        self.numeric_kinds = numeric_kinds or []
        # Type-blind digest of the result values (see ``result_values_digest``).
        self.values_digest = values_digest
        self.result = result
        self.sliced = sliced
        self.error = error
        # The SQL as executed, which is not always the SQL as generated -- see
        # ``_sanitize_sql_for_dialect``. Classifying ``error`` means stripping
        # the statement the driver echoed, so it has to be this text.
        self.statement = statement
        # Set when the statement that should be submitted is not the statement
        # that was generated.
        self.sql_correction = sql_correction


def _sanitize_sql_for_dialect(sql: str, dialect: str) -> str:
    """Strip invalid schema qualifiers for dialects that don't use them."""
    if (dialect or "").lower() == "sqlite":
        return re.sub(r"\bPUBLIC\.", "", sql, flags=re.IGNORECASE)
    return sql


def _sql_correction_for(
    original_sql: str,
    executed_sql: str,
    dialect: str,
    frame: Any,
    record_numeric_kinds: bool,
) -> Optional[str]:
    """The statement that should be submitted, or None when *original_sql* is fine.

    A bare JSON numeric projection comes back as text, and a sqlite schema
    qualifier is stripped before execution. Either way the generated SQL is
    not what ran. The corrected text is reported back rather than submitted
    from here.
    """
    if not record_numeric_kinds:
        return None
    rewritten = cast_jsonb_numeric_projections(executed_sql, dialect, frame)
    candidate = rewritten if rewritten else executed_sql
    if candidate.strip() == original_sql.strip():
        return None
    return candidate.strip()


def _deduplicate_result_columns(df: Any) -> Any:
    """Return ``df`` with deterministic, unique result-column labels.

    Existing unique labels are reserved so resolving one duplicate cannot steal
    a label belonging to another result column. For example, ``ID, ID, ID__2``
    becomes ``ID, ID__3, ID__2``.
    """
    columns = list(df.columns)
    reserved = set(columns)
    used: set[Any] = set()
    occurrences: dict[Any, int] = {}
    normalized: list[Any] = []

    for column in columns:
        occurrences[column] = occurrences.get(column, 0) + 1
        if column not in used:
            normalized.append(column)
            used.add(column)
            continue

        suffix = occurrences[column]
        candidate = f"{column}__{suffix}"
        while candidate in reserved or candidate in used:
            suffix += 1
            candidate = f"{column}__{suffix}"
        normalized.append(candidate)
        used.add(candidate)

    if normalized != columns:
        df = df.copy()
        df.columns = normalized
    return df


def _run_sql(
    sql: str,
    connector: SQLDatabase | None,
    *,
    record_numeric_kinds: bool = False,
) -> QueryResponse:
    """Execute SQL against the supplied ``connector``.

    Caller is responsible for picking the right connector for ``sql``
    (e.g. by matching ``relevant_tables[*].database_name`` against
    ``connector.database_name``).
    """
    if connector is None:
        return QueryResponse(
            result=None, sliced=False, error="No connector available to execute SQL."
        )
    try:
        original_sql = sql
        dialect = getattr(connector, "dialect", "")
        sql = _sanitize_sql_for_dialect(sql, dialect)
        df = execute_chat_sql(connector, sql)
        sql_correction = _sql_correction_for(
            original_sql, sql, dialect, df, record_numeric_kinds
        )
        df = _deduplicate_result_columns(df)
        numeric_kinds = numeric_column_kinds(df) if record_numeric_kinds else []
        values_digest = result_values_digest(df) if record_numeric_kinds else None
        payload = (
            df.to_json(orient="records", date_format="iso", default_handler=str)
            if len(df)
            else "[]"
        )
    except Exception as e:
        logger.exception("SQL execution failed (injected connector)")
        return QueryResponse(result=None, sliced=False, error=str(e), statement=sql)
    return QueryResponse(
        result=[payload],
        sliced=False,
        error=None,
        statement=sql,
        numeric_kinds=numeric_kinds,
        values_digest=values_digest,
        sql_correction=sql_correction,
    )
