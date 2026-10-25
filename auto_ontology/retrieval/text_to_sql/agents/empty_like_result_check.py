# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Post-execution result integrity checks.

Three checks run after successful SQL execution:

1. NULL result values: when a projected row-level expression returns NULL, add
   the corresponding ``IS NOT NULL`` predicate deterministically and execute the
   corrected SQL once before involving reconstruction.

2. All-NULL JSONB columns: if a JSONB-accessed computed column returns NULL for
   every row, the path is likely wrong (wrong nesting level or wrong key name).
   Routes once to reconstruction with a targeted error — no LLM call needed.

3. Empty-result LIKE check: if execution returns zero rows and the SQL contains
   LIKE/ILIKE predicates, uses an LLM to identify non-essential filters and
   routes once to reconstruction with explicit removal guidance.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.messages import SystemMessage
from pydantic import BaseModel, ConfigDict, Field
import sqlglot
from sqlglot import exp

from auto_ontology.retrieval.text_to_sql.base import BaseAgent
from auto_ontology.retrieval.text_to_sql.connector_routing import (
    resolve_connector_from_tables,
)
from auto_ontology.retrieval.text_to_sql.prompts import (
    create_empty_like_check_prompt,
    format_dual_question_block,
)
from auto_ontology.retrieval.text_to_sql.state import (
    AgentState,
    get_original_question,
    get_question_for_processing,
)
from auto_ontology.utils.llm_invoke import invoke_with_structured_output

logger = logging.getLogger(__name__)

_LIKE_PATTERN = re.compile(r"\bI?LIKE\b", re.IGNORECASE)

# Matches a JSONB leaf access (->>) followed (within the same column expression) by AS alias.
# Covers: (col->>'key')::type AS alias  and  col->'a'->>'b' AS alias
# Note: no \b after closing quote — "'" is non-word so \b never fires there.
_JSONB_ALIAS_RE = re.compile(r"->>'[^']+'\s*[^,\n]*?\bAS\s+(\w+)", re.IGNORECASE)

_EMPTY_LIKE_SYSTEM_PROMPT = """You classify LIKE/ILIKE predicates in SQL queries.
Be conservative: only mark a predicate non-essential when it clearly filters on
a feature, preference, or additional attribute rather than the main subject."""


class EmptyLikeCheckModel(BaseModel):
    """LLM classification of LIKE/ILIKE predicates after an empty result set."""

    model_config = ConfigDict(extra="forbid")

    non_essential_like_predicates: list[str] = Field(
        default_factory=list,
        description=(
            "LIKE/ILIKE predicates to remove or relax. Each entry should quote "
            "the full predicate exactly as it appears in the SQL."
        ),
    )
    essential_like_predicates: list[str] = Field(
        default_factory=list,
        description=(
            "LIKE/ILIKE predicates that identify the main subject and must be "
            "preserved. Each entry should quote the full predicate exactly as "
            "it appears in the SQL."
        ),
    )


def _is_empty_db_result(result: Any) -> bool:
    if result is None:
        return False
    if isinstance(result, list):
        if not result:
            return True
        if len(result) == 1 and isinstance(result[0], str):
            try:
                parsed = json.loads(result[0])
                return isinstance(parsed, list) and len(parsed) == 0
            except json.JSONDecodeError:
                return result[0].strip() in ("[]", "")
        return all(_is_empty_db_result(item) for item in result)
    if isinstance(result, str):
        stripped = result.strip()
        if stripped in ("[]", ""):
            return True
        try:
            parsed = json.loads(stripped)
            return isinstance(parsed, list) and len(parsed) == 0
        except json.JSONDecodeError:
            return False
    return False


def _sql_has_like(sql: str) -> bool:
    return bool(_LIKE_PATTERN.search(sql or ""))


def _parse_db_rows(db_result: Any) -> list[dict]:
    """Best-effort parse of db_result into a list of row dicts."""
    if db_result is None:
        return []
    if isinstance(db_result, list):
        if db_result and isinstance(db_result[0], dict):
            return db_result
        if len(db_result) == 1 and isinstance(db_result[0], str):
            try:
                parsed = json.loads(db_result[0])
                if isinstance(parsed, list):
                    return [r for r in parsed if isinstance(r, dict)]
            except (json.JSONDecodeError, TypeError):
                pass
    if isinstance(db_result, str):
        try:
            parsed = json.loads(db_result)
            if isinstance(parsed, list):
                return [r for r in parsed if isinstance(r, dict)]
        except (json.JSONDecodeError, TypeError):
            pass
    return []


def _null_result_columns(db_result: Any) -> list[str]:
    """Return result-column labels containing at least one SQL NULL."""
    rows = _parse_db_rows(db_result)
    if not rows:
        return []
    return list(
        dict.fromkeys(
            str(column)
            for row in rows
            for column, value in row.items()
            if value is None
        )
    )


def _row_level_projection(expression: exp.Expression) -> exp.Expression | None:
    """Return a projection expression that is safe to repeat in ``WHERE``."""
    source = expression.this if isinstance(expression, exp.Alias) else expression
    if isinstance(source, exp.Star):
        return None
    if any(
        source.find(node_type) is not None
        for node_type in (exp.AggFunc, exp.Window, exp.Subquery, exp.Select)
    ):
        return None
    return source.copy()


def add_not_null_filters_for_result(
    sql: str,
    db_result: Any,
    dialect: str | None = None,
) -> tuple[str | None, list[str]]:
    """Add predicates for projected row-level expressions that returned NULL.

    The rewrite is deliberately conservative. It handles a single top-level
    SELECT whose output labels can be mapped back to projections. Aggregate,
    window, wildcard, subquery, and set-operation outputs are left to
    reconstruction because moving those expressions into ``WHERE`` can change
    semantics or produce invalid SQL.
    """
    null_columns = _null_result_columns(db_result)
    if not null_columns:
        return None, []

    try:
        tree = sqlglot.parse_one(sql, read=dialect or None)
    except Exception:
        return None, null_columns
    if not isinstance(tree, exp.Select):
        return None, null_columns

    projections = {
        projection.alias_or_name.casefold(): projection
        for projection in tree.expressions
        if projection.alias_or_name
    }
    predicates: list[exp.Expression] = []
    filtered_columns: list[str] = []
    for column in null_columns:
        projection = projections.get(column.casefold())
        if projection is None:
            continue
        source = _row_level_projection(projection)
        if source is None:
            continue
        predicates.append(exp.Is(this=source, expression=exp.Null(), negate=True))
        filtered_columns.append(column)

    if not predicates:
        return None, null_columns

    rewritten: exp.Expression = tree
    for predicate in predicates:
        rewritten = rewritten.where(predicate)
    try:
        return rewritten.sql(dialect=dialect or None), filtered_columns
    except Exception:
        return None, null_columns


def _null_jsonb_aliases(sql: str, db_result: Any) -> list[str]:
    """Return aliases of JSONB-accessed SELECT columns where every row value is NULL.

    Only fires when the SQL contains JSONB access operators, the result is
    non-empty, and at least one aliased JSONB column is all-NULL — a strong
    signal that the path is navigating to the wrong nesting level.
    """
    if not sql or "->>" not in sql:
        return []
    aliases = list(
        dict.fromkeys(
            m.group(1).lower() for m in _JSONB_ALIAS_RE.finditer(sql) if m.group(1)
        )
    )
    if not aliases:
        return []
    rows = _parse_db_rows(db_result)
    if not rows:
        return []
    return [
        alias
        for alias in aliases
        if (vals := [row.get(alias) for row in rows if alias in row])
        and all(v is None for v in vals)
    ]


def _build_null_jsonb_error(null_aliases: list[str]) -> str:
    cols = ", ".join(f"`{a}`" for a in null_aliases)
    return (
        f"SQL executed but column(s) {cols} returned NULL for every row. "
        "This means the JSONB path is wrong. Two possible causes: "
        "(1) wrong nesting level — the key exists but is inside an intermediate object, "
        "so flat access (->>'key') should be (->'container'->>'key'); "
        "(2) wrong key name — the key is abbreviated or named differently in the schema "
        "than expected. Check the column schema and correct whichever applies."
    )


def _build_null_result_error(null_columns: list[str]) -> str:
    columns = ", ".join(f"`{column}`" for column in null_columns)
    return (
        f"SQL executed successfully but returned NULL in result column(s) {columns}. "
        "The final result must not contain NULL values. Add an IS NOT NULL filter "
        "for each nullable projected field without changing the requested output, "
        "grain, joins, or other filters."
    )


def _get_sql_code(path_state: dict) -> str:
    sql_code = path_state.get("sql_code")
    if sql_code and str(sql_code).strip():
        return str(sql_code)
    llm_result = path_state.get("sql_generation_result")
    return getattr(llm_result, "sql_code", "") or ""


def _set_sql_code(path_state: dict, sql_code: str) -> None:
    """Store a self-applied *sql_code* onto every field a downstream
    consumer might read it from.

    ``path_state["sql_code"]`` and ``path_state["sql_generation_result"].sql_code``
    are two separate copies read with different precedence by different
    consumers (``_get_sql_code`` above — used by ``sql_execution`` and the
    other combined_precheck sub-checks — prefers ``sql_code``; ``response.py``'s
    ``calculation_response``, which builds what actually gets submitted,
    reads only ``sql_generation_result.sql_code``). Writing just one of them
    left the two out of sync: a self-applied precheck fix (e.g. a
    missing-bridge join rewrite) could land in ``sql_generation_result`` only,
    so ``sql_execution`` validated the pre-fix SQL while submission sent the
    post-fix one — never executed or validated at all. Always write both so
    they can't diverge, regardless of which one a given reader prefers.
    """
    path_state["sql_code"] = sql_code
    response = path_state.get("sql_generation_result")
    if response is not None:
        path_state["sql_generation_result"] = response.model_copy(
            update={"sql_code": sql_code}
        )


def build_empty_like_reconstruction_error(
    essential: list[str],
    non_essential: list[str],
) -> str:
    essential_lines = "\n".join(f"- {item}" for item in essential) or "- (none)"
    non_essential_lines = "\n".join(f"- {item}" for item in non_essential)
    return (
        "SQL executed successfully but returned an empty result set. The SQL "
        "contains LIKE/ILIKE filters. Reconstruct the SQL by removing or "
        "relaxing ONLY the non-essential LIKE/ILIKE predicates listed below. "
        "Never remove predicates that identify the main subject of the question. "
        "Do not remove joins, numeric thresholds, or other essential filters.\n\n"
        "Remove these non-essential LIKE/ILIKE predicates:\n"
        f"{non_essential_lines}\n\n"
        "Preserve these essential LIKE/ILIKE predicates:\n"
        f"{essential_lines}"
    )


class EmptyLikeResultCheckAgent(BaseAgent):
    """Use LLM to detect removable LIKE filters after an empty execution result."""

    def __init__(self) -> None:
        super().__init__("empty_like_result_check")

    def validate_input(self, state: AgentState) -> bool:
        path_state = state.get("path_state", {})
        if path_state.get("sql_response_from_db") is None:
            self.logger.warning("No SQL execution result found for empty LIKE check")
            return False
        return True

    def execute(self, state: AgentState) -> dict[str, Any]:
        path_state = dict(state.get("path_state", {}))
        sql_code = _get_sql_code(path_state)
        db_result = path_state.get("sql_response_from_db")

        if not _is_empty_db_result(db_result):
            # Non-empty result: check for JSONB columns that are all-NULL,
            # which indicates a wrong nesting path rather than an empty table.
            if not path_state.get("null_jsonb_retry_attempted"):
                null_aliases = _null_jsonb_aliases(sql_code, db_result)
                if null_aliases:
                    path_state["null_jsonb_retry_attempted"] = True
                    path_state["error"] = _build_null_jsonb_error(null_aliases)
                    self.logger.info(
                        "All-NULL JSONB column(s) %s — routing to reconstruction",
                        null_aliases,
                    )
                    return {"decision": "invalid_sql", "path_state": path_state}

            null_columns = _null_result_columns(db_result)
            if null_columns:
                if path_state.get("null_result_filter_retry_sql") == sql_code.strip():
                    path_state["error"] = _build_null_result_error(null_columns)
                    self.logger.info(
                        "NULL values remained after deterministic filtering in %s — "
                        "routing to reconstruction",
                        null_columns,
                    )
                    return {"decision": "invalid_sql", "path_state": path_state}

                connectors = state.get("connectors") or []
                relevant_tables = path_state.get("relevant_tables", [])
                connector = resolve_connector_from_tables(relevant_tables, connectors)
                dialect = getattr(connector, "dialect", None)
                rewritten_sql, filtered_columns = add_not_null_filters_for_result(
                    sql_code, db_result, dialect
                )
                if rewritten_sql and rewritten_sql.strip() != sql_code.strip():
                    _set_sql_code(path_state, rewritten_sql)
                    path_state["null_result_filter_retry_sql"] = rewritten_sql.strip()
                    path_state["sql_response_from_db"] = None
                    self.logger.info(
                        "Added NOT NULL result filters for %s — re-executing SQL",
                        filtered_columns,
                    )
                    return {"decision": "retry_sql", "path_state": path_state}

                path_state["error"] = _build_null_result_error(null_columns)
                self.logger.info(
                    "Could not safely add NOT NULL filters for %s — routing to "
                    "reconstruction",
                    null_columns,
                )
                return {"decision": "invalid_sql", "path_state": path_state}
            return {"decision": "valid_sql", "path_state": path_state}

        if state.get("shorten_answer", False):
            return {"decision": "valid_sql", "path_state": path_state}

        if not _sql_has_like(sql_code):
            self.logger.info("Empty result without LIKE/ILIKE — passing through")
            return {"decision": "valid_sql", "path_state": path_state}

        if path_state.get("empty_like_retry_attempted"):
            self.logger.info(
                "Empty LIKE retry already attempted — passing through empty result"
            )
            return {"decision": "valid_sql", "path_state": path_state}

        original_question = get_original_question(state)
        sanitized_question = get_question_for_processing(state)
        question_block = format_dual_question_block(
            original_question, sanitized_question
        )

        prompt = create_empty_like_check_prompt(question_block, sql_code)
        messages = [
            SystemMessage(content=_EMPTY_LIKE_SYSTEM_PROMPT),
            SystemMessage(content=prompt),
        ]

        try:
            result = invoke_with_structured_output(
                state["llm"], messages, EmptyLikeCheckModel
            )
        except Exception as exc:
            self.logger.warning("Empty LIKE LLM check failed: %s", exc)
            return {"decision": "valid_sql", "path_state": path_state}

        if result is None:
            self.logger.warning("Empty LIKE LLM check returned None")
            return {"decision": "valid_sql", "path_state": path_state}

        non_essential = [
            item.strip()
            for item in result.non_essential_like_predicates
            if item.strip()
        ]
        essential = [
            item.strip() for item in result.essential_like_predicates if item.strip()
        ]

        if not non_essential:
            self.logger.info(
                "LLM found no non-essential LIKE/ILIKE predicates — passing through"
            )
            return {"decision": "valid_sql", "path_state": path_state}

        path_state["empty_like_retry_attempted"] = True
        path_state["error"] = build_empty_like_reconstruction_error(
            essential, non_essential
        )
        self.logger.info(
            "Empty result — routing to reconstruction to remove %d non-essential "
            "LIKE/ILIKE predicate(s)",
            len(non_essential),
        )
        return {"decision": "invalid_sql", "path_state": path_state}
