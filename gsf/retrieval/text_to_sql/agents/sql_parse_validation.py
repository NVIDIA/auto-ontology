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
from typing import Dict, Any

import sqlglot
from sqlglot import expressions as exp

from nemo_retriever.tabular_data.ingestion.services.queries import parse_query_single
from gsf.retrieval.text_to_sql.base import BaseAgent
from gsf.retrieval.text_to_sql.state import AgentState
from gsf.retrieval.data_access.custom_analyses import get_custom_analyses_ids
from gsf.retrieval.data_access.graph_schemas import (
    fetch_all_schema_ids,
    get_schemas_by_ids,
)

logger = logging.getLogger(__name__)

# sqlglot dialect names differ slightly from our connector dialect strings.
_SQLGLOT_DIALECTS = {
    "sqlite": "sqlite",
    "postgres": "postgres",
    "postgresql": "postgres",
    "snowflake": "snowflake",
    "duckdb": "duckdb",
    "mysql": "mysql",
    "heavydb": "postgres",
}


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

    read = _SQLGLOT_DIALECTS.get((dialect or "").strip().lower())
    try:
        parsed = sqlglot.parse_one(sql, read=read)
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
        schemas_ids = fetch_all_schema_ids()
        schemas = get_schemas_by_ids(schemas_ids)

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

        degenerate_dialect = dialects[0] if dialects else None
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
