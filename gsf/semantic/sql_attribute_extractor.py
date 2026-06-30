"""LLM extraction of derived business metrics (SqlAttributes) from table columns."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from gsf.semantic.models import (
    SqlAttributeExtractionResult,
    SqlAttributeProposal,
)
from gsf.server.sql_utils import SqlParseError, get_dialects, get_schemas, validate_sql
from gsf.utils.llm_invoke import get_llm_client, invoke_with_structured_output

logger = logging.getLogger(__name__)

_SYSTEM = """\
You are an analytics engineer. Given a relational table with its columns, \
propose up to 5 non-trivial derived business metrics that combine TWO or MORE \
columns from the same table.

Rules:
1. Each metric MUST use at least two columns in a meaningful formula \
(sum, difference, ratio, conditional aggregation, CASE expression, etc.). \
Do NOT propose metrics that simply rename or alias a single column.
2. The SQL expression must be a valid SELECT statement using fully qualified \
column names in the format schema.table.column (the schema and table names \
are provided below). Example: SELECT col_a - col_b FROM schema.table
3. Do NOT use aggregate functions (SUM, COUNT, AVG, etc.) that would require \
GROUP BY — produce row-level expressions only.
4. Metric names must be user-friendly with spaces between words \
(e.g. Net Revenue, Profit Margin, Days To Ship).
5. Provide a concise business description for each metric.
6. Return an empty list if the table has fewer than two numeric/date columns \
or no meaningful multi-column metric can be derived.
7. Focus on metrics that would be genuinely useful for business analysis."""


def _format_column_line(col: dict[str, Any]) -> str:
    dtype = col.get("data_type") or "unknown"
    desc = col.get("description")
    line = f"  - {col['name']} ({dtype})"
    if desc:
        line += f" — {desc}"
    return line


def extract_sql_attributes(
    table: dict[str, Any],
    ctx: dict[str, Any],
    schema_name: str | None,
    term: dict[str, Any],
    database_name: str,
) -> list[SqlAttributeProposal]:
    """Ask the LLM to propose derived metrics, validate each SQL, return survivors."""
    columns = ctx.get("columns", [])
    if len(columns) < 2:
        return []

    qualified_table = f"{schema_name}.{table['name']}" if schema_name else table["name"]
    col_lines = "\n".join(_format_column_line(c) for c in columns[:40])

    prompt = (
        f"Table: {qualified_table}\n"
        f"Business entity (Term): {term.get('name', '')}\n"
        f"Term description: {term.get('description', '')}\n"
        f"Columns:\n{col_lines}\n\n"
        f"Propose derived metrics using schema={schema_name or 'public'} "
        f"and table={table['name']} for qualified references."
    )

    result = invoke_with_structured_output(
        get_llm_client(temperature=0.0, max_tokens=4096),
        [SystemMessage(content=_SYSTEM), HumanMessage(content=prompt)],
        SqlAttributeExtractionResult,
    )
    if result is None:
        logger.warning(
            "LLM returned no SqlAttribute proposals for table %s", table["name"]
        )
        return []

    dialects = get_dialects(database_name)
    schemas = get_schemas(database_name)

    valid: list[SqlAttributeProposal] = []
    for proposal in result.metrics:
        try:
            validate_sql(proposal.expression, dialects, schemas)
            valid.append(proposal)
        except (SqlParseError, Exception):
            logger.debug(
                "Dropping invalid SqlAttribute proposal %r for table %s: bad SQL",
                proposal.name,
                table["name"],
            )
    return valid
