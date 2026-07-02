"""LLM extraction of derived business metrics (SqlAttributes) from table columns."""

from __future__ import annotations

import json
import re
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
You are an analytics engineer building a semantic layer for a text-to-SQL system.

CONTEXT: You are given the columns that are semantically mapped to a specific \
business Term — these are the meaningful, non-key columns that describe the entity. \
Use them to propose derived business metrics.

PURPOSE: Metrics are stored in a semantic search index. When a business user asks \
a natural-language question, the system retrieves matching metrics to help generate \
SQL. For each metric you MUST provide a realistic example question a user would ask.

PRIMARY RULE: Only propose metrics that are recognised standard measures in global \
business ontologies and industry KPI frameworks (e.g. schema.org, FIBO, GS1, \
SCOR, financial reporting standards). Think: margin, utilisation, lead time, \
fill rate, cycle time, turnover, coverage ratio. If a metric would not appear in \
an industry ontology or standard KPI catalogue, do not propose it.

Quality over quantity — it is perfectly fine to return zero metrics.

Rules:
1. Each metric MUST combine at least two columns with an arithmetic operator, \
CASE expression, or function. NEVER propose a metric that simply renames or \
aliases a single column (e.g. SELECT col AS "Label" is forbidden unless it \
also has a WHERE clause that adds meaningful filtering).
2. NEVER use a literal value in a WHERE clause unless that exact value appears in \
the column's listed sample values. If no sample values are listed for a column, \
do not filter on it at all.
3. Audit and system columns (e.g. last_edited_by, created_when, row_version) are \
NOT business values — do not use them.
4. Record-versioning timestamp columns (valid_from, valid_to, validfrom, validto) \
track when a database row was valid, NOT a business event. Do NOT build metrics \
from them.
5. NEVER use string-length functions on text columns. Character counts are not \
business values.
6. NEVER divide by a year or month value (whether from EXTRACT or a column named \
"year" / "month"). Calendar ordinals are not meaningful divisors.
7. The SQL expression must be a valid SELECT ... FROM statement using fully \
qualified column names: schema.table.column.
8. Do NOT use aggregate functions (SUM, COUNT, AVG) that require GROUP BY — \
produce row-level expressions only.
9. Metric names must be user-friendly with spaces (e.g. Net Revenue, Lead Time Days).
10. NO DUPLICATES — every metric must have a unique formula.
11. Return an empty list when no ontology-standard metric can be derived.

Examples of GOOD metrics:
  GOOD: question="What is the gross margin per order line?"
        SELECT (unitprice - unitcost) / NULLIF(unitprice, 0) AS "Gross Margin" FROM ...
  GOOD: question="How many days until expected delivery?"
        SELECT expecteddeliverydate - orderdate AS "Lead Time Days" FROM ...
  GOOD: question="What is the inventory fill rate?"
        SELECT quantityshipped / NULLIF(quantityordered, 0) AS "Fill Rate" FROM ...

Examples of BAD metrics (do NOT propose):
  BAD: SELECT LENGTH(email) - LENGTH(phone)  — character counts, not in any ontology
  BAD: SELECT total_sales / year  — calendar year is not a divisor
  BAD: SELECT (validto - validfrom) AS "Duration"  — record versioning, not business
  BAD: SELECT amount WHERE status = 'ACTIVE'  — "ACTIVE" not in sample values"""


# ---------------------------------------------------------------------------
# Column formatting
# ---------------------------------------------------------------------------


def _format_column_line(col: dict[str, Any]) -> str:
    """Format a single column for the LLM prompt, including sample values if present."""
    dtype = col.get("data_type") or "unknown"
    desc = col.get("description")
    line = f"  - {col['name']} ({dtype})"
    if desc:
        line += f" — {desc}"

    raw_sv = col.get("sample_values")
    if raw_sv:
        if isinstance(raw_sv, str):
            try:
                raw_sv = json.loads(raw_sv)
            except Exception:
                raw_sv = None
        if isinstance(raw_sv, list) and raw_sv:
            samples = ", ".join(repr(v) for v in raw_sv[:5])
            line += f" [sample values: {samples}]"

    return line


# ---------------------------------------------------------------------------
# Post-LLM heuristic filters
# ---------------------------------------------------------------------------

_DIVIDE_BY_YEAR_MONTH = re.compile(
    r"/\s*(?:"
    r"extract\s*\(\s*(?:year|month)\s+from\b"
    r"|(?:\w+\.)*\w*\.?\byear\b"
    r"|(?:\w+\.)*\w*\.?\bmonth\b"
    r")",
    re.IGNORECASE,
)

_LENGTH_ARITH = re.compile(
    r"(?:length|len|char_length)\s*\(.*?\)\s*[+\-*/]"
    r"|[+\-*/]\s*(?:length|len|char_length)\s*\(",
    re.IGNORECASE,
)

_VALIDFROM_TO = re.compile(
    r"\bvalid_?(?:from|to)\b",
    re.IGNORECASE,
)

# Matches a SELECT with a single bare column (no operators) and no WHERE clause:
# e.g. SELECT schema.table.col AS "Label" FROM ...
_SINGLE_COLUMN_ALIAS = re.compile(
    r"^\s*SELECT\s+(?:\w+\.)*\w+\s+AS\b(?!.*\bWHERE\b)",
    re.IGNORECASE | re.DOTALL,
)

_WHERE_WITH_LITERAL = re.compile(
    r"\bWHERE\b.+?(?:=\s*'|IN\s*\()",
    re.IGNORECASE | re.DOTALL,
)

_QUOTED_LITERAL = re.compile(r"'([^']*)'")


def _collect_sample_values(columns: list[dict[str, Any]]) -> set[str]:
    """Return a lowercased flat set of all sample values across the given columns."""
    values: set[str] = set()
    for col in columns:
        raw = col.get("sample_values")
        if not raw:
            continue
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except Exception:
                continue
        if isinstance(raw, list):
            values.update(str(v).lower() for v in raw)
    return values


def _where_literals_valid(expression: str, columns: list[dict[str, Any]]) -> bool:
    """Return False if the expression has a WHERE literal not backed by sample values."""
    if not _WHERE_WITH_LITERAL.search(expression):
        return True

    all_samples = _collect_sample_values(columns)
    if not all_samples:
        return False  # WHERE with literals but no sample values at all

    literals = _QUOTED_LITERAL.findall(expression)
    if not literals:
        return True  # numeric comparisons — allow (hard to validate without schema)

    return all(lit.lower() in all_samples for lit in literals)


def _uses_length_arithmetic(expression: str) -> bool:
    return bool(_LENGTH_ARITH.search(expression))


def _uses_year_month_divisor(expression: str) -> bool:
    return bool(_DIVIDE_BY_YEAR_MONTH.search(expression))


def _uses_validfrom_to(expression: str) -> bool:
    return bool(_VALIDFROM_TO.search(expression))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def extract_sql_attributes(
    table: dict[str, Any],
    columns: list[dict[str, Any]],
    schema_name: str | None,
    term: dict[str, Any],
    database_name: str,
) -> list[SqlAttributeProposal]:
    """Ask the LLM to propose derived metrics, validate each SQL, return survivors."""
    if len(columns) < 2:
        return []

    table_name = table["name"]
    qualified_table = f"{schema_name}.{table_name}" if schema_name else table_name

    col_lines = "\n".join(_format_column_line(col) for col in columns[:40])

    prompt = (
        f"Table: {qualified_table}\n"
        f"Business entity (Term): {term.get('name', '')}\n"
        f"Term description: {term.get('description', '')}\n"
        f"Columns (semantically mapped to this term):\n{col_lines}\n\n"
        f"Propose derived metrics using schema={schema_name or 'public'} "
        f"and table={table_name} for qualified references."
    )

    result = invoke_with_structured_output(
        get_llm_client(temperature=0.0, max_tokens=4096),
        [SystemMessage(content=_SYSTEM), HumanMessage(content=prompt)],
        SqlAttributeExtractionResult,
    )
    if result is None:
        logger.warning(
            "LLM returned no SqlAttribute proposals for table %s", table_name
        )
        return []

    dialects = get_dialects()
    schemas = get_schemas()

    valid: list[SqlAttributeProposal] = []
    seen_expressions: set[str] = set()
    for proposal in result.metrics:
        normalized = " ".join(proposal.expression.lower().split())
        if normalized in seen_expressions:
            logger.debug("Dropping duplicate proposal %r", proposal.name)
            continue
        seen_expressions.add(normalized)

        if not _where_literals_valid(proposal.expression, columns):
            logger.debug(
                "Dropping proposal %r: WHERE literal not in sample values",
                proposal.name,
            )
            continue

        if _SINGLE_COLUMN_ALIAS.match(proposal.expression):
            logger.debug(
                "Dropping proposal %r: single-column alias with no computation",
                proposal.name,
            )
            continue

        if _uses_length_arithmetic(proposal.expression):
            logger.debug(
                "Dropping proposal %r: uses LENGTH() in arithmetic",
                proposal.name,
            )
            continue

        if _uses_year_month_divisor(proposal.expression):
            logger.debug(
                "Dropping proposal %r: divides by year/month value",
                proposal.name,
            )
            continue

        if _uses_validfrom_to(proposal.expression):
            logger.debug(
                "Dropping proposal %r: uses validfrom/validto versioning columns",
                proposal.name,
            )
            continue

        try:
            validate_sql(proposal.expression, dialects, schemas)
            valid.append(proposal)
        except (SqlParseError, Exception):
            logger.debug(
                "Dropping invalid SqlAttribute proposal %r for table %s: bad SQL",
                proposal.name,
                table_name,
            )

    return valid
