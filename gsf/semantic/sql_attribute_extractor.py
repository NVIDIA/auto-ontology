"""LLM extraction of derived business metrics (SqlAttributes) from table columns."""

from __future__ import annotations

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

_ID_SUFFIX = re.compile(r"(id|_id|code|_code|number|_number|num|_num)$", re.IGNORECASE)

_SYSTEM = """\
You are an analytics engineer building a semantic layer for a text-to-SQL \
system. Given a relational table with its columns, propose derived business \
metrics that combine TWO or MORE columns from the same table.

PURPOSE: These metrics are stored in a semantic search index. When a \
business user types a natural-language question (e.g. "What is the profit \
margin per order?"), the system retrieves matching metrics to help generate \
SQL. Only propose metrics that answer questions a real business analyst \
would ask. For each metric you MUST provide a realistic example question \
a user would ask that this metric answers.

The Term below describes the business entity this table represents. \
Metrics should relate to that entity's real-world meaning.

Quality over quantity — it is perfectly fine to return zero metrics if \
none are meaningful.

Each column below is tagged with a role:
  [MEASURE] — a numeric or date value with real business magnitude \
(price, quantity, amount, date, temperature, etc.).
  [ID]      — a surrogate key or primary-key identifier.

Rules:
1. Each metric MUST use at least two MEASURE columns in a meaningful formula \
(sum, difference, ratio, CASE expression, date arithmetic, etc.). \
Do NOT propose metrics that simply rename or alias a single column.
2. NEVER use [ID] columns in arithmetic (division, multiplication, \
subtraction). They are categorical identifiers, not business values. \
They may only appear in CASE/WHERE equality checks (e.g. IS NOT NULL).
3. Audit and system columns (e.g. last_edited_by, valid_from, created_when, \
row_version) are NOT business values — do not use them in arithmetic.
4. Record-versioning timestamp columns (temporal table pattern) track when \
a database ROW was valid, NOT a business event. Do NOT build metrics from \
them (no "validity duration", "is active", "days of validity", etc.).
5. NEVER use string-length functions on text columns to create metrics. \
Character counts of names, emails, URLs, or comments are not business values.
6. NEVER divide by a year or month value — whether from EXTRACT(YEAR ...) \
or from a column literally named "year" or "month". Calendar ordinals \
are not meaningful divisors.
7. The SQL expression must be a valid SELECT ... FROM statement using \
fully qualified column names: schema.table.column. \
Example: SELECT col_a - col_b FROM schema.table
8. Do NOT use aggregate functions (SUM, COUNT, AVG) that require GROUP BY \
— produce row-level expressions only.
9. Metric names must be user-friendly with spaces (e.g. Net Revenue).
10. NO DUPLICATES — every metric must have a unique formula.
11. Return an empty list when no genuinely useful metric can be derived.

Examples of GOOD metrics (note the realistic user question):
  GOOD: question="What is the net revenue per order line?"
        SELECT unitprice * quantity - taxamount AS "Net Revenue" FROM ...
  GOOD: question="How many days until expected delivery?"
        SELECT expecteddeliverydate - orderdate AS "Lead Time Days" FROM ...
  GOOD: question="Which items need restocking?"
        SELECT CASE WHEN quantity > reorderpoint THEN 1 ELSE 0 END FROM ...

Examples of BAD metrics (do NOT propose — no user would ask these):
  BAD: SELECT quantity / supplierid  — nobody asks "divide quantity by ID"
  BAD: SELECT LENGTH(email) - LENGTH(phone)  — nobody asks about character counts
  BAD: SELECT total_sales / year  — calendar year is not a divisor
  BAD: SELECT (validto - validfrom) AS "Duration" — record-versioning, not business"""


# ---------------------------------------------------------------------------
# Column classification
# ---------------------------------------------------------------------------


def _classify_column(col_name: str, table_name: str) -> str:
    """Return a role tag: ID or MEASURE."""
    lower = col_name.lower()
    if _ID_SUFFIX.search(lower):
        return "ID"
    if lower == table_name.lower() + "id":
        return "ID"
    return "MEASURE"


def _format_column_line(
    col: dict[str, Any],
    role: str,
) -> str:
    dtype = col.get("data_type") or "unknown"
    desc = col.get("description")
    line = f"  - {col['name']} ({dtype}) [{role}]"
    if desc:
        line += f" — {desc}"
    return line


# ---------------------------------------------------------------------------
# Post-LLM heuristic filter
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


def _uses_id_in_arithmetic(
    expression: str,
    id_columns: set[str],
) -> bool:
    """Return True if the expression uses any ID column in arithmetic."""
    if not id_columns:
        return False
    expr_lower = expression.lower()
    for col in id_columns:
        pattern = re.compile(
            rf"(?:[*/\-]|\b\w+\s*[*/\-])\s*\b{re.escape(col)}\b"
            rf"|\b{re.escape(col)}\b\s*[*/\-]",
        )
        if pattern.search(expr_lower):
            return True
    return False


def _uses_length_arithmetic(expression: str) -> bool:
    """Return True if LENGTH/LEN/CHAR_LENGTH appears in arithmetic."""
    return bool(_LENGTH_ARITH.search(expression))


def _uses_year_month_divisor(expression: str) -> bool:
    """Return True if a year/month value is used as a divisor."""
    return bool(_DIVIDE_BY_YEAR_MONTH.search(expression))


def _uses_validfrom_to(expression: str) -> bool:
    """Return True if expression references validfrom/validto columns."""
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

    roles: dict[str, str] = {}
    col_lines_parts: list[str] = []
    for col in columns[:40]:
        role = _classify_column(col["name"], table_name)
        roles[col["name"].lower()] = role
        col_lines_parts.append(_format_column_line(col, role))

    col_lines = "\n".join(col_lines_parts)

    id_cols = {name for name, role in roles.items() if role == "ID"}

    measure_count = sum(1 for r in roles.values() if r == "MEASURE")
    if measure_count < 2:
        logger.debug(
            "Table %s has only %d MEASURE columns — skipping SqlAttribute extraction",
            table_name,
            measure_count,
        )
        return []

    prompt = (
        f"Table: {qualified_table}\n"
        f"Business entity (Term): {term.get('name', '')}\n"
        f"Term description: {term.get('description', '')}\n"
        f"Columns:\n{col_lines}\n\n"
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

        if _uses_id_in_arithmetic(proposal.expression, id_cols):
            logger.debug(
                "Dropping proposal %r: uses ID column in arithmetic",
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
