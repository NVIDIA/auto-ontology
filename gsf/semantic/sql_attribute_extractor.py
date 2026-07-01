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

_ID_SUFFIX = re.compile(r"(id|_id)$", re.IGNORECASE)

_SYSTEM = """\
You are an analytics engineer. Given a relational table with its columns, \
propose non-trivial derived business metrics that combine TWO or MORE \
columns from the same table. Only propose metrics that are genuinely \
valuable — quality over quantity. It is perfectly fine to return zero \
metrics if none are meaningful.

Each column below is tagged with a role:
  [MEASURE] — a numeric or date value with real business magnitude \
(price, quantity, amount, date, temperature, population, etc.).
  [ID]      — a surrogate key or primary-key identifier.
  [FK]      — a foreign-key reference to another table.

Rules:
1. Each metric MUST use at least two MEASURE columns in a meaningful formula \
(sum, difference, ratio, CASE expression, date arithmetic, etc.). \
Do NOT propose metrics that simply rename or alias a single column.
2. NEVER use [ID] or [FK] columns in arithmetic (division, multiplication, \
subtraction). They are categorical identifiers, not business values. \
They may only appear in CASE/WHERE equality checks (e.g. IS NOT NULL).
3. Audit and system columns (e.g. last_edited_by, valid_from, created_when, \
row_version) are NOT business values — do not use them in arithmetic.
4. NEVER divide by EXTRACT(YEAR ...) or EXTRACT(MONTH ...) — the calendar \
year or month number is not a meaningful divisor.
5. The SQL expression must be a valid SELECT ... FROM statement using \
fully qualified column names: schema.table.column. \
Example: SELECT col_a - col_b FROM schema.table
6. Do NOT use aggregate functions (SUM, COUNT, AVG) that require GROUP BY \
— produce row-level expressions only.
7. Metric names must be user-friendly with spaces (e.g. Net Revenue).
8. NO DUPLICATES — every metric must have a unique formula.
9. Return an empty list when no genuinely useful metric can be derived.

Examples of GOOD metrics:
  GOOD: SELECT unitprice * quantity - taxamount AS "Net Revenue" FROM ...
  GOOD: SELECT expecteddeliverydate - orderdate AS "Lead Time Days" FROM ...
  GOOD: SELECT CASE WHEN quantity > reorderpoint THEN 1 ELSE 0 END FROM ...

Examples of BAD metrics (do NOT propose these):
  BAD: SELECT quantity / supplierid  — dividing by an ID is meaningless
  BAD: SELECT lasteditedby * price   — audit column in arithmetic
  BAD: SELECT length(name) / id      — string length ratio to an ID
  BAD: SELECT quantity / EXTRACT(YEAR FROM orderdate) — year as divisor"""


# ---------------------------------------------------------------------------
# Column classification
# ---------------------------------------------------------------------------


def _classify_column(
    col_name: str,
    fk_columns: set[str],
    table_name: str,
) -> str:
    """Return a role tag: FK, ID, or MEASURE."""
    lower = col_name.lower()
    if lower in fk_columns:
        return "FK"
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


def _build_fk_set(ctx: dict[str, Any]) -> set[str]:
    """Collect FK source column names from context (lowercased)."""
    fk_cols: set[str] = set()
    for fk in ctx.get("fks", []):
        src = fk.get("source_column")
        if src:
            fk_cols.add(src.lower())
    return fk_cols


# ---------------------------------------------------------------------------
# Post-LLM heuristic filter
# ---------------------------------------------------------------------------

_DIVIDE_BY_YEAR_MONTH = re.compile(
    r"/\s*(?:extract\s*\(\s*(?:year|month)\s+from\b)",
    re.IGNORECASE,
)


def _uses_id_in_arithmetic(
    expression: str,
    id_columns: set[str],
) -> bool:
    """Return True if the expression uses any ID/FK column in arithmetic."""
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


def _uses_year_month_divisor(expression: str) -> bool:
    """Return True if EXTRACT(YEAR/MONTH ...) is used as a divisor."""
    return bool(_DIVIDE_BY_YEAR_MONTH.search(expression))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


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

    fk_columns = _build_fk_set(ctx)
    table_name = table["name"]
    qualified_table = f"{schema_name}.{table_name}" if schema_name else table_name

    roles: dict[str, str] = {}
    col_lines_parts: list[str] = []
    for col in columns[:40]:
        role = _classify_column(col["name"], fk_columns, table_name)
        roles[col["name"].lower()] = role
        col_lines_parts.append(_format_column_line(col, role))

    col_lines = "\n".join(col_lines_parts)

    id_and_fk_cols = {name for name, role in roles.items() if role in ("ID", "FK")}

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

        if _uses_id_in_arithmetic(proposal.expression, id_and_fk_cols):
            logger.debug(
                "Dropping proposal %r: uses ID/FK/META column in arithmetic",
                proposal.name,
            )
            continue

        if _uses_year_month_divisor(proposal.expression):
            logger.debug(
                "Dropping proposal %r: divides by EXTRACT(YEAR/MONTH)",
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
