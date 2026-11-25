# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LLM extraction of derived business metrics (SqlAttributes) from table columns.

Gated by ``ENABLE_TABLE_SQL_ATTRIBUTES`` (see
:func:`auto_ontology.infra.feature_flags.is_table_sql_attribute_extraction_enabled`).
"""

from __future__ import annotations

import json
import logging
import numbers
import re
from typing import Any

import sqlglot
from langchain_core.messages import HumanMessage, SystemMessage
from sqlglot import exp

from auto_ontology.semantic.models import (
    SqlAttributeExtractionResult,
    SqlAttributeProposal,
)
from auto_ontology.utils.llm_invoke import (
    get_llm_client,
    invoke_with_structured_output,
)

logger = logging.getLogger(__name__)

_MAX_PROMPT_COLUMNS = 40
_MAX_PROMPT_SAMPLES = 5


class SqlAttributeExtractionFailed(RuntimeError):
    """The LLM call produced no result, as distinct from proposing zero metrics."""


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
9. Metric names must be user-friendly with spaces and MUST include the Term name \
to be specific.
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


def _sample_list(col: dict[str, Any]) -> list[Any]:
    """A column's sample values as a list, whether stored natively or as JSON."""
    raw = col.get("sample_values")
    if not raw:
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return []
    return list(raw) if isinstance(raw, (list, tuple)) else []


def _format_column_line(col: dict[str, Any]) -> str:
    """Format a single column for the LLM prompt, including sample values if present."""
    dtype = col.get("data_type") or "unknown"
    desc = col.get("description")
    line = f"  - {col['name']} ({dtype})"
    if desc:
        line += f" — {desc}"

    samples = _sample_list(col)
    if samples:
        rendered = ", ".join(repr(v) for v in samples[:_MAX_PROMPT_SAMPLES])
        line += f" [sample values: {rendered}]"

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

# A SELECT of one bare column (no operators) with no WHERE clause:
# e.g. SELECT schema.table.col AS "Label" FROM ...
_SINGLE_COLUMN_ALIAS = re.compile(
    r"^\s*SELECT\s+(?:\w+\.)*\w+\s+AS\b(?!.*\bWHERE\b)",
    re.IGNORECASE | re.DOTALL,
)

_WHERE = re.compile(r"\bWHERE\b", re.IGNORECASE)


def _is_number(value: Any) -> bool:
    return isinstance(value, numbers.Number) and not isinstance(value, bool)


def _literal_in_samples(literal: exp.Literal, samples: list[Any]) -> bool:
    """Exact match: strings by case-sensitive text, numbers by numeric value."""
    if literal.is_string:
        return any(isinstance(s, str) and s == literal.this for s in samples)
    try:
        value = float(literal.this)
    except ValueError:
        return False
    return any(_is_number(s) and float(s) == value for s in samples)


def _compared_column(
    predicate: exp.Expression,
) -> tuple[exp.Column, list[exp.Literal]] | None:
    """The bare column and the literals an ``=``, ``<>`` or ``IN`` compares it with."""
    if isinstance(predicate, exp.In):
        column, others = predicate.this, predicate.expressions
    elif isinstance(predicate.this, exp.Column):
        column, others = predicate.this, [predicate.expression]
    else:
        column, others = predicate.expression, [predicate.this]
    if not isinstance(column, exp.Column):
        return None
    return column, [other for other in others if isinstance(other, exp.Literal)]


def _where_literals_valid(expression: str, columns: list[dict[str, Any]]) -> bool:
    """Every WHERE literal compared with a column must be one of *its* samples.

    String literals that cannot be tied to a known column by ``=``, ``<>`` or
    ``IN`` fail too: the prompt only allows filtering on sampled values.
    """
    try:
        where = sqlglot.parse_one(expression).args.get("where")
    except sqlglot.errors.ParseError:
        return not _WHERE.search(expression)
    if where is None:
        return True

    samples_by_column = {col["name"].lower(): _sample_list(col) for col in columns}
    checked: set[int] = set()
    for predicate in where.find_all(exp.EQ, exp.NEQ, exp.In):
        pair = _compared_column(predicate)
        if pair is None:
            continue
        column, literals = pair
        samples = samples_by_column.get(column.name.lower())
        for literal in literals:
            if not samples or not _literal_in_samples(literal, samples):
                return False
            checked.add(id(literal))

    return all(
        id(lit) in checked for lit in where.find_all(exp.Literal) if lit.is_string
    )


def _uses_length_arithmetic(expression: str) -> bool:
    return bool(_LENGTH_ARITH.search(expression))


def _uses_year_month_divisor(expression: str) -> bool:
    return bool(_DIVIDE_BY_YEAR_MONTH.search(expression))


def _uses_validfrom_to(expression: str) -> bool:
    return bool(_VALIDFROM_TO.search(expression))


def _dedup_key(expression: str) -> str:
    """Formatting- and case-insensitive key that keeps literals and quoted names."""
    try:
        return sqlglot.parse_one(expression).sql(normalize=True)
    except sqlglot.errors.ParseError:
        return " ".join(expression.split())


def _rejection_reason(expression: str, columns: list[dict[str, Any]]) -> str | None:
    """Why a proposal must be dropped, or None when it passes every filter."""
    if not _where_literals_valid(expression, columns):
        return "WHERE literal not in sample values"
    if _SINGLE_COLUMN_ALIAS.match(expression):
        return "single-column alias with no computation"
    if _uses_length_arithmetic(expression):
        return "uses LENGTH() in arithmetic"
    if _uses_year_month_divisor(expression):
        return "divides by year/month value"
    if _uses_validfrom_to(expression):
        return "uses validfrom/validto versioning columns"
    return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def extract_sql_attributes(
    table: dict[str, Any],
    columns: list[dict[str, Any]],
    schema_name: str | None,
    term: dict[str, Any],
) -> list[SqlAttributeProposal]:
    """Ask the LLM to propose derived metrics, filter them, return survivors.

    *columns* must already exclude PII-tagged columns: their names,
    descriptions, and sample values all go into the prompt.

    Raises :class:`SqlAttributeExtractionFailed` when the LLM returns nothing,
    so callers can tell a failed call from an empty proposal list.
    """
    if len(columns) < 2:
        return []

    table_name = table["name"]
    qualified_table = f"{schema_name}.{table_name}" if schema_name else table_name

    col_lines = "\n".join(
        _format_column_line(col) for col in columns[:_MAX_PROMPT_COLUMNS]
    )

    prompt = (
        f"Table: {qualified_table}\n"
        f"Business entity (Term): {term.get('name', '')}\n"
        f"Term description: {term.get('description', '')}\n"
        f"Columns (semantically mapped to this term):\n{col_lines}\n\n"
        f"Propose derived metrics using schema={schema_name or 'public'} "
        f"and table={table_name} for qualified references."
    )

    result = invoke_with_structured_output(
        get_llm_client(temperature=0.0),
        [SystemMessage(content=_SYSTEM), HumanMessage(content=prompt)],
        SqlAttributeExtractionResult,
    )
    if result is None:
        raise SqlAttributeExtractionFailed(
            f"LLM returned no result for table {qualified_table}, "
            f"term {term.get('name', '')!r}"
        )

    accepted: list[SqlAttributeProposal] = []
    seen_expressions: set[str] = set()
    for proposal in result.metrics:
        normalized = _dedup_key(proposal.expression)
        if normalized in seen_expressions:
            logger.debug("Dropping duplicate proposal %r", proposal.name)
            continue
        seen_expressions.add(normalized)

        reason = _rejection_reason(proposal.expression, columns)
        if reason is not None:
            logger.debug("Dropping proposal %r: %s", proposal.name, reason)
            continue

        accepted.append(proposal)

    return accepted
