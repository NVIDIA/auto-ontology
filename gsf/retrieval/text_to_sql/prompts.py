# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

# Controls how many entity noun phrases are extracted for SQL generation.
SQL_GEN_MAX_ENTITIES: int = int(os.environ.get("SQL_GEN_MAX_ENTITIES", "5"))

main_system_prompt_template = (
    "Today's date is: {{ 'Year': {date.year}, 'Month': {date.month}, 'Day': {date.day}, "
    "'Time': '{date.hour:02}:{date.minute:02}:{date.second:02}' }}.\n\n"
    "{custom_prompts}"
)


create_sql_user_prompt = (
    "## Task\n"
    "Construct a SQL query that answers the user's question.\n"
    "Dialect: {dialect}.\n\n"
    "## Question\n"
    "{main_question}\n"
    "{observation_block}\n\n"
    "## Available Schema\n"
    "Use ONLY the tables and columns listed below. "
    "Do NOT invent tables, schemas, or columns.\n\n"
    "{tables}\n\n"
    "## Example SQL Queries\n"
    "{queries}\n\n"
    "## Conversation History\n"
    "{qa_from_conversations}\n\n"
    "{custom_analyses}"
    "## Rules\n\n"
    "**Correctness**\n"
    "- Every alias used in SELECT / WHERE / GROUP BY / ORDER BY / HAVING "
    "must be defined in FROM or JOIN. Never reference an undefined alias.\n"
    "- Verify each column exists in the table you reference it from. "
    "Do not confuse columns across tables.\n"
    "- GROUP BY must include all non-aggregated columns in SELECT.\n"
    "- ORDER BY must only reference aggregated aliases or columns "
    "present in SELECT/GROUP BY.\n\n"
    "**Joins**\n"
    "- Join only when necessary; choose join type (INNER / LEFT / RIGHT) "
    "based on the question's intent. Avoid fan-out from many-to-many joins.\n\n"
    "{join_paths}\n\n"
    "**Aggregation**\n"
    "- Never use FILTER (WHERE ...) on aggregates — it is not supported in all dialects. "
    "Use CASE WHEN inside aggregates instead: "
    "COUNT(CASE WHEN condition THEN 1 END) or SUM(CASE WHEN condition THEN 1 ELSE 0 END).\n"
    "- If business categories are specified, use CASE WHEN to classify.\n\n"
    "**Example Queries**\n"
    "- Review example queries for WHERE values that match the question's intent. "
    "If a value or filter condition is relevant to what is being asked, include it in your SQL.\n\n"
    "**Dialect & Syntax**\n"
    "- Never use :: casts, QUALIFY, DISTINCT ON, GROUP BY ALL, PIVOT, UNPIVOT, "
    "CROSS JOIN LATERAL, LATERAL JOIN, NATURAL JOIN, implicit comma joins, "
    "or any other vendor-specific or non-standard syntax.\n"
    "- Preserve the exact capitalization of values, names, and identifiers "
    "from the user's question.\n\n"
    "{dialect_rules}"
    "**Style**\n"
    "- SELECT only the columns explicitly asked; extra columns make the result "
    "wrong even when the rows are right. For superlative/ranking questions "
    "(most/least/top/highest/lowest/peak/best/worst), select ONLY the item named "
    "— the ranking key OR the aggregated value, never both — and never add the "
    "ORDER BY metric unless its value is asked. To identify an entity "
    "(who/which/what), return one identifying column (name if it exists, else id), "
    "not both.\n"
    "- If evidence maps an answer concept to specific columns, preserve that "
    "projection exactly; do not collapse, reshape, or replace those columns "
    "unless the question explicitly asks for a transformed value.\n"
    "- Time windows: apply a date/year filter ONLY when the question's data "
    "request names a period; 'last week/month/year' then means the most "
    "recent completed calendar period, not a rolling window.\n"
    "- Infer LIMIT from the question's intent: "
    "if a superlative (most/least/highest/lowest/best/worst/top/bottom) "
    "is paired with a number, add LIMIT with that number; "
    "if a superlative appears without a number, add LIMIT 1; "
    "if a specific count is requested without a superlative, "
    "add LIMIT with that number; "
    "otherwise do not add LIMIT.\n"
    "- Do NOT include comments in the SQL.\n"
    "- Do NOT use ellipsis as placeholder — output the complete SQL.\n"
)


# Functions the LLM reaches for (Postgres / Snowflake / BigQuery / PostGIS
# habits) that are absent from the stdlib SQLite build used at execution time.
# Math builtins (sin/cos/acos/radians/sqrt/pi/pow/ln/log/exp/mod/…) ARE
# available, so distance math can be written by hand.
_SQLITE_DIALECT_RULES = (
    "**SQLite-specific (STRICT — these will error at execution)**\n"
    "- No LEAST / GREATEST. Use scalar MIN(a, b, …) / MAX(a, b, …) instead.\n"
    "- No spatial / PostGIS functions (ST_Distance, ST_X, ST_Y, ST_DWithin, "
    "POINT, distance(), …). Compute great-circle distance by hand with the "
    "Haversine formula using sin/cos/acos/radians/sqrt (all available).\n"
    "- No statistical aggregates (STDDEV, STDDEV_POP, VARIANCE, VAR_POP, "
    "PERCENTILE_CONT, PERCENTILE_DISC, MEDIAN, CORR, REGR_*). Derive them with "
    "plain arithmetic (AVG, SUM, COUNT, window functions).\n"
    "- No STRING_AGG / ARRAY_AGG — use GROUP_CONCAT. No generate_series.\n"
    "- No EXTRACT(...) / DATE_TRUNC / DATE_PART / AGE / NOW() / INTERVAL "
    "literals. Use strftime(), date(), datetime() for all date/time work.\n"
    "- No :: casts and no ILIKE. Use CAST(x AS type); LIKE is case-insensitive "
    "for ASCII.\n"
    "- Coordinates and other composite columns are stored as TEXT, not JSON or "
    "arrays. Do NOT use json_extract on non-JSON text — inspect the value shape "
    "and parse with substr()/instr()/CAST as needed.\n"
    "- Prefer built-in aggregate/math functions and window functions only.\n\n"
)


# Snowflake folds unquoted identifiers to UPPERCASE. Datasets loaded from
# BigQuery/Google-public-data (e.g. Spider2 PATENTS) keep their original
# lowercase, case-sensitive column names, so an unquoted/upper reference raises
# "invalid identifier". Nested BigQuery RECORD/REPEATED fields land as VARIANT
# arrays that need LATERAL FLATTEN to unnest.
_SNOWFLAKE_DIALECT_RULES = (
    "**Snowflake-specific (STRICT — these OVERRIDE the generic syntax bans above)**\n"
    "- The general rule against `::` casts and LATERAL joins does NOT apply here: "
    "Snowflake REQUIRES `::type` casts and `LATERAL FLATTEN` to read VARIANT data.\n"
    "- Identifiers are CASE-SENSITIVE when quoted, and Snowflake folds unquoted "
    "names to UPPERCASE. The schema above lists the real stored names. Wrap every "
    "table and column identifier in double quotes using the EXACT case shown, e.g. "
    '`SELECT t."publication_number" FROM "PATENTS"."PUBLICATIONS" AS t`. '
    "Never reference a lowercase column unquoted — it will fail as 'invalid identifier'.\n"
    "- Aliases you define may stay unquoted; only real table/column names need the "
    "exact-case double quotes.\n"
    "- VARIANT / ARRAY / OBJECT columns hold nested (semi-structured) data. To read "
    "fields inside them, use LATERAL FLATTEN: "
    '`FROM "T", LATERAL FLATTEN(input => "T"."assignee_harmonized") f` then '
    'access `f.value:"name"::string`. Selecting a VARIANT column directly returns '
    "the whole JSON, not scalar fields.\n"
    "- Use `:` / `[...]` path syntax for OBJECT fields and `::type` casts on the "
    'extracted values (e.g. `f.value:"name"::string`).\n'
    "- Date columns loaded from BigQuery are often integer epoch/`YYYYMMDD` NUMBERs, "
    "not DATE types — check the sample values and cast/parse accordingly.\n\n"
)


# Dialects with a single flat namespace (no schemas): tables are referenced by
# bare name. Everything else (Postgres, Snowflake, HeavyDB) namespaces tables
# under a schema that MUST be kept in the identifier (``schema.table``).
_SCHEMALESS_DIALECTS = {"sqlite", "duckdb"}

_POSTGRES_DIALECT_RULES = (
    "**PostgreSQL-specific (STRICT — these will error at execution)**\n"
    "- WHERE and HAVING cannot reference SELECT aliases. Repeat the full expression or wrap in a subquery/CTE.\n"
    "- GROUP BY cannot reference SELECT aliases. Repeat the full expression "
    "(including CASE WHEN blocks) in GROUP BY, or wrap the query in a subquery/CTE.\n"
    "- Postgres folds unquoted identifiers to lowercase. If a table or column name shown in "
    "AVAILABLE TABLES / KNOWN COLUMN MAPPINGS contains any uppercase letter, wrap it in double "
    "quotes using the EXACT case shown — leaving it unquoted silently resolves to the wrong "
    "(lowercase) relation and errors as 'does not exist'. All-lowercase names need no quoting.\n\n"
)


# Shared output-field spec for SQL-generation prompts (candidates-based and
# table-based) — kept as a single source so the two call sites can't drift.
_SQL_GENERATION_OUTPUT_SPEC = """Output (fill fields in this exact order):
- thought: briefly explain your approach and state every assumption the
  request or schema doesn't uniquely determine. For each that applies,
  state the choice AND the reason ("X, because Y"): zero/missing values
  (included, excluded, or coerced to 0; how division guards a zero
  denominator), and ties (what breaks a tie in a ranking/superlative
  query).
- sql_code: the complete SQL, no comments or delimiters.
- response: 2-4 sentences for the end user, in plain English. Describe WHAT is
  being calculated, WHICH tables and columns are used, any FILTERS or time
  windows applied, and the GROUPING/ORDERING.
  Do NOT include SQL and code fences, raw identifiers like ``schema.table``,
  or meta-commentary about your reasoning. Refer to tables
  and columns by their human-readable names.
- All fields are required."""


def format_dialect_rules(dialect: str | None) -> str:
    """Return dialect-specific SQL rules for the ``dialect_rules`` prompt slot.

    SQLite lacks many functions the model habitually emits (spatial,
    LEAST/GREATEST, stats aggregates, EXTRACT/DATE_TRUNC). Snowflake needs
    identifier-quoting and VARIANT/FLATTEN guidance for case-sensitive
    lowercase columns (Spider2 PATENTS etc.). Returns '' for other dialects.
    """
    normalized = (dialect or "").strip().lower()
    if normalized == "sqlite":
        return _SQLITE_DIALECT_RULES
    if normalized == "snowflake":
        return _SNOWFLAKE_DIALECT_RULES
    if normalized in ("postgres", "postgresql"):
        return _POSTGRES_DIALECT_RULES
    return ""


def create_sql_from_candidates_prompt(
    *,
    dialect: str | None = None,
    target_db: str | None = None,
    has_evidence: bool = False,
) -> str:
    """System prompt for SQL generation from semantic retrieval candidates.

    Table naming is gated on the **dialect**, not on ``target_db``: schema-less
    dialects (SQLite/DuckDB) use bare table names, while schema
    dialects (Postgres/Snowflake) keep the ``schema.table`` qualifier. Scoping a
    query to one database (``target_db``) removes only the *database* prefix — the
    schema is still required to resolve the table, so it is never dropped here.
    """
    bare_table_names = (dialect or "").lower() in _SCHEMALESS_DIALECTS
    if bare_table_names:
        table_name_rule = (
            "- Use table names exactly as shown in AVAILABLE TABLES "
            "(unqualified — do NOT add a schema or database prefix).\n"
        )
        join_template = "    JOIN target_table ON source_table.source_column = target_table.target_column"
        example_sql = """SELECT c.country_name, SUM(s.sales_amount) AS total_sales
FROM sales AS s
JOIN customers AS c ON s.customer_id = c.customer_id
WHERE s.order_date BETWEEN '2024-01-01' AND '2024-03-31'
GROUP BY c.country_name
ORDER BY total_sales DESC;"""
    else:
        table_name_rule = (
            "- Use table names exactly as shown in AVAILABLE TABLES, INCLUDING the "
            "schema prefix (e.g., schema.table_name). Never drop the schema; do NOT "
            "add a database-name prefix.\n"
        )
        join_template = (
            "    JOIN target_schema.target_table ON source_schema.source_table.source_column\n"
            "         = target_schema.target_table.target_column"
        )
        example_sql = """SELECT c.country_name, SUM(s.sales_amount) AS total_sales
FROM PUBLIC.SALES AS s
JOIN PUBLIC.CUSTOMERS AS c ON s.customer_id = c.customer_id
WHERE s.order_date BETWEEN
  DATE_TRUNC('quarter', ADD_MONTHS(CURRENT_DATE, -3))
  AND LAST_DAY(ADD_MONTHS(DATE_TRUNC('quarter', CURRENT_DATE), -1))
GROUP BY c.country_name
ORDER BY total_sales DESC;"""

    evidence_block = (
        "## Evidence Priority\n"
        "The request includes evidence — treat it as authoritative "
        "ground truth. Evidence overrides semantic hints, examples, descriptions, and "
        "your own interpretation. Apply every evidence clause exactly: use named "
        "columns/tables, formulas, filters, synonyms, ranking rules, and LIKE patterns "
        "as specified. If evidence maps a requested answer to columns, SELECT those "
        "columns exactly. Do NOT substitute semantically similar columns or raw "
        "question literals when evidence gives an exact SQL mapping. "
        "Follow any explicit evidence formula verbatim: same operands and same "
        "numerator/denominator order even if the question implies the opposite, "
        "with no added * 100, - 1, ROUND, or extra columns.\n\n"
        if has_evidence
        else ""
    )

    return f"""You are an expert SQL query builder. You MUST always produce a SQL query.

{evidence_block}Key rules:
{table_name_rule}
- When SQL snippets are provided as reference, do NOT copy their aliases.
  Define your own aliases in FROM/JOIN and use only those.
- File contents (if present) are inputs only — use them as literals, filters,
  or CASE logic within the SQL.
- SEMANTIC HINT (if present) shows a likely starting table and suggested join
  paths derived from the semantic model. Treat it as a strong hint: prefer it
  when it fits, but if AVAILABLE TABLES provide a simpler or more direct answer,
  use them instead. Never force the semantic hint if it doesn't match the question.
- SUGGESTED JOIN PATHS show column-level join conditions. Use only the hops you
  actually need:
{join_template}
  Follow hops in order when the path spans more than one table.
- DOMAIN-SPECIFIC CUSTOM ANALYSES: if one closely matches the question, use or
  adapt its full SQL directly as your starting point — you may reuse it wholesale,
  trimming only what does not apply. Do NOT copy its aliases.
- SQL ATTRIBUTES: derived metrics or formulas with pre-defined SQL expressions.
  If one matches the question's intent, incorporate its expression or SQL pattern
  into your query. Treat them like reusable building blocks for calculations.
- Prefer the fewest joins that still correctly answer the question. If all
  required fields exist in a single table, use only that table. If a shorter
  join path covers the question equally well, choose it over a longer chain.
- When creating a JOIN, both sides of the ON condition must use columns with
  the same data type. Never join a text column to a numeric column or a date
  column to an integer column, or uuid column to a string column.
- Never match a human name/label against an id or foreign-key column (`*_id`,
  `link_to_*`). To filter by a name, join to the table holding the name columns
  (first_name/last_name/*_name) and filter there. Join each foreign key to the
  primary key it actually references (e.g. `expense.link_to_member` =
  `member.member_id`, never `event.event_id`).
- Use only standard JOIN types with explicit ON conditions: INNER JOIN, LEFT JOIN,
  RIGHT JOIN, FULL OUTER JOIN. Never use CROSS JOIN LATERAL, LATERAL JOIN,
  NATURAL JOIN, implicit comma joins, or any other non-standard join syntax.
- If the question filters by a single constant value on a column,
  do NOT include that column in SELECT — it adds no information since every row has the same value.

{_SQL_GENERATION_OUTPUT_SPEC}

Example:

thought:
Join sales and customers, filter last full quarter, aggregate by country.
"Total sales" means gross SUM(sales_amount), with no refund adjustment
since the question didn't ask for one.

sql_code:
{example_sql}

response:
This calculates total sales revenue per country for the most recently completed
calendar quarter. It combines the sales records with the customers list so each
sale is attributed to a country, sums the sales amounts within that quarter,
and then groups the results by country and orders them from highest to lowest
total sales.
"""


create_sql_general_prompt = f"""You are an expert SQL query builder.
You will receive a user question and a list of relevant tables.

If no tables are relevant, explain politely and suggest rephrasing.
Otherwise, construct an optimized SQL query to answer the question.

{_SQL_GENERATION_OUTPUT_SPEC}

Do NOT mention corrected errors.
Do NOT force a match if the tables are not relevant to the question."""


INTENT_VALIDATION_SYSTEM_PROMPT = """You are a SQL
validation expert. Your job is to check if a generated
SQL query has any CRITICAL issues that would prevent it
from answering the user's question.

Be LENIENT - only mark as invalid if there are serious
problems. Minor issues or alternative approaches are
acceptable.

Check for CRITICAL issues only:
1. **Seriously Wrong Joins**: Flag only joins that are
nonsensical or clearly break the question (e.g. joining
unrelated tables, inventing keys). Alternate but plausible
join paths that still answer the question are acceptable —
including a different entity for a filter dimension, a
different field/role for the same concept, a
shorter/longer path, or another valid FK chain. Do NOT
fail for those.
2. **Clearly Wrong Aggregations**: Are aggregations
completely incorrect? (e.g., COUNT when user explicitly
asks for SUM) (Minor variations are acceptable)

When DOMAIN-SPECIFIC CUSTOM ANALYSES are provided, treat
their SQL patterns as intentional user-defined domain
definitions. Fragments that look unusual, incomplete, or
nonstandard in isolation are still valid if they follow
those custom analyses — do NOT mark them as critical issues
solely for that reason.

When AUTHORITATIVE JOIN PATHS are provided, they come from
the verified semantic model. If the generated SQL uses a
join condition from those paths, keep it and do NOT flag
that join as invalid.

IMPORTANT: Be generous in your validation. If the SQL
could reasonably answer the question, mark it as valid.
Only fail validation for serious, critical errors that
would make the query unusable."""


# INTENT_VALIDATION_JOINS_VALIDATED_ELSEWHERE variant of the system prompt —
# used only when a separate deterministic check (e.g. db_probe.join_path_check)
# already validates join legality, so this LLM check can assume every join is
# real and focus on whether it reaches the right entity. See intent_validation.py.
INTENT_VALIDATION_SYSTEM_PROMPT_JOINS_VALIDATED_ELSEWHERE = """You are a SQL
validation expert. Your job is to check if a generated
SQL query has any CRITICAL issues that would prevent it
from answering the user's question.

Be LENIENT - only mark as invalid if there are serious
problems. Minor issues or alternative approaches are
acceptable.

Check for CRITICAL issues only:
1. **Semantically Wrong Joins**: Assume every join in the
query is real — do not question whether the relation exists.
Flag a join if either (a) it is self-evidently broken
regardless of any alternative — e.g. a tautological
condition (`a.x = a.x`), a table joined to itself, or
columns of clearly unrelated meaning being equated — or
(b) it reaches a different, wrong entity for the question
when you can name a specific, better-fitting real
relationship instead (e.g. a related-but-different table, or
the wrong field/role for the same concept). Do not flag a
join just because it looks unfamiliar — alternate but
plausible join paths (a different entity for a filter
dimension, a shorter/longer path, another valid FK chain)
are acceptable.
2. **Clearly Wrong Aggregations**: Are aggregations
completely incorrect? (e.g., COUNT when user explicitly
asks for SUM) (Minor variations are acceptable)

When DOMAIN-SPECIFIC CUSTOM ANALYSES are provided, treat
their SQL patterns as intentional user-defined domain
definitions. Fragments that look unusual, incomplete, or
nonstandard in isolation are still valid if they follow
those custom analyses — do NOT mark them as critical issues
solely for that reason.

When AUTHORITATIVE JOIN PATHS are provided, they come from
the verified semantic model. If the generated SQL uses a
join condition from those paths, keep it and do NOT flag
that join as invalid.

IMPORTANT: Be generous in your validation. If the SQL
could reasonably answer the question, mark it as valid.
Only fail validation for serious, critical errors that
would make the query unusable."""


def format_dual_question_block(
    original_question: str,
    sanitized_question: str,
    processing_question: str = "",
) -> str:
    """Format original and sanitized questions for SQL generation/validation.

    When ``processing_question`` is given (e.g. a follow-up resolved into a
    standalone question), it is rendered alongside the other two so the model can
    see what was actually asked, what it was resolved to, and what was retrieved on.
    """
    original = original_question.strip()
    normalized = sanitized_question.strip()
    processing = processing_question.strip()

    if original == normalized:
        return sanitized_question

    processing_block = ""
    if processing and processing not in {original, normalized}:
        processing_block = f"Standalone processing question:\n{processing}\n\n"
    return (
        f"Original user request:\n{original}\n\n"
        f"{processing_block}"
        f"Normalized retrieval question:\n{normalized}"
    )


def format_custom_analyses_section(custom_analyses: list[dict] | None) -> str:
    """Render custom analyses (name / description / SQL) for prompt injection.

    Matches the DOMAIN-SPECIFIC CUSTOM ANALYSES block used by SQL generation.
    Returns "" when there is nothing to inject.
    """
    if not custom_analyses:
        return ""
    ca_lines: list[str] = []
    for analysis in custom_analyses:
        line = f"- {analysis.get('name', '(unnamed)')}"
        desc = (analysis.get("description") or "").strip()
        if desc:
            line += f": {desc}"
        sql = (analysis.get("sql") or "").strip()
        if sql:
            line += f"\n  SQL: {sql}"
        ca_lines.append(line)
    if not ca_lines:
        return ""
    return (
        "DOMAIN-SPECIFIC CUSTOM ANALYSES (use their SQL patterns as guidance):\n"
        + "\n".join(ca_lines)
        + "\n\n"
    )


def create_empty_like_check_prompt(
    question_block: str,
    sql_code: str,
) -> str:
    return f"""You analyze a SQL query that executed successfully but returned zero rows.

The SQL contains LIKE or ILIKE predicates. Your job is to classify each LIKE/ILIKE
predicate as essential or non-essential.

Definitions:
- Essential: identifies the main subject of the question — the thing the user is
  searching for.
- Non-essential: constrains a feature, preference, descriptive attribute, or
  additional filter that is not the main subject.

Rules:
- List every LIKE/ILIKE predicate from the SQL exactly as it appears (column,
  operator, and pattern).
- Put predicates to remove in non_essential_like_predicates.
- Put predicates that must be preserved in essential_like_predicates.
- If uncertain whether a predicate is essential, treat it as essential.
- Do not suggest removing joins, numeric thresholds, or non-LIKE filters.

User question:
{question_block}

SQL:
```sql
{sql_code}
```
"""


def create_question_sanitization_prompt(question: str) -> str:
    return f"""You rewrite conversational user requests into concise, SQL-ready questions.

Rules:
- Remove personal background, narrative fluff, and filler.
- Preserve every factual constraint: numbers, product names, brands, categories, and qualifiers
  such as "similar", "natural ingredients", or "expensive is okay".
- Do NOT invent constraints that are not in the original text.
- Output one concise question or search intent, not a paragraph.

Examples:

Input: We're planning a road trip next summer and my whole family loves hiking.
I need a tent that can fit 4 people, and lighter is better since we'll carry it.
Output: Find a 4-person tent, prioritizing lighter weight.

Input: My old headphones broke. I mostly listen on the train so I'd really like
good noise cancelling, and I'd prefer to stay under $200.
Output: Find noise-cancelling headphones under $200.

Input: How many shipments were delivered last month?
Output: How many shipments were delivered last month?

Input: {question}
Output:"""


def create_prediction_classification_prompt(question: str) -> str:
    return f"""You are a router that decides whether a question requires a PREDICTION.

A PREDICTION question asks for a FUTURE, expected, or currently-unknown value that
must be forecast or estimated from patterns in the data — it cannot be answered by
simply querying rows that already exist.

Decision rule: default to NOT a prediction. Answer True ONLY if the question is
explicitly about the future or an unknown outcome — typically signalled by words
like "will", "predict", "forecast", "expected", "projected", "likely", "next
month/quarter/year", "going to", or "at risk".

A question about what ALREADY happened is NEVER a prediction, even when it names a
specific date, month, or year (past OR future) — counting, listing, or aggregating
existing rows is a plain data query. "How many X were created/added/sold in <year>"
asks to COUNT rows that already exist, so it is NOT a prediction.

Prediction (True):
- "How many orders will customer 42 place in the next 30 days?"
- "Predict which customers are likely to churn."
- "What is the expected revenue next quarter?"
- "Will this user upgrade their subscription?"

Not a prediction (False) — answerable from existing data:
- "How many GPUs created in 2025?"          (counts existing rows for a year)
- "How many orders were placed in 2025?"
- "How many orders did customer 42 place last month?"
- "List the top 10 customers by revenue."
- "What was total revenue last quarter?"

Question: {question}

Decide: is this a prediction request?"""


def create_pql_generation_prompt(question: str, schema_text: str) -> str:
    return f"""You translate a natural-language question into a single KumoRFM
Predictive Query Language (PQL) query.

PQL structure: PREDICT <target> FOR <entity> [WHERE <filters>]
- Target: an aggregation over related rows across a future window, or a column.
  Aggregations take (column_or_*, start_offset, end_offset, unit), e.g.
  SUM(orders.price, 0, 30, days), COUNT(orders.*, 0, 90, days).
- Entity: a table's primary key selecting the row(s) to predict for, e.g.
  users.user_id=42, or users.user_id (all rows).

Examples:
- PREDICT SUM(orders.price, 0, 30, days) FOR users.user_id=42
- PREDICT COUNT(orders.*, 0, 90, days) = 0 FOR users.user_id=42
- PREDICT users.age FOR users.user_id=42

Rules:
- Use ONLY the tables and columns listed below. Do not invent names.
- Reference columns as table.column exactly as named.
- Return exactly one valid PQL query.

## Available tables and columns
{schema_text}

## Question
{question}

Produce the PQL query."""


def create_intent_validation_prompt(
    original_question: str,
    processing_question: str,
    sanitized_question: str,
    sql_code: str,
    custom_analyses: str = "",
    join_paths: str = "",
    joins_validated_elsewhere: bool = False,
) -> str:
    question_block = format_dual_question_block(
        original_question, sanitized_question, processing_question
    )
    custom_analyses_block = f"\n{custom_analyses}" if custom_analyses.strip() else ""
    # joins_validated_elsewhere (INTENT_VALIDATION_JOINS_VALIDATED_ELSEWHERE,
    # off by default) opts into this branch's own, more permissive join
    # criterion instead of main's current one, and never shows AUTHORITATIVE
    # JOIN PATHS — appropriate only when a separate deterministic check (e.g.
    # db_probe.join_path_check) already covers join legality, so this LLM
    # check doesn't have to. See the flag's docstring in intent_validation.py.
    if joins_validated_elsewhere:
        join_paths_block = ""
        join_criterion = (
            "1. Every join in this query is already known to be real — do not question whether it exists. "
            "Flag it only if either (a) it is self-evidently broken regardless of any alternative (a "
            "tautological condition, a table joined to itself, columns of clearly unrelated meaning being "
            "equated), or (b) it clearly reaches the wrong entity for the question and you can name a "
            "specific, better-fitting real relationship instead. Do not flag a join just because it merely "
            "looks unfamiliar (different fields/roles for the same concept, e.g. customer vs supplier "
            "delivery city for a region filter, are OK)."
        )
        authoritative_note = ""
    else:
        join_paths_block = f"\n{join_paths}" if join_paths.strip() else ""
        join_criterion = (
            "1. Are any joins nonsensical or clearly broken for the question? Alternate but plausible "
            "join paths that could still answer it are OK — including different fields/roles for the same "
            "concept (e.g. customer vs supplier delivery city for a region filter). Do NOT fail for those."
        )
        authoritative_note = (
            "\nIf AUTHORITATIVE JOIN PATHS are listed above, do not flag a generated join that follows "
            "one of those verified paths."
        )
    return f"""User's Question:
{question_block}
{custom_analyses_block}
{join_paths_block}
Generated SQL Query:
```sql
{sql_code}
```

Check for CRITICAL issues ONLY (be lenient):
{join_criterion}
2. Are aggregations CLEARLY WRONG for the question? (e.g., COUNT when explicitly asking for SUM) (Variations are OK)

Only mark as invalid if there are SERIOUS problems. If the SQL could reasonably work, mark it as VALID.
If DOMAIN-SPECIFIC CUSTOM ANALYSES are listed above, treat their SQL as intentional domain \
definitions — do not flag the generated query as invalid merely for following those \
patterns.{authoritative_note}

Provide your analysis."""


def create_entity_extraction_prompt(question: str) -> str:
    return f"""You are a database schema analyst. Given a question, populate the field \
"required_entity_name" with 1–{SQL_GEN_MAX_ENTITIES} noun phrases that correspond to database tables, \
columns, or relationships.

Preserve the exact casing of terms as they appear in the question. Do not lowercase,
uppercase, or normalize them.

Guidelines for what to include in required_entity_name:
- Subject nouns and domain terms ("invoice", "customer", "shipment")
- Qualified entity phrases that combine a subject with its relevant action or attribute
  ("order shipment", "employee hire", "ticket resolution")
- Filter-item rule: when several words together describe a single item the user wants to
  filter or search for, keep them in one phrase. Do not split modifier, noun, and purpose
  of the same filter item into separate entries.
  Example: "waterproof hiking tent for family camping" → ["waterproof hiking tent for family camping"],
  not ["waterproof hiking tent", "family camping"].
  This rule applies only to one filterable item. Do not merge separate retrieval targets
  (e.g. a subject entity and a time dimension still get separate entries when appropriate).
- Keep names and descriptive text that identify something: brand names, product names,
  vendor names, categories, and other named constants (e.g. "Salomon Speedcross", "Grip Rx").
- For interrogative words (who/what/which/whose), resolve to the implied entity type
  AND, if the question contains a qualifying descriptor, include it twice: once alone
  and once combined with the resolved type.
  Example: "who are the active assignees" → ["assignee", "active assignee"]

Guidelines for what to exclude from required_entity_name:
- Bare action verbs ("submitted", "approved", "closed", "assigned")
- Numeric values: counts, amounts, prices, years, and other number literals
  (e.g. 1000, $150, 2023, Q2) — omit these from phrases; they are not entity names
- Date/time values when they are numeric or calendar literals, not named descriptions
- Aggregation indicators ("count", "total", "average", "sum", "min", "max")
  when standing alone, not part of a measurable phrase
- Status and filter adjectives when standing alone ("open", "active", "high-priority")

Date rule: When a question references a time-qualified event, collapse subject + action
+ granularity into one compact phrase ending with "date". Do NOT emit the verb, a
column-name guess, the date value, and the granularity as separate entries.
  If a granularity is mentioned (quarter, month, week, year, day), include it before "date".
  If no granularity is mentioned, end with just "date".
  Example: "invoices closed in Q2" → required_entity_name: ["invoice", "invoice closure quarter date"]
  Example: "orders placed last year" → required_entity_name: ["order", "order placement date"]

Examples:
  Q: "How many shipments were delivered last month?"
  → required_entity_name: ["shipment", "shipment delivery month date"]

  Q: "What is the average salary of engineers hired in 2023?"
  → required_entity_name: ["salary", "engineer", "engineer hire date"]

  Q: "Who are the reviewers assigned to pending tasks?"
  → required_entity_name: ["task", "reviewer", "assigned reviewer"]

  Q: "Find a waterproof hiking tent for family camping."
  → required_entity_name: ["waterproof hiking tent for family camping"]

  Q: "Recommend trail running shoes similar to Salomon Speedcross."
  → required_entity_name: ["trail running shoes similar to Salomon Speedcross"]

Question: {question}
"""


CUSTOM_ANALYSIS_RELEVANCE_FILTER_PROMPT = """You are a database domain expert.
Given a user's question and retrieved custom analyses, decide which analyses
are NOT relevant to answering the question.

Rules:
- Only remove an analysis if you are confident it is NOT needed.
- When in doubt, keep it — it is safer to include an extra analysis
  than to remove a necessary one.
- Consider both the analysis description AND its SQL when judging relevance.

User's question:
{question}

Retrieved custom analyses:
{analyses_summary}

Return the names of analyses to REMOVE. If unsure, return an empty list."""


TABLE_RELEVANCE_FILTER_PROMPT = """You are a database schema expert.
Given a user's question and a list of candidate tables, decide which tables
are actually needed to answer the question.

Rules:
- Only remove tables you are confident are NOT needed in the SQL query.
- If table A must be joined through tables B and C to reach table D, do NOT
  remove any table in the join chain (A, B, C, or D). The join paths below
  show real table connections. Keep the full bridges between tables that
  you deem relevant.
- If a selected custom analysis references a table in its SQL, do NOT
  remove that table.
- When in doubt, do NOT remove — it is safer to include an extra table
  than to remove a necessary one.

{domain_rules}{custom_analyses}{join_paths}{enriched_question}User's question:
{question}

Candidate tables:
{tables_summary}

Provide brief reasoning (1-2 sentences) then return the names of tables that can be safely REMOVED.
Only remove a table if you are confident it is not needed. When in doubt, do NOT remove."""


def create_follow_up_resolution_prompt(
    *, question: str, conversation_history: str
) -> str:
    """Build the prompt that turns a contextual follow-up into a standalone query."""

    return f"""
You resolve conversational follow-up questions for a text-to-SQL agent.

Use only the completed conversation turns below. Never invent a table, filter,
entity, metric, date range, or other constraint that is not present in the
current question or the history.

Return:
- is_follow_up=true only when the current question depends on prior context,
  such as pronouns, omitted subjects, "same", "also", "instead", "what about",
  or a modification to the preceding request.
- standalone_question as a complete, natural-language question containing all
  context needed to answer the current request.
- For an independent question, set is_follow_up=false and copy the current
  question unchanged into standalone_question.

Do not answer the question and do not generate SQL.

COMPLETED CONVERSATION HISTORY:
{conversation_history}

CURRENT QUESTION:
{question}
""".strip()
