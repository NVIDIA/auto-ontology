# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
    "{search_values}"
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
    "**Style**\n"
    "- When the question refers to an entity or a list of entities "
    "(e.g. 'which item', 'list the products'), select only that entity's "
    "ID column. Do not select its name column, and never select both.\n"
    "- Time windows: 'last week/month/year' means the most recent "
    "completed calendar period, not a rolling window.\n"
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


def create_sql_from_candidates_prompt() -> str:
    """System prompt for SQL generation from semantic retrieval candidates."""
    return """You are an expert SQL query builder. You MUST always produce a SQL query.

Key rules:
- Use fully qualified table names exactly as provided (e.g., schema.table_name).
  Never drop the schema/database prefix.
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
    JOIN target_schema.target_table ON source_schema.source_table.source_column
         = target_schema.target_table.target_column
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
- Use only standard JOIN types with explicit ON conditions: INNER JOIN, LEFT JOIN,
  RIGHT JOIN, FULL OUTER JOIN. Never use CROSS JOIN LATERAL, LATERAL JOIN,
  NATURAL JOIN, implicit comma joins, or any other non-standard join syntax.
- If the question filters by a single constant value on a column,
  do NOT include that column in SELECT — it adds no information since every row has the same value.

Output (fill fields in this exact order):
- thought: 1-2 sentence internal reasoning — your approach and key decisions.
- sql_code: the complete SQL, no comments or delimiters.
- response: 2-4 sentences for the end user, in plain English. Describe WHAT is
  being calculated, WHICH tables and columns are used, any FILTERS or time
  windows applied, and the GROUPING/ORDERING.
  Do NOT include SQL and code fences, raw identifiers like ``schema.table``,
  or meta-commentary about your reasoning. Refer to tables
  and columns by their human-readable names.
- All fields are required.

Example:

thought:
Join sales and customers, filter last full quarter, aggregate by country.

sql_code:
SELECT c.country_name, SUM(s.sales_amount) AS total_sales
FROM PUBLIC.SALES AS s
JOIN PUBLIC.CUSTOMERS AS c ON s.customer_id = c.customer_id
WHERE s.order_date BETWEEN
  DATE_TRUNC('quarter', ADD_MONTHS(CURRENT_DATE, -3))
  AND LAST_DAY(ADD_MONTHS(DATE_TRUNC('quarter', CURRENT_DATE), -1))
GROUP BY c.country_name
ORDER BY total_sales DESC;

response:
This calculates total sales revenue per country for the most recently completed
calendar quarter. It combines the sales records with the customers list so each
sale is attributed to a country, sums the sales amounts within that quarter,
and then groups the results by country and orders them from highest to lowest
total sales.
"""


create_sql_general_prompt = """You are an expert SQL query builder.
You will receive a user question and a list of relevant tables.

If no tables are relevant, explain politely and suggest rephrasing.
Otherwise, construct an optimized SQL query to answer the question.

Output (fill fields in this exact order):
- thought: 1-2 sentence internal reasoning — your approach and key decisions.
- sql_code: the complete SQL, no comments or delimiters.
- response: 2-4 sentences for the end user, in plain English. Describe WHAT is
  being calculated, WHICH tables and columns are used, any FILTERS or time
  windows applied, and the GROUPING/ORDERING.
  Do NOT include SQL and code fences, raw identifiers like ``schema.table``,
  or meta-commentary about your reasoning. Refer to tables
  and columns by their human-readable names.
- All fields are required.

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
1. **Seriously Wrong Joins**: Are there joins that would
produce completely wrong results? (Minor join variations
are acceptable)
2. **Clearly Wrong Aggregations**: Are aggregations
completely incorrect? (e.g., COUNT when user explicitly
asks for SUM) (Minor variations are acceptable)

IMPORTANT: Be generous in your validation. If the SQL
could reasonably answer the question, mark it as valid.
Only fail validation for serious, critical errors that
would make the query unusable."""


def format_dual_question_block(original_question: str, sanitized_question: str) -> str:
    """Format original and sanitized questions for SQL generation/validation."""
    if original_question.strip() == sanitized_question.strip():
        return sanitized_question
    return (
        f"Original user request:\n{original_question}\n\n"
        f"Sanitized SQL intent:\n{sanitized_question}"
    )


def create_question_understanding_prompt(question: str) -> str:
    return f"""You are a database schema analyst. Given a conversational user request, \
produce THREE things in one structured response:

1. normalized_question — a concise, SQL-ready rewrite of the request.
2. metadata — schema-level concepts (columns, tables, relationships).
3. values — concrete items/values the user is searching or filtering for.

====================================================================
PART 1 — normalized_question
====================================================================
Rewrite the user's request into one concise, SQL-ready question.
Rules:
- Remove personal background, narrative fluff, and filler.
- Preserve EVERY factual constraint: numbers, thresholds, prices, counts,
  product names, brands, categories, and qualifiers such as "similar",
  "natural ingredients", or "expensive is okay". Do NOT drop constraints.
- Do NOT invent constraints that are not in the original text.
- If the input is already a direct question, return it unchanged.
- Output one concise question or search intent, not a paragraph.

Examples:
  Input: We're planning a road trip next summer and my whole family loves hiking.
  I need a tent that can fit 4 people, and lighter is better since we'll carry it.
  normalized_question: Find a 4-person tent, prioritizing lighter weight.

  Input: How many shipments were delivered last month?
  normalized_question: How many shipments were delivered last month?

TWO GLOBAL RULES (apply to both metadata and values):
- NO DUPLICATION: a concept appears in AT MOST ONE list. Never repeat the same
  term — or any component word of a value item — across both lists.
- NEVER SPLIT: one real-world item is exactly ONE entry. Never break an item into
  its modifier, noun, and purpose, and never emit the item's component nouns as
  separate entries anywhere.

Both "metadata" and "values" are arrays of plain strings.

====================================================================
PART 2 — metadata (schema-level concepts)
====================================================================
Populate "metadata" with schema-level dimensions the user EXPLICITLY filters,
groups, or aggregates on that are DISTINCT from the item being searched. Each
entry is a single string (preserve the exact casing from the question).

Include:
- Explicit filter/aggregate dimensions: an explicit price/quantity/category or
  date constraint ("under $100" -> "price"; "in 2023" -> a date concept).
- Time dimensions collapsed into one compact phrase ending with "date". If a
  granularity is mentioned (quarter/month/week/year/day), include it before
  "date". Example: "invoices closed in Q2" -> "invoice closure quarter date".
- For pure counting/aggregation questions with NO product/item search, the
  subject nouns go here (e.g. "shipment", "salary", "engineer") and values is [].

Exclude from metadata:
- The searched item or ANY of its component nouns. If the value item is a
  "waterproof hiking tent for two", then "tent", "hiking", and "waterproof" must
  NOT appear here.
- Bare action verbs, numeric/date literals, and standalone aggregation words.
- Soft preferences that are not real filters ("a bit expensive", "if possible").

====================================================================
PART 3 — values (concrete search/filter targets)
====================================================================
Populate "values" with the concrete item(s) the user wants to find (the things
that become LIKE/ILIKE/WHERE filters). Usually there is exactly ONE such item.
Each entry is a single, SHORT, product-style phrase — the concise form only:
- Drop connective/purpose words ("to hold", "that can", "for", "used for") and
  incidental descriptors (quantities, "empty", condition or preference words),
  then rephrase into a canonical noun phrase while KEEPING the qualifier that
  defines what the item IS. Never reduce it to a bare generic noun (do not shrink
  "waterproof hiking tent" to "tent").
  Example: "stand to hold potted plants" -> "plant stand".
- Preserve the exact casing of proper nouns from the question.
- Keep brand names, product names, vendor names, and named constants.
- Keep the item whole: do NOT create separate entries for its parts, and do NOT
  also list those parts under metadata.

Examples:
  Q: "Recommend trail running shoes similar to Salomon Speedcross under $150."
  -> metadata: ["price"]
     values:   ["trail running shoes"]

  Q: "What is the average salary of engineers hired in 2023?"
  -> metadata: ["salary", "engineer", "engineer hire date"]
     values:   []

Question: {question}
"""


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
    sanitized_question: str,
    entities_text: str,
    sql_code: str,
) -> str:
    question_block = format_dual_question_block(original_question, sanitized_question)
    return f"""User's Question:
{question_block}

Generated SQL Query:
```sql
{sql_code}
```

Check for CRITICAL issues ONLY (be lenient):
1. Are there any joins that would produce COMPLETELY WRONG results? (Alternative join approaches are OK)
2. Are aggregations CLEARLY WRONG for the question? (e.g., COUNT when explicitly asking for SUM) (Variations are OK)

Only mark as invalid if there are SERIOUS problems. If the SQL could reasonably work, mark it as VALID.

Provide your analysis."""


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
- If table A must be joined through table B to reach table C, do NOT
  remove any table in the join chain (A, B, or C).
- If a selected custom analysis references a table in its SQL, do NOT
  remove that table.
- When in doubt, do NOT remove — it is safer to include an extra table
  than to remove a necessary one.

{domain_rules}{custom_analyses}User's question:
{question}

Candidate tables:
{tables_summary}

Provide brief reasoning (1-2 sentences) then return the names of tables that can be safely REMOVED.
Only remove a table if you are confident it is not needed. When in doubt, do NOT remove."""
