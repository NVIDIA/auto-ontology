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
    "- Prefer name columns over ID columns when both are available.\n"
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


def create_sql_from_candidates_prompt(
    *,
    dialect: str | None = None,
    target_db: str | None = None,
) -> str:
    """System prompt for SQL generation from semantic retrieval candidates."""
    bare_table_names = target_db is not None or (dialect or "").lower() == "sqlite"
    if bare_table_names:
        table_name_rule = (
            "- Use table names exactly as shown in AVAILABLE TABLES "
            "(no schema or database prefix).\n"
        )
        example_sql = """SELECT c.country_name, SUM(s.sales_amount) AS total_sales
FROM sales AS s
JOIN customers AS c ON s.customer_id = c.customer_id
WHERE s.order_date BETWEEN '2024-01-01' AND '2024-03-31'
GROUP BY c.country_name
ORDER BY total_sales DESC;"""
    else:
        table_name_rule = (
            "- Use fully qualified table names exactly as provided "
            "(e.g., schema.table_name).\n"
            "  Never drop the schema/database prefix.\n"
        )
        example_sql = """SELECT c.country_name, SUM(s.sales_amount) AS total_sales
FROM PUBLIC.SALES AS s
JOIN PUBLIC.CUSTOMERS AS c ON s.customer_id = c.customer_id
WHERE s.order_date BETWEEN
  DATE_TRUNC('quarter', ADD_MONTHS(CURRENT_DATE, -3))
  AND LAST_DAY(ADD_MONTHS(DATE_TRUNC('quarter', CURRENT_DATE), -1))
GROUP BY c.country_name
ORDER BY total_sales DESC;"""

    return f"""You are an expert SQL query builder. You MUST always produce a SQL query.

Key rules:
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
    JOIN target_table ON source_table.source_column = target_table.target_column
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
{example_sql}

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
- If the input is already a direct question, return it unchanged.
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


def create_entity_extraction_prompt(question: str) -> str:
    return f"""You are a database schema analyst. Given a question, populate the field \
"required_entity_name" with 1–5 noun phrases that correspond to database tables, \
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
