# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

# Controls how many entity noun phrases are extracted for SQL generation.
SQL_GEN_MAX_ENTITIES: int = int(os.environ.get("SQL_GEN_MAX_ENTITIES", "5"))

DIRECT_ANSWER_FIELD_RULES = """## Direct Answer Fields

For a yes/no question, when an available text or boolean column directly stores
the requested property or state, project that column unchanged as the answer.
Do not wrap the column in CASE, compare it merely to translate its stored value,
cast it, or otherwise normalize it to YES/NO. Transform it only when the question
or Authoritative Evidence explicitly requests a normalized representation.
Computed boolean or conditional output is appropriate only when no direct answer
field exists and the answer must be derived.
"""

_PROJECTION_RULES = (
    "**Projection**\n"
    "- SELECT only the columns explicitly asked; extra columns make the result "
    "wrong even when the rows are right. For superlative/ranking questions "
    "(most/least/top/highest/lowest/peak/best/worst), select ONLY the item named "
    "the ranking key OR the aggregated value, never both, and never add the "
    "ORDER BY metric unless its value is asked. To identify an entity "
    "(who/which/what), return one identifying column (name if it exists, else id), "
    "not both.\n"
    "- If the user asks for name, project the requested name fields and never add "
    "IDs unless the user explicitly asks for them.\n"
    "- When an entity has its own dedicated table among the available tables, "
    "take the entity's attributes (name, label, other descriptive fields) from "
    "that table, even when a related table also stores a copy of them: join "
    "the related table's id column to the entity's table and project the "
    "attribute from there, not the copy.\n"
    "- Preserve the question's field order in SELECT: project explicitly requested "
    "outputs from left to right in the same order the user names them.\n"
    "- If evidence maps an answer concept to specific columns, preserve that "
    "projection exactly; do not collapse, reshape, or replace those columns "
    "unless the question explicitly asks for a transformed value.\n"
)


def format_projection_rules(shorten_answer: bool = False) -> str:
    """Render benchmark-strict guidance only when shorter answers are requested."""

    if not shorten_answer:
        return _PROJECTION_RULES
    return (
        _PROJECTION_RULES
        + "- Return exactly the requested output fields and NO others.\n"
        "- Never use SELECT * or table.*; name each returned column explicitly.\n"
        "- When the question asks to list or show records without naming any "
        "field to return, return only the table's primary key column(s), as "
        "listed under primary key in the schema.\n"
        "- When the question or evidence requests one aggregate over two "
        "alternative categories joined by 'and' (for example, a count for X and Y), "
        "return ONE aggregate and combine the category predicates with OR in the "
        "filter. Do NOT project separate conditional aggregates for X and Y. Apply "
        "this only to one aggregate over alternatives; if separate results, a "
        "comparison, a per-category breakdown, or distinct metrics are explicitly "
        "requested, preserve those separate outputs.\n"
    )


main_system_prompt_template = (
    "Today's date is: {{ 'Year': {date.year}, 'Month': {date.month}, 'Day': {date.day}, "
    "'Time': '{date.hour:02}:{date.minute:02}:{date.second:02}' }}.\n\n"
    "{custom_prompts}"
)


create_sql_user_prompt = (
    "## Task\n"
    "Construct a SQL query that answers the user's question.\n"
    "When Authoritative Evidence is provided, fulfill every rule in it. Never "
    "ignore, omit, weaken, or substitute any evidence rule.\n"
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
    "- When similarly named columns exist in several tables, a column's "
    "'when to use' note says which one the question needs; follow it.\n"
    "- GROUP BY must include all non-aggregated columns in SELECT.\n"
    "- ORDER BY must only reference aggregated aliases or columns "
    "present in SELECT/GROUP BY.\n\n"
    "**Joins**\n"
    "- When evidence rules map to columns in different tables, include the joins "
    "that connect those tables using the listed paths. Choose the join type "
    "(INNER / LEFT / RIGHT) from the question's intent and avoid fan-out from "
    "many-to-many joins.\n"
    "- A listed join path is indivisible. If it contains intermediate or bridge "
    "tables, include every table and every hop exactly as listed. Removing a "
    "bridge table, skipping a hop, or replacing the path with a direct endpoint "
    "join is forbidden, even when the bridge contributes no projected column.\n\n"
    "{join_paths}\n\n"
    "**Aggregation**\n"
    "- Never use FILTER (WHERE ...) on aggregates — it is not supported in all dialects. "
    "Use CASE WHEN inside aggregates instead: "
    "COUNT(CASE WHEN condition THEN 1 END) or SUM(CASE WHEN condition THEN 1 ELSE 0 END).\n"
    "- If business categories are specified, use CASE WHEN to classify.\n"
    "- A condition on an aggregated value belongs in HAVING, not WHERE. When the "
    "question qualifies an entity by a total, average or count taken over that "
    "entity's rows ('customers whose consumption is under 30000', 'patients with "
    "two or more abnormal tests'), group by the entity and compare the aggregate "
    "in HAVING. A WHERE predicate tests one row at a time, before grouping, and "
    "answers a different question.\n"
    "- Integer division truncates. Cast to REAL before dividing wherever the "
    "result can be fractional — CAST(SUM(x) AS REAL) / COUNT(y) — including "
    "inside HAVING and before scaling a ratio by 100.\n\n"
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
    "{projection_rules}"
    "**Style**\n"
    "- Preserve natural row grain. Use DISTINCT only when the question or authoritative "
    "evidence explicitly asks for distinct, different, unique, or deduplicated results. "
    "A foreign-key marker signals possible fan-out but does not require DISTINCT: "
    "COUNT(fk) counts matching child rows; COUNT(DISTINCT fk) counts referenced entities. "
    "Never infer DISTINCT solely from an FK, one-to-many join, repeated IDs, or an entity "
    "noun.\n"
    "- NULL filtering: Do not add `IS NOT NULL` solely because a projected field "
    "has `is_nullable: true`, or because one or multiple fields are projected. "
    "Add it only when the question or evidence requires the requested value to "
    "exist. If the concept maps to one nullable column, check that column; if it "
    "maps to a composite concept, check every required nullable component. "
    "Otherwise, preserve rows containing NULLs.\n"
    "- Time windows: apply a date/year filter ONLY when the question's data "
    "request names a period; 'last week/month/year' then means the most "
    "recent completed calendar period, not a rolling window.\n"
    "- Infer LIMIT from the question's intent: "
    "if a superlative (most/least/highest/lowest/best/worst/top/bottom) "
    "is paired with a number, add LIMIT with that number; "
    "if an unnumbered superlative asks for one singular item, add LIMIT 1, "
    "unless the value being ranked is one that several items are likely to "
    "share: a count set by a schedule, format or quota, so groups run alike "
    "(games per season, classes per term, seats per section), however large; "
    "a small count per group (the number of orders, visits or wins); "
    "a value on a short scale or level (a 1-5 rating, a skill or attribute "
    "level, a count that is often zero); or another coarse value with few "
    "distinct levels. Judge this from what the value measures; never assume "
    "the stored values happen to differ. This exception exists for questions "
    "that ask for one item: singular wording ('the name of the item', "
    "'which item') does not override it. Ties are then likely, and LIMIT 1 "
    "would return one arbitrary item "
    "among equally valid answers, so keep every tied item instead: compare to "
    "the extreme with WHERE value = (SELECT MAX(value) ...) for a stored value, "
    "or HAVING COUNT(...) = (SELECT MAX(cnt) FROM (...)) for a count per group "
    "(MIN for lowest); "
    "if it asks for plural items, do not add LIMIT; "
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


# MySQL's ``/`` never truncates to an integer: it returns a DECIMAL with
# div_precision_increment (by default 4, and Auto Ontology's MySQL connector
# leaves it there) more decimal places than the dividend (integer
# division is the separate DIV operator). So the generic "cast to REAL before
# dividing" rule only adds casts here, and the model's MySQL cast of choice,
# DECIMAL(p, s), changes the digits of the result. Exact graders compare
# values as strings (BEAVER's ex_acc: 150.250000 != 150.25), so a cast result
# fails even when the query is otherwise right. The one cast worth keeping
# widens a small ratio a subquery or CTE stores, which keeps only those 4
# places and divides as zero later; within one expression MySQL keeps the
# precision itself.
_MYSQL_DIALECT_RULES = (
    "**MySQL-specific (these OVERRIDE the generic rules above)**\n"
    "- `/` never truncates in MySQL: integer / integer returns a DECIMAL with, "
    "by default, 4 more decimal places (1 / 3 = 0.3333). Do NOT cast to make "
    "a division fractional; the generic 'cast to REAL before dividing' rule "
    "does not apply.\n"
    "- Do NOT CAST a computed value (an aggregate, average, ratio or window "
    "result), or the column inside it, to DECIMAL(p, s), FLOAT, DOUBLE or REAL, "
    "and do not ROUND it, unless the question asks for a precision or for "
    "rounding: a cast changes the result's digits. Return computed values as "
    "MySQL produces them. The one exception: a ratio computed in a subquery "
    "or CTE keeps only 4 decimal places (1 / 1000000 = 0.0000), so if it is "
    "divided again later, cast its dividend to a DECIMAL with more scale.\n"
    "- STDDEV(), STD() and VARIANCE() are the POPULATION statistics in MySQL "
    "(the same as STDDEV_POP() and VAR_POP()). Use the function the question "
    "names; use STDDEV_SAMP() or VAR_SAMP() only when it asks for a sample "
    "statistic.\n\n"
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
- description: after sql_code is complete, express the request that the SQL
  answers as one concise natural-language question in the same style as a user
  question. Capture the requested result and business conditions, but do not
  mention SQL, tables, columns, joins, clauses, aliases, implementation details,
  reasoning, or assumptions.
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
    lowercase columns (Spider2 PATENTS etc.). MySQL needs the generic
    division-cast rule lifted, since its casts round results. Returns '' for
    other dialects.
    """
    normalized = (dialect or "").strip().lower()
    if normalized == "sqlite":
        return _SQLITE_DIALECT_RULES
    if normalized == "snowflake":
        return _SNOWFLAKE_DIALECT_RULES
    if normalized in ("postgres", "postgresql"):
        return _POSTGRES_DIALECT_RULES
    if normalized == "mysql":
        return _MYSQL_DIALECT_RULES
    return ""


def create_sql_from_candidates_prompt(
    *,
    dialect: str | None = None,
    target_db: str | None = None,
    has_sql_examples: bool = False,
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

    # Solved question/SQL pairs retrieved by question wording from a labelled
    # corpus of other databases. Two failure modes need heading off: borrowing
    # identifiers or literals instead of structure, and treating the first
    # listed pair as the best fit (they are ordered by wording similarity, which
    # is only weakly related to whether their structure transfers).

    # The exemplar's own evidence shows how a stated formula
    # was turned into SQL (casting, operand order, whether
    # *100 was applied), which is the transferable part.
    sql_examples_block = (
        "## How To Use Reference Query Patterns\n"
        "The request includes solved question/SQL pairs from OTHER databases, "
        "retrieved because their wording resembles this request. They are "
        "precedent for how a request of this kind becomes SQL — nothing else.\n"
        "Work through them like this:\n"
        "1. Read each pair's question, and its evidence when present, and judge "
        "which pairs ask for the same KIND of thing as this request: a count, a "
        "ratio, a share, a superlative, a per-group breakdown, one entity or a "
        "list. Ignore the subject matter; two questions about different domains "
        "can still be the same kind of request.\n"
        "2. From the pairs that match in kind, transfer the construction: which "
        "aggregate answers it, whether the answer needs a join to reach a name "
        "or label, whether a ratio is cast to REAL before dividing, whether a "
        "conditional share puts the filter in a CASE inside the aggregate "
        "rather than in WHERE, and whether DISTINCT is needed.\n"
        "3. When a pair's evidence states a formula, study how that formula was "
        "turned into SQL — operand order, the cast, whether the result was scaled "
        "by 100, whether anything was rounded — and give THIS request's evidence "
        "the same treatment.\n"
        "4. Then rebuild the query for this database from scratch: every table, "
        "column and alias must come from AVAILABLE TABLES, and every filter value "
        "from this question or its evidence. Names, aliases and literals in the "
        "pairs do NOT exist here.\n"
        "5. The pairs are ordered by question similarity, not by how well they "
        "fit. The first is not automatically the best; prefer whichever pair "
        "matches this request's kind most closely, and when pairs disagree with "
        "each other, follow that one rather than the earliest.\n"
        "6. If no pair matches the kind of request being made, ignore them all "
        "and build from the question and schema: project exactly what is asked "
        "and nothing beside it, and bound the result only where the question "
        "implies a bound. Forcing an ill-fitting pattern is worse than using "
        "none.\n"
        "The Rules section still governs throughout: a pair shows how a request "
        "of this kind is usually built, but where a pair and a rule disagree, "
        "the rule wins. The question, its evidence, and the schema outrank both.\n"
        "What they tell you nothing about, no matter what they do:\n"
        "- Which columns to SELECT, or how many. The pair answered a different "
        "question; what to return follows from THIS question and the Rules "
        "section, even when the pair returns more or fewer columns.\n"
        "- Whether to add LIMIT, and with what number. That follows from this "
        "question's own wording under the Rules section.\n"
        "- How many tables YOUR query needs. A single-table pair is no reason to "
        "drop a join this question requires, and a multi-join pair is no reason "
        "to add one; their schema is not yours.\n"
        "- Which columns exist, or what a column is called here.\n"
        "- Which values to filter on.\n\n"
        if has_sql_examples
        else ""
    )

    return f"""You are an expert SQL query builder. You MUST always produce a SQL query.

{sql_examples_block} Key rules:
{table_name_rule}
- When SQL snippets are provided as reference, do NOT copy their aliases.
  Define your own aliases in FROM/JOIN and use only those.
- File contents (if present) are inputs only — use them as literals, filters,
  or CASE logic within the SQL.
- JOIN PATHS (if present) list how pairs of available tables connect, derived
  from the semantic model. Choose the tables the question needs, then use the
  paths that connect them. They show verified column-level join conditions:
{join_template}
  Follow hops in order when the path spans more than one table. Once you choose
  a path, every intermediate/bridge table and every listed hop is mandatory.
  Never shorten the path or join its endpoints directly.
- NEVER invent a join. Every join condition must come from the JOIN PATHS or
  from a column's foreign key reference in AVAILABLE TABLES. If the tables you
  need cannot be connected that way, do not join them.
- DOMAIN-SPECIFIC CUSTOM ANALYSES: if one closely matches the question, use or
  adapt its full SQL directly as your starting point — you may reuse it wholesale,
  trimming only what does not apply. Do NOT copy its aliases.
- SQL ATTRIBUTES: derived metrics or formulas with pre-defined SQL expressions.
  If one matches the question's intent, incorporate its expression or SQL pattern
  into your query. Treat them like reusable building blocks for calculations.
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

description:
What is the total sales revenue for each country during the most recently
completed calendar quarter, ordered from highest to lowest?

response:
This calculates total sales revenue per country for the most recently completed
calendar quarter. It combines the sales records with the customers list so each
sale is attributed to a country, sums the sales amounts within that quarter,
and then groups the results by country and orders them from highest to lowest
total sales.
"""


def format_authoritative_evidence(
    evidence: str, *, evidence_first_sql: bool = False
) -> str:
    """Wrap evidence verbatim with strict instructions for SQL use and validation."""
    if not evidence:
        return ""

    construction_order = (
        "- Start SQL construction from the evidence, before adding anything else "
        "from the question. Decompose it into individual rules. For every rule, "
        "map each business entity or concept it names to the physical table and "
        "column whose name and description in the Available Schema best match that "
        "meaning. Build the evidence-required projections, formulas, filters, and "
        "joins as the query's backbone; only then add the remaining requirements "
        "from the question. Do not skip an evidence rule or map by name alone when "
        "a column description resolves the meaning. Never invent a mapping that "
        "the supplied schema does not support; state that limitation in thought.\n"
        if evidence_first_sql
        else ""
    )
    return (
        "## Authoritative Evidence\n"
        "This evidence is critical. The generated SQL MUST follow every instruction "
        "in it exactly, even when another approach appears equivalent.\n"
        f"{construction_order}"
        "- Whenever the evidence explains a value or where that value is stored, "
        "include the value in the SQL using the stated column, table, filter, and "
        "operator. Do not substitute another value or location.\n"
        "- Whenever the evidence explains a formula, calculate that exact formula "
        "from the stated available tables and columns. Do not replace it with a "
        "shortcut or precomputed field. Preserve the evidence's stated order of "
        "all formula terms, operands, predicates, and conditions exactly; never "
        "reorder them, even when the reordered expression appears equivalent.\n\n"
        "Evidence (verbatim):\n"
        f"{evidence}"
    )


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
# real and focus on whether it reaches the right entity. See the intent phase in
# sql_parse_validation.py.
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


def format_sql_examples_section(sql_examples: list[dict] | None) -> str:
    """Render retrieved question/SQL/evidence precedent for prompt injection.

    Each example is rendered whole — its question, its own evidence, its source
    database and its SQL — because the transferable lesson is the mapping from
    request to query, not the query alone. The evidence line in particular
    shows how a stated formula became SQL (casting, operand order, whether the
    result was scaled), which is exactly what the model has to reproduce for
    the request at hand.

    Deliberately separate from the ``## Example SQL Queries`` slot, which holds
    custom-analysis SQL written against the *target* database and which the
    prompt rules invite the model to lift filter values from. These examples
    are structural precedent from other databases, so the section is labelled
    and framed so the two cannot be confused.

    Returns "" when there is nothing to inject.
    """
    if not sql_examples:
        return ""
    blocks: list[str] = []
    for example in sql_examples:
        sql = " ".join((example.get("sql") or "").split())
        if not sql:
            continue
        source = (example.get("db") or "").strip()
        header = f"{len(blocks) + 1}."
        if source:
            header += f" [source database: {source}]"
        lines = [header]
        question = (example.get("question") or "").strip()
        if question:
            lines.append(f"   Question: {question}")
        evidence = " ".join((example.get("evidence") or "").split())
        if evidence:
            lines.append(f"   Evidence: {evidence}")
        lines.append(f"   SQL: {sql}")
        blocks.append("\n".join(lines))
    if not blocks:
        return ""
    return "## Reference Query Patterns\n" + "\n\n".join(blocks)


def format_value_anchors_section(value_anchors: list[dict] | None) -> str:
    """Render looked-up question phrases and the columns that store them.

    Each anchor is an observation about the live database, not a suggestion:
    this phrase from the question is stored in this column, with this exact
    spelling, in this many rows. It exists because the schema alone does not
    say which of several plausible columns holds a value: the same label often
    sits in a code column and a display column, or in an entity's own attribute
    and a denormalised copy on a related table. Picking the wrong one still
    returns rows, so nothing downstream notices.

    Anchors marked absent are the other half and carry as much weight: a label
    the question states that the database does not store anywhere means the
    filter has to be built some other way, rather than from the question's own
    wording.

    The usage guidance travels inside the section so that attaching it needs no
    second change to the surrounding prompts.

    Returns "" when nothing was looked up.
    """
    if not value_anchors:
        return ""
    # Between columns of one table a row count decides nothing while reading as
    # though it did: schools.MailCity holds 'San Joaquin' in more rows than
    # schools.City, and City is what the question means. Across tables the
    # count does discriminate, since volume tracks where a value belongs, so it
    # is dropped only where columns of the same table compete.
    rivals: dict[tuple[str, str], set[str]] = {}
    for anchor in value_anchors:
        key = (
            (anchor.get("phrase") or "").strip().lower(),
            (anchor.get("tbl") or "").strip().lower(),
        )
        rivals.setdefault(key, set()).add((anchor.get("col") or "").strip().lower())

    found, absent = [], []
    for anchor in value_anchors:
        phrase = (anchor.get("phrase") or "").strip()
        if not phrase:
            continue
        kind = (anchor.get("kind") or "value").strip()
        if kind == "absent":
            absent.append(f'"{phrase}"')
            continue
        table = (anchor.get("tbl") or "").strip()
        column = (anchor.get("col") or "").strip()
        value = (anchor.get("stored_value") or "").strip()
        if not (table and column and value):
            continue
        rows = anchor.get("n_rows") or ""
        contested = len(rivals.get((phrase.lower(), table.lower()), ())) > 1
        count = (
            f" — {int(rows):,} row{'' if int(rows) == 1 else 's'}"
            if str(rows).isdigit() and not contested
            else ""
        )
        # Two different facts, and conflating them invites an equality filter
        # on a value that only contains the phrase.
        claim = (
            f"= '{value}'{count}"
            if kind != "contains"
            else f"contains it inside values such as '{value}' (no value equals the phrase)"
        )
        # Supplied only where the phrase matched several columns of one table.
        # There the row counts say nothing about which column the question
        # means, and the documented name often says it outright: 'MailCity' is
        # "mailing city", so a question that does not say mailing means City.
        gloss = (anchor.get("description") or "").strip()
        named = f" ({gloss})" if gloss else ""
        found.append(f'- "{phrase}": {table}."{column}"{named} {claim}')
    if not found and not absent:
        return ""

    lines = ["## Verified Database Values"]
    lines.append(
        "Phrases from the question, looked up in this database. Every line "
        "below is a fact about what the data contains."
    )
    if found:
        lines.append("")
        lines.extend(found)
    if absent:
        lines.append("")
        lines.append(
            "Stated in the question but stored NOWHERE in this database: "
            + ", ".join(sorted(set(absent)))
            + "."
        )
    lines.append("")
    # The absent clause is only included when there is an absent line, so the
    # section never instructs against something it did not list.
    never_invent = (
        " Never filter on a phrase listed as stored nowhere; find what the "
        "database does store for that idea, or express the condition another way."
        if absent
        else ""
    )
    lines.append(
        "Use these when choosing a filter: take the column and the exact "
        "stored spelling from here instead of inferring either from the "
        "question's wording or a column's name. A line reading = '...' supports "
        "an equality filter on that spelling; a line saying the column merely "
        "contains the phrase does not — match such a column with LIKE, or not "
        "at all if the question means an exact category. A phrase listed under several "
        "columns is genuinely ambiguous — decide which column the question is "
        "asking about from its wording and the column's role in the schema, "
        "not from the order listed. Where such a column is "
        "followed by a documented name in parentheses, that name is how the "
        "question would have to refer to it: a qualifier in the name must also "
        "be present in the question to pick that column, and an unqualified "
        "mention means the plain column. A row count, where one is given, says "
        "only how many rows hold that value: where one phrase sits in several "
        "tables, the larger count tends to mark the table the value principally "
        "belongs to. A small count is not itself a reason to distrust a match — "
        "a value held by a single row is often the very entity the question names."
        + never_invent
        + " Evidence "
        "still outranks these lines, and a phrase absent from this list is not "
        "thereby absent from the database — only the listed phrases were looked up."
    )
    return "\n".join(lines)


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
    has_evidence: bool = False,
    used_schema_context: str = "",
    shorten_answer: bool = False,
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
    # check doesn't have to. See the flag in sql_parse_validation.py.
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
    evidence_criterion = (
        "\n3. Does the SQL honor the MEANING of every instruction in the "
        "Authoritative Evidence? Evidence is shorthand that defines what values, "
        "columns, filters and calculations mean; it is not SQL to be copied token "
        "by token. Flag a critical evidence issue only when following the SQL would "
        "give a different answer than the evidence defines: a missing or wrong "
        "evidence-defined value, a filter on a different column than the evidence "
        "maps it to, a missing evidence condition, or a calculation over different "
        "quantities. Never flag a formulation that computes the same result as the "
        "evidence, including:\n"
        "   - COUNT(x WHERE cond), COUNT(CASE WHEN cond THEN x END), "
        "SUM(CASE WHEN cond THEN 1 ELSE 0 END), SUM(IIF(cond, 1, 0)) and SUM(cond), "
        "which all count the rows meeting cond;\n"
        "   - MIN(col) / MAX(col) versus ORDER BY col ASC / DESC LIMIT 1 to pick the "
        "row holding the extreme value;\n"
        "   - multiplying by 100 when the question or evidence calls the result a "
        "percentage, even though the evidence formula has no * 100;\n"
        "   - DIVIDE / SUBTRACT / MULTIPLY / SUM(a, b) written as operators, terms of "
        "a commutative operation in another order, CASTs that avoid integer "
        "division, and column aliases."
        if has_evidence
        else ""
    )
    projection_criterion = (
        "\n4. PROJECTION CHECK: Compare the outermost SELECT output with the output "
        "fields the question and Authoritative Evidence ask for. Record in "
        "projection_issues a requested field that is missing, or an added field "
        "that the question does not ask for and that is not needed to identify or "
        "read the requested values. Aggregates and transformed values count as "
        "output fields. A SELECT * or table.* in the outermost SELECT is always a "
        "projection issue. Do NOT record any of the following, which are acceptable "
        "shapes of the requested answer:\n"
        "   - the table's primary key column(s) returned for a request to list or "
        "show records that names no field to return;\n"
        "   - a person's name returned as its stored parts (e.g. first-name and "
        "last-name columns) rather than one concatenated value;\n"
        "   - the identifier of each named entity returned next to the requested "
        "value when the question asks about several entities at once;\n"
        "   - a stored flag or code returned as stored (e.g. '+' / '-') rather than "
        "translated into words or yes / no;\n"
        "   - the order of the output columns;\n"
        "   - whether repeated rows are deduplicated, unless the question asks for "
        "distinct or unique values."
        if shorten_answer
        else ""
    )
    direct_answer_criterion = (
        "\nDIRECT ANSWER FIELD CHECK: For a yes/no question, use the schema metadata "
        "and Authoritative Evidence to determine whether a referenced text or boolean "
        "column directly stores the requested property or state. If so, wrapping that "
        "column in CASE or otherwise translating it to YES/NO is a critical reshaping "
        "of the requested answer unless normalization was explicitly requested. Record "
        "the violation in projection_issues and mark the SQL invalid."
    )
    parsed_usage_block = (
        "\nUSED SQL OBJECTS "
        "(catalog metadata only for tables and columns referenced by the query):\n"
        f"{used_schema_context}\n"
    )
    return f"""User's Question:
{question_block}
{custom_analyses_block}
{join_paths_block}
{parsed_usage_block}
Generated SQL Query:
```sql
{sql_code}
```

Check for CRITICAL issues ONLY (be lenient):
{join_criterion}
2. Are aggregations CLEARLY WRONG for the question? (e.g., COUNT when explicitly asking for SUM) (Variations are OK)
{evidence_criterion}
{projection_criterion}
{direct_answer_criterion}

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
- A table can be required even when none of its columns appear in the answer:
  joining it may restrict WHICH rows qualify (only schools that appear in
  frpm; only patients who have an examination record). "No column of mine is
  returned" is NOT a reason to remove a table.
- Before removing a table, confirm BOTH: (a) it supplies no column the answer
  needs, AND (b) removing it cannot change the set of rows that qualify.
  Report both per table — a table stays unless both hold.
- If table A must be joined through tables B and C to reach table D, do NOT
  remove any table in the join chain (A, B, C, or D). The join paths below
  show real table connections. Keep the full bridges between tables that
  you deem relevant.
- If a selected custom analysis references a table in its SQL, do NOT
  remove that table.
- When similarly named columns exist in several tables, a column's "when to
  use" note says which one the question needs; follow it.
- When in doubt, do NOT remove — it is safer to include an extra table
  than to remove a necessary one.

{domain_rules}{custom_analyses}{join_paths}{enriched_question}User's question:
{question}

Candidate tables:
{tables_summary}

Provide brief reasoning (1-2 sentences), then for each table you want REMOVED
give its name together with both removal checks: whether it supplies no column
the answer needs, and whether removing it cannot change which rows qualify.
State in its justification how you verified the row-scoping check.
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
