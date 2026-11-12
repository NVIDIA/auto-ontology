# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prompt text shared outside the ReAct agent's own prompts."""

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


_POSTGRES_DIALECT_RULES = (
    "**PostgreSQL-specific (STRICT — these will error at execution)**\n"
    "- WHERE and HAVING cannot reference SELECT aliases. Repeat the full expression or wrap in a subquery/CTE.\n"
    "- GROUP BY cannot reference SELECT aliases. Repeat the full expression "
    "(including CASE WHEN blocks) in GROUP BY, or wrap the query in a subquery/CTE.\n"
    "- The JSONB ->> operator always returns TEXT, even for JSON numbers. Whenever a JSONB "
    "field holds a numeric metric, score, count, percentage, or measurement, you MUST cast "
    "the ->> expression to NUMERIC (e.g. CAST(col->'a'->>'b' AS NUMERIC)) in SELECT as well "
    "as in WHERE/ORDER BY — never project a numeric JSONB field as raw text.\n"
    "- In PostgreSQL, AVG() and arithmetic (+, -, *, /) on REAL/DOUBLE PRECISION columns "
    "return floating-point (double precision); wrap computed AVG() and arithmetic metric "
    "projections in CAST(... AS NUMERIC) so fractional metrics are returned as NUMERIC.\n"
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
