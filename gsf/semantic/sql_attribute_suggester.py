# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Automatic SqlAttribute suggestion from SQL query history.

After the semantic FK pass this module runs once per compilation:

1. For every Term, fetch the SQL queries linked to its tables.
2. Extract SQL expressions (filters, aggregations, functions) from each query.
3. Score each expression by summing the 3-month usage counters of the queries
   it appears in.
4. Send the top 10 expressions per term to an LLM that judges whether any
   should become named SqlAttributes.
5. Persist new suggestions in Neo4j (skipping existing semantic SqlAttributes)
   and embed only the newly written nodes into the semantic VDB.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from gsf.dal.terms import fetch_table_schema_map, fetch_terms_with_sqls
from gsf.semantic.constants import SEMANTIC_SOURCE, SQL_ATTR_SOURCE_SQL
from gsf.server.sql_attributes.service import (
    SqlAttributeNameConflict,
    SqlAttributeSqlError,
    create_sql_attribute,
)
from gsf.utils.llm_invoke import (
    get_non_reasoning_llm_client,
    invoke_with_structured_output,
)

logger = logging.getLogger(__name__)

_TOP_N = 10
_TERM_WORKERS = 4
_COUNTER_RE = re.compile(r"^count_monthly_(\d{4})_(\d{2})$")


# ---------------------------------------------------------------------------
# LLM output schema
# ---------------------------------------------------------------------------


class _SuggestedAttr(BaseModel):
    name: str
    description: str
    expression: str


class _Suggestions(BaseModel):
    suggestions: list[_SuggestedAttr]


# ---------------------------------------------------------------------------
# Usage scoring
# ---------------------------------------------------------------------------


def _latest_3month_score(props: dict[str, Any]) -> float:
    """Return the sum of the three most recent monthly counters in *props*."""
    monthly: list[tuple[tuple[int, int], float]] = []
    for key, val in props.items():
        m = _COUNTER_RE.match(key)
        if m and val is not None:
            try:
                monthly.append(((int(m.group(1)), int(m.group(2))), float(val)))
            except (TypeError, ValueError):
                pass
    monthly.sort(key=lambda x: x[0], reverse=True)
    return sum(v for _, v in monthly[:3])


# ---------------------------------------------------------------------------
# Expression extraction via sqlglot
# ---------------------------------------------------------------------------

_BARE_KEYWORDS = frozenset(
    {"select", "where", "from", "join", "on", "union", "intersect", "except", "having"}
)


def _extract_expressions(sql_text: str) -> list[str]:
    """Parse *sql_text* and return deduplicated SQL sub-expressions.

    Extracted categories:
    - WHERE / JOIN ON predicates (comparisons with at least one non-column operand)
    - Aggregations, anonymous functions, CASE expressions
    - HAVING predicates
    - Full JOIN ON predicate (when it contains non-FK conditions)
    - Whole UNION / INTERSECT / EXCEPT expression and each individual branch

    Excluded:
    - Bare column references, star, plain literals
    - Pure FK equalities (col_a = col_b, both sides are columns with no literal)
    - Degenerate sqlglot emissions that are just a SQL keyword token
    """
    try:
        import sqlglot
        import sqlglot.expressions as exp

        ast = sqlglot.parse_one(sql_text, error_level=sqlglot.ErrorLevel.IGNORE)
        if ast is None:
            return []
    except Exception:
        return []

    seen: set[str] = set()
    results: list[str] = []

    cmp_types = (
        exp.EQ,
        exp.NEQ,
        exp.GT,
        exp.GTE,
        exp.LT,
        exp.LTE,
        exp.In,
        exp.Between,
        exp.Like,
        exp.Is,
    )

    def _is_pure_fk_eq(node: exp.Expression) -> bool:
        """Return True for bare col = col equalities (no literals on either side)."""
        return (
            isinstance(node, exp.EQ)
            and isinstance(node.left, exp.Column)
            and isinstance(node.right, exp.Column)
        )

    def _add(node: exp.Expression) -> None:
        text = node.sql().strip()
        key = text.lower()
        if len(text) < 5 or key in seen:
            return
        if isinstance(node, (exp.Column, exp.Star, exp.Literal)):
            return
        if key in _BARE_KEYWORDS:
            return
        if _is_pure_fk_eq(node):
            return
        seen.add(key)
        results.append(text)

    # WHERE predicates — each individual comparison condition
    where = ast.find(exp.Where)
    if where:
        for node in where.walk():
            if isinstance(node, cmp_types):
                _add(node)

    # Aggregations, anonymous functions, CASE anywhere in the query
    for node in ast.walk():
        if isinstance(node, (exp.AggFunc, exp.Anonymous, exp.Case)):
            _add(node)

    # HAVING expression
    having = ast.find(exp.Having)
    if having and having.this:
        _add(having.this)

    # JOIN ON conditions — capture the full ON predicate and each sub-predicate.
    # These encode business rules beyond simple FK equality.
    for join in ast.find_all(exp.Join):
        on = join.args.get("on")
        if on:
            _add(on)
            for node in on.walk():
                if isinstance(node, cmp_types):
                    _add(node)

    # UNION / INTERSECT / EXCEPT — capture the whole set-operation and each branch.
    for set_op in ast.find_all(exp.Union, exp.Intersect, exp.Except):
        _add(set_op)
        for branch in (set_op.left, set_op.right):
            if branch is not None:
                _add(branch)

    return results


# ---------------------------------------------------------------------------
# Expression → valid SELECT transformation
# ---------------------------------------------------------------------------


def _try_wrap_in_select(
    expression: str,
    source_sql: str,
    expr_ast: object,
    is_agg: bool,
) -> str | None:
    """Build ``SELECT … FROM … [JOINs] [WHERE expression]`` from *source_sql*.

    Finds the innermost SELECT node whose table aliases cover every alias
    referenced in *expression*, extracts its FROM + JOIN clauses, and wraps
    the expression as a WHERE predicate (conditions/filters) or a SELECT-list
    item (aggregations).

    Returns ``None`` when parsing fails or no suitable SELECT is found.
    """
    try:
        import sqlglot
        import sqlglot.expressions as exp

        src_ast = sqlglot.parse_one(source_sql, error_level=sqlglot.ErrorLevel.IGNORE)
    except Exception:
        return None
    if src_ast is None:
        return None

    # Aliases referenced by columns inside the expression.
    expr_table_aliases: set[str] = set()
    if expr_ast is not None:
        try:
            import sqlglot.expressions as exp  # noqa: F811 (re-import inside try block)

            for col in expr_ast.find_all(exp.Column):  # type: ignore[attr-defined]
                if col.table:
                    expr_table_aliases.add(col.table.lower())
        except Exception:
            pass

    best_select = None
    try:
        import sqlglot.expressions as exp  # noqa: F811

        for sel in src_ast.find_all(exp.Select):  # type: ignore[attr-defined]
            from_part = sel.args.get("from_")
            if from_part is None:
                continue
            sel_aliases: set[str] = set()
            for tbl in sel.find_all(exp.Table):
                if tbl.alias:
                    sel_aliases.add(tbl.alias.lower())
                sel_aliases.add(tbl.name.lower())
            if expr_table_aliases.issubset(sel_aliases) or not expr_table_aliases:
                best_select = sel
                break  # take the first (outermost) matching one

        if best_select is None:
            # Fallback: first SELECT that has a FROM clause
            for sel in src_ast.find_all(exp.Select):
                if sel.args.get("from_"):
                    best_select = sel
                    break
    except Exception:
        return None

    if best_select is None:
        return None

    try:
        from_clause = best_select.args.get("from_")
        if from_clause is None:
            return None
        joins = best_select.args.get("joins") or []

        # Only keep JOINs whose table name/alias is actually referenced by
        # columns in the expression.  When the expression has no explicit table
        # qualifiers (expr_table_aliases is empty) we omit all JOINs and just
        # use the primary FROM table.
        if expr_table_aliases:
            filtered_joins = [
                j
                for j in joins
                if any(
                    (tbl.alias.lower() if tbl.alias else tbl.name.lower())
                    in expr_table_aliases
                    for tbl in j.find_all(exp.Table)  # type: ignore[attr-defined]
                )
            ]
        else:
            filtered_joins = []

        from_str = from_clause.sql()
        joins_str = " ".join(j.sql() for j in filtered_joins)
    except Exception:
        return None

    parts = ["SELECT", expression if is_agg else "*", from_str]
    if joins_str:
        parts.append(joins_str)
    if not is_agg:
        parts.append(f"WHERE {expression}")
    return " ".join(parts)


def _qualify_table_names(sql: str, schema_map: dict[str, str]) -> str:
    """Add ``schema.`` prefix to bare table references in *sql*.

    Only tables whose names appear in *schema_map* (table_name_lower →
    schema_name) and that do not already carry a schema qualifier are updated.
    Returns *sql* unchanged when *schema_map* is empty or parsing fails.
    """
    if not schema_map:
        return sql
    try:
        import sqlglot
        import sqlglot.expressions as exp

        ast = sqlglot.parse_one(sql, error_level=sqlglot.ErrorLevel.IGNORE)
        if ast is None:
            return sql

        def _add_schema(node: object) -> object:
            if (
                isinstance(node, exp.Table)
                and not node.args.get("db")
                and not node.args.get("catalog")
            ):
                schema = schema_map.get(node.name.lower())
                if schema:
                    n = node.copy()
                    n.set("db", exp.to_identifier(schema))
                    return n
            return node  # type: ignore[return-value]

        return ast.transform(_add_schema).sql()
    except Exception:
        return sql


def _build_valid_sql(
    expression: str,
    source_sql_texts: list[str],
    schema_map: dict[str, str] | None = None,
) -> str:
    """Return a syntactically valid SELECT statement wrapping *expression*.

    Cases:
    * Expression is already a full SELECT / UNION / INTERSECT / EXCEPT:
      returned as-is (with table names qualified if *schema_map* provided).
    * Filter or JOIN condition: ``SELECT * FROM <from+joins> WHERE <expression>``
      where the FROM clause is extracted from the first source SQL that contains
      the expression.
    * Aggregation / function: ``SELECT <expression> FROM <from+joins>``.

    Table names in the result are schema-qualified using *schema_map* when
    provided.

    If no source SQL can provide the context, raises ``ValueError`` so the
    caller can decide how to proceed.
    """
    _schema_map = schema_map or {}

    def _qualify(sql: str) -> str:
        return _qualify_table_names(sql, _schema_map)

    try:
        import sqlglot
        import sqlglot.expressions as exp

        ast = sqlglot.parse_one(expression, error_level=sqlglot.ErrorLevel.IGNORE)
        if isinstance(ast, (exp.Select, exp.Union, exp.Intersect, exp.Except)):
            return _qualify(expression)
    except Exception:
        ast = None

    # Determine whether the expression is an aggregation/function.
    is_agg = False
    try:
        import sqlglot.expressions as exp  # noqa: F811

        if ast is not None:
            is_agg = isinstance(ast, (exp.AggFunc, exp.Anonymous))
    except Exception:
        pass

    # Try to wrap using context from each source SQL.
    expr_lower = expression.lower()
    for sql_text in source_sql_texts:
        if not sql_text:
            continue
        # Quick pre-filter: skip source SQLs that don't share any token with the
        # expression (avoids pointless parse overhead).
        if not any(
            tok in sql_text.lower() for tok in expr_lower.split() if len(tok) > 3
        ):
            continue
        wrapped = _try_wrap_in_select(expression, sql_text, ast, is_agg)
        if wrapped:
            return _qualify(wrapped)

    # No source SQL could provide context — try every source SQL without filtering.
    for sql_text in source_sql_texts:
        if not sql_text:
            continue
        wrapped = _try_wrap_in_select(expression, sql_text, ast, is_agg)
        if wrapped:
            return _qualify(wrapped)

    raise ValueError(
        f"Cannot build a valid SELECT for expression {expression!r}: "
        "no source SQL context available."
    )


# ---------------------------------------------------------------------------
# Score and rank across all SQLs for a term
# ---------------------------------------------------------------------------


def _rank_expressions(sqls: list[dict[str, Any]]) -> list[tuple[str, float, list[str]]]:
    """Return ``(expression, score, sql_ids)`` triples sorted by score descending.

    *sql_ids* lists the Neo4j Sql node ids of every query the expression
    appeared in — used later to link the created SqlAttribute back to its
    source queries.
    """
    scores: dict[str, float] = defaultdict(float)
    expr_sql_ids: dict[str, list[str]] = defaultdict(list)
    for item in sqls:
        sql_text = item.get("sql_text") or ""
        sql_id = item.get("sql_id") or ""
        sql_score = _latest_3month_score(item.get("props") or {})
        if not sql_text:
            continue
        for expr in _extract_expressions(sql_text):
            scores[expr] += sql_score
            if sql_id and sql_id not in expr_sql_ids[expr]:
                expr_sql_ids[expr].append(sql_id)
    return [
        (expr, score, expr_sql_ids[expr])
        for expr, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)
    ]


# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------


def _judge_with_llm(
    term_name: str,
    term_description: str,
    expressions: list[tuple[str, float]],
) -> list[_SuggestedAttr]:
    """Ask the LLM whether any of *expressions* should become SqlAttributes."""
    if not expressions:
        return []

    expr_block = "\n".join(
        f"{i + 1}. [{score:.0f} uses] {expr}"
        for i, (expr, score) in enumerate(expressions)
    )

    llm = get_non_reasoning_llm_client(temperature=0.0)
    messages = [
        SystemMessage(
            content=(
                "You are an expert data engineer reviewing SQL expressions to decide "
                "which ones represent reusable business logic worth naming. "
                "A SqlAttribute is a named, reusable SQL expression — a filter, "
                "aggregation, or calculation — that captures a clear business concept "
                "and is specific enough to be useful while general enough to recur."
            )
        ),
        HumanMessage(
            content=(
                f"Business term: {term_name}\n"
                f"Description: {term_description or '(none)'}\n\n"
                f"Top SQL expressions extracted from queries on tables for this term "
                f"(ranked by 3-month execution count):\n\n"
                f"{expr_block}\n\n"
                f"Which expressions should become a named SqlAttribute for the term "
                f"'{term_name}'?\n\n"
                "Rules you MUST follow:\n"
                "  1. ALWAYS mark an expression as a SqlAttribute if it compares a column "
                "against a literal value (e.g. status = 'active', amount > 1000, "
                "type IN ('A','B')). Literal values encode business thresholds and "
                "categories that are exactly what SqlAttributes are designed to capture. "
                "Exception: do NOT promote an expression whose only literal is an integer "
                "that looks like a surrogate or primary-key identifier "
                "(e.g. request_id = 50000, id = 12). Such values are instance-specific "
                "record lookups, not reusable business logic. Only promote an expression "
                "containing such an integer if the surrounding context also encodes a "
                "reusable business rule (e.g. the same WHERE clause additionally filters "
                "on a status, category, or date range).\n"
                "  2. NEVER mark an expression as a SqlAttribute if it only references "
                "columns from the same table with no literal value, constant, or "
                "cross-table condition (e.g. start_date < end_date). Such expressions "
                "describe structural integrity, not reusable business logic.\n"
                "  4. NEVER mark an IS NULL / IS NOT NULL check on a surrogate or "
                "primary-key column (any column named 'id' or ending in '_id') as a "
                "SqlAttribute. These are LEFT JOIN absence checks — structural plumbing "
                "that detects whether a join matched, not a business concept "
                "(e.g. request_task_collaborators.id IS NULL). "
                "Exception: IS NULL on non-key columns that carry business meaning IS "
                "valid (e.g. deleted_at IS NULL, approved_by IS NULL).\n"
                "  3. ALWAYS mark a JOIN ON condition or a UNION/INTERSECT/EXCEPT branch "
                "as a SqlAttribute when it appears frequently AND any of the following "
                "is true:\n"
                "     a. It encodes obvious business logic — "
                "e.g. a join that filters to active records, a union arm that defines a "
                "named sub-population, or a cross-table condition capturing a business "
                "relationship.\n"
                "     b. It is used to define a meaningful business alias — e.g. a SELECT "
                "branch or subquery whose result column is aliased with a descriptive "
                "business name (e.g. 'AS active_assignees', 'AS overdue_tasks'). "
                "The alias itself signals that the expression represents a named concept "
                "worth capturing.\n"
                "     c. A UNION combines two or more SELECT branches that each use a "
                "different join path to reach the same entity type (e.g. one branch joins "
                "via a direct ownership FK, another via a membership/collaborator table). "
                "This pattern structurally defines a named aggregate concept — all members "
                "of a group regardless of their role. Promote the whole UNION as a "
                "SqlAttribute representing that concept (e.g. 'Task Participants', "
                "'Order Stakeholders'). When naming and describing it, ignore any "
                "instance-specific WHERE predicates (like record-ID literals) and focus "
                "on the reusable structural pattern.\n"
                "     Ignore pure FK-only joins and UNION branches that select from a "
                "single table with no join or filter.\n\n"
                "For each chosen expression provide:\n"
                "  • name — user-friendly Title Case label with spaces between words, "
                "matching the ColumnAttribute naming style "
                "(e.g. 'Active Customer Filter', 'Revenue Above Threshold', "
                "'Last 90 Days Activity')\n"
                "  • description — what business logic it captures, why it matters for "
                "this term, and when a data consumer would apply it\n"
                "  • expression — the exact SQL expression copied verbatim\n\n"
                "Return an empty list if none qualify."
            )
        ),
    ]

    result = invoke_with_structured_output(llm, messages, _Suggestions)
    return result.suggestions if result else []


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def suggest_sql_attributes(database_name: str) -> int:
    """Suggest and persist SqlAttributes from query history for every Term.

    For each Term:
      1. Collects SQL queries from connected tables.
      2. Extracts and scores expressions by 3-month usage.
      3. Sends the top 10 to the LLM for SqlAttribute judgment.
      4. Transforms each approved expression into a valid SELECT statement.
      5. Persists new SqlAttributes via the service (validates SQL, creates Sql
         node, embeds).  Existing ones (same name) are skipped automatically.
      6. Links each new SqlAttribute to the original source Sql nodes.

    Returns the total number of new SqlAttribute nodes written.
    """
    logger.info("Collecting SQL expressions per term…")
    term_rows = fetch_terms_with_sqls(SEMANTIC_SOURCE)

    if not term_rows:
        logger.info(
            "No terms with associated SQL queries — skipping SqlAttribute suggestion."
        )
        return 0

    logger.info("Found %d term(s) with SQL queries.", len(term_rows))

    schema_map: dict[str, str] = {}
    try:
        schema_map = fetch_table_schema_map(database_name)
        logger.info(
            "Loaded schema map: %d table(s) with schema qualifiers.", len(schema_map)
        )
    except Exception:
        logger.warning(
            "Could not fetch table→schema map; table names will not be schema-qualified.",
            exc_info=True,
        )

    new_attr_ids: list[str] = []
    # Shared across workers — guards seen_expressions to prevent duplicate
    # SqlAttribute nodes when terms share the same underlying table.
    seen_expressions: set[str] = set()
    seen_lock = Lock()

    def _process_term(row: dict[str, Any]) -> list[str]:
        term_id: str = row["term_id"]
        term_name: str = row["term_name"]
        term_description: str = row.get("term_description") or ""
        sqls: list[dict] = row.get("sqls") or []

        ranked = _rank_expressions(sqls)
        top = ranked[:_TOP_N]
        if not top:
            logger.debug("Term %r: no scoreable expressions — skipping.", term_name)
            return []

        logger.info(
            "Term %r: %d SQL(s), %d unique expression(s) — sending top %d to LLM.",
            term_name,
            len(sqls),
            len(ranked),
            len(top),
        )

        # _judge_with_llm only needs (expr, score) pairs.
        suggestions = _judge_with_llm(
            term_name,
            term_description,
            [(expr, score) for expr, score, _ in top],
        )

        if not suggestions:
            logger.info("Term %r: LLM suggested 0 SqlAttributes.", term_name)
            return []

        logger.info(
            "Term %r: LLM suggested %d SqlAttribute(s).", term_name, len(suggestions)
        )

        # Build lookup structures from the ranked results.
        sql_text_by_id = {
            item["sql_id"]: item["sql_text"]
            for item in sqls
            if item.get("sql_id") and item.get("sql_text")
        }
        expr_to_sql_ids: dict[str, list[str]] = {
            expr: sql_ids for expr, _, sql_ids in ranked
        }

        created: list[str] = []
        for s in suggestions:
            expr_key = s.expression.strip().lower()
            with seen_lock:
                if expr_key in seen_expressions:
                    logger.debug(
                        "SqlAttribute expression already handled this run — "
                        "skipping duplicate for term %r: %r",
                        term_name,
                        s.expression,
                    )
                    continue
                seen_expressions.add(expr_key)

            source_sql_ids = expr_to_sql_ids.get(s.expression, [])
            source_sql_texts = [
                sql_text_by_id[sid] for sid in source_sql_ids if sid in sql_text_by_id
            ]

            # Transform the expression into a valid SELECT statement.
            try:
                valid_sql = _build_valid_sql(s.expression, source_sql_texts, schema_map)
            except ValueError as exc:
                logger.warning(
                    "Term %r: cannot build valid SQL for %r — skipping: %s",
                    term_name,
                    s.name,
                    exc,
                )
                continue

            # Persist via the service (validates SQL, creates Sql node, embeds).
            try:
                row_result = create_sql_attribute(
                    name=s.name,
                    description=s.description,
                    expression=valid_sql,
                    term_id=term_id,
                    connector=database_name,
                    source=SQL_ATTR_SOURCE_SQL,
                )
                attr_id: str = row_result["id"]
            except SqlAttributeNameConflict:
                logger.debug("SqlAttribute %r already exists — skipping.", s.name)
                continue
            except SqlAttributeSqlError as exc:
                logger.warning(
                    "Term %r: generated SQL for %r failed schema validation — "
                    "skipping: %s",
                    term_name,
                    s.name,
                    exc,
                )
                continue
            except Exception as exc:
                logger.warning(
                    "Term %r: unexpected error creating SqlAttribute %r — skipping: %s",
                    term_name,
                    s.name,
                    exc,
                )
                continue

            logger.info("Created SqlAttribute %r (id=%s).", s.name, attr_id)

            created.append(attr_id)
        return created

    with ThreadPoolExecutor(max_workers=_TERM_WORKERS) as pool:
        futures = {pool.submit(_process_term, row): row for row in term_rows}
        for future in as_completed(futures):
            try:
                new_attr_ids.extend(future.result())
            except Exception:
                row = futures[future]
                logger.exception("Error processing term %r", row.get("term_name"))

    total = len(new_attr_ids)
    logger.info(
        "SqlAttribute suggestion pass complete — %d new node(s) written.", total
    )
    return total
