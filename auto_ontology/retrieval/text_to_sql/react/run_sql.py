# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ReAct agent's ``run_sql`` tool: check a query against the catalog, then run it.

Checking comes first and is deterministic. A query naming a table or column
the catalog does not have is answered with exactly which names are unknown —
and the closest real ones — without touching the database. The fixed
pipeline's parser cannot do this: :func:`parse_query_slim` skips names it
cannot resolve rather than reporting them, because it was written to record
queries, not to judge them.

A query that passes runs read-only under the chat statement timeout, and the
first rows come back as text. The model uses the same tool to look at stored
values (``SELECT DISTINCT status FROM ...``) and to try a draft, so nothing
here assumes which of the two it is doing.
"""

from __future__ import annotations

import difflib
import logging
from dataclasses import dataclass
from typing import Any, Protocol

import sqlglot
from sqlglot import exp

from auto_ontology.connectors.base import SQLDatabase
from auto_ontology.dal.datasources import fetch_all_schema_ids
from auto_ontology.retrieval.data_access.graph_schemas import get_schemas_by_ids
from auto_ontology.retrieval.text_to_sql.agents.sql_execution import (
    _deduplicate_result_columns,
    _sanitize_sql_for_dialect,
)
from auto_ontology.retrieval.text_to_sql.chat_sql import execute_chat_sql
from auto_ontology.retrieval.text_to_sql.connector_routing import (
    resolve_connector_from_tables,
)
from auto_ontology.retrieval.text_to_sql.react import config
from auto_ontology.retrieval.text_to_sql.react.context import ToolContext

logger = logging.getLogger(__name__)

_MAX_PROBLEMS = 10
_MAX_CELL_CHARS = 120


class CatalogNames(Protocol):
    """What the hallucination check needs to know about the catalog."""

    def columns(self, table: str, schema: str | None) -> list[str] | None:
        """Column names of *table*, or ``None`` when the catalog has no such table."""


class SchemaCatalog:
    """:class:`CatalogNames` over the catalog's in-memory :class:`Schema` objects."""

    def __init__(self, schemas: dict[str, Any]):
        self._schemas = schemas

    def _schema_for(self, table: str, schema: str | None) -> Any | None:
        if schema:
            for key, candidate in self._schemas.items():
                if key.lower() == schema.lower() and candidate.table_exists(table):
                    return candidate
            return None
        for candidate in self._schemas.values():
            if candidate.table_exists(table):
                return candidate
        return None

    def columns(self, table: str, schema: str | None) -> list[str] | None:
        owner = self._schema_for(table, schema)
        if owner is None:
            return None
        columns_map = getattr(owner, "columns_map", None)
        if columns_map is not None:
            return list(columns_map.get(table.lower(), {}))
        node = owner.get_table_node(table)
        return list(owner.tables_to_columns.get(node, []))


@dataclass
class CheckResult:
    """A parsed query, or why it cannot run."""

    problems: list[str]
    statement: exp.Expression | None = None


def _derived_names(statement: exp.Expression) -> set[str]:
    """Names that are not catalog tables: CTEs and aliased subqueries."""
    names = {cte.alias_or_name.lower() for cte in statement.find_all(exp.CTE)}
    names |= {sq.alias.lower() for sq in statement.find_all(exp.Subquery) if sq.alias}
    return names


def _suggest(name: str, options: list[str]) -> str:
    close = difflib.get_close_matches(name.lower(), [o.lower() for o in options], n=3)
    return f" (did you mean: {', '.join(close)}?)" if close else ""


def find_unknown_names(statement: exp.Expression, catalog: CatalogNames) -> list[str]:
    """Tables and columns *statement* names that the catalog does not have.

    Qualified columns are checked against the table their qualifier points at.
    An unqualified column is checked against every catalog table the query
    reads, and only when the query has no CTE or subquery that could be
    supplying it — a name that might be an inner query's output is not
    reported, because a false "unknown column" sends the model after a mistake
    it did not make.
    """
    derived = _derived_names(statement)
    output_aliases = {a.alias.lower() for a in statement.find_all(exp.Alias) if a.alias}

    problems: list[str] = []
    tables: dict[str, list[str] | None] = {}
    for table in statement.find_all(exp.Table):
        name = table.name
        if not name or name.lower() in derived:
            continue
        columns = catalog.columns(name, table.db or None)
        tables[table.alias_or_name.lower()] = columns
        tables.setdefault(name.lower(), columns)
        if columns is None:
            problems.append(f"Unknown table: {table.sql()}")

    known_columns = {
        c.lower() for columns in tables.values() if columns for c in columns
    }
    for column in statement.find_all(exp.Column):
        name = column.name
        if not name or isinstance(column.this, exp.Star):
            continue
        qualifier = (column.table or "").lower()
        if qualifier:
            if qualifier in derived or qualifier not in tables:
                continue
            columns = tables[qualifier]
            if columns is not None and name.lower() not in {c.lower() for c in columns}:
                problems.append(
                    f"Unknown column: {column.table}.{name}{_suggest(name, columns)}"
                )
        elif (
            not derived
            and name.lower() not in output_aliases
            and tables
            and all(cols is not None for cols in tables.values())
            and name.lower() not in known_columns
        ):
            problems.append(
                f"Unknown column: {name}{_suggest(name, sorted(known_columns))}"
            )
    return list(dict.fromkeys(problems))[:_MAX_PROBLEMS]


def check_sql(sql: str, dialect: str | None, catalog: CatalogNames) -> CheckResult:
    """Parse *sql*, refuse anything but one read query, and find unknown names."""
    try:
        statements = [s for s in sqlglot.parse(sql, read=dialect or None) if s]
    except sqlglot.errors.ParseError as error:
        return CheckResult([f"SQL does not parse: {str(error).splitlines()[0]}"])
    if len(statements) != 1:
        return CheckResult(["Send exactly one SQL statement."])
    statement = statements[0]
    if not isinstance(statement, exp.Query):
        return CheckResult(["Only read queries (SELECT or WITH … SELECT) can run."])
    return CheckResult(find_unknown_names(statement, catalog), statement)


def catalog_for(ctx: ToolContext) -> CatalogNames:
    """The catalog the checks resolve names against, loaded once per episode."""
    if ctx.schemas is None:
        ctx.schemas = get_schemas_by_ids(fetch_all_schema_ids())
    return SchemaCatalog(ctx.schemas)


def connector_for(ctx: ToolContext) -> SQLDatabase | None:
    """The database the episode's SQL runs on.

    The target database when the caller named one; otherwise the connector that
    owns the tables ``retrieve`` has shown, or the only one configured.
    """
    if ctx.database_name:
        for connector in ctx.connectors:
            if getattr(connector, "database_name", None) == ctx.database_name:
                return connector
    if len(ctx.connectors) == 1:
        return ctx.connectors[0]
    try:
        return resolve_connector_from_tables(
            list(ctx.seen_tables.values()), ctx.connectors
        )
    except ValueError:
        return None


def _cell(value: Any) -> str:
    text = "NULL" if value is None else str(value)
    return text if len(text) <= _MAX_CELL_CHARS else text[: _MAX_CELL_CHARS - 1] + "…"


def render_rows(frame: Any) -> str:
    """The first rows of *frame* as pipe-separated text, with the total count."""
    total = len(frame)
    if total == 0:
        return (
            "0 rows. If rows were expected, check the filter literals against the "
            "stored values (SELECT DISTINCT <column> ...)."
        )
    shown = frame.head(config.max_result_rows())
    header = " | ".join(str(c) for c in shown.columns)
    lines = [header] + [
        " | ".join(_cell(v) for v in row) for row in shown.itertuples(index=False)
    ]
    summary = f"{total} rows" + (
        f", first {len(shown)} shown" if total > len(shown) else ""
    )
    return summary + "\n" + "\n".join(lines)


def _clip(text: str) -> str:
    limit = config.max_observation_chars()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def run_sql(ctx: ToolContext, sql: str) -> str:
    """Check *sql* against the catalog and, if it passes, run it."""
    sql = (sql or "").strip().rstrip(";")
    if not sql:
        return "Error: sql is empty."
    if ctx.sql_runs >= config.max_sql_runs():
        if config.submit_tool_enabled():
            return (
                "Error: no run_sql calls left in this episode. Submit your best query."
            )
        return (
            "Error: no exploratory run_sql calls left in this episode. Run your "
            "best query with final=true."
        )
    connector = connector_for(ctx)
    if connector is None:
        return "Error: no database to run on. Retrieve the tables you need first."
    dialect = getattr(connector, "dialect", None)
    check = check_sql(sql, dialect, catalog_for(ctx))
    if check.problems:
        return "Not run. Fix these first:\n- " + "\n- ".join(check.problems)

    ctx.sql_runs += 1
    try:
        frame = execute_chat_sql(
            connector,
            _sanitize_sql_for_dialect(sql, dialect or ""),
            kind="ReAct agent SQL",
        )
    except Exception as error:  # noqa: BLE001 — the model must see the failure
        logger.info("ReAct run_sql failed: %s", error)
        return _clip(f"Error from the database: {' '.join(str(error).split())}")
    frame = _deduplicate_result_columns(frame)
    text = render_rows(frame)
    if config.grep_enabled():
        text = _save_page(ctx, frame, text)
    return _clip(text)


def _save_page(ctx: ToolContext, frame: Any, text: str) -> str:
    """Keep a result too long to show for ``grep``, and say so on its first line.

    The note goes on the summary line because clipping keeps the start.
    """
    if (
        len(frame) <= config.max_result_rows()
        and len(text) <= config.max_observation_chars()
    ):
        return text
    kept = frame.head(config.max_page_rows())
    ctx.pages.append(kept)
    page = len(ctx.pages)
    rows = (
        f"all {len(kept)} rows"
        if len(kept) == len(frame)
        else f"its first {len(kept)} rows"
    )
    summary, _, rest = text.partition("\n")
    note = (
        f"{summary}. Saved as page {page}: grep(page={page}, pattern=...) "
        f"searches {rows}, and offset reads further rows."
    )
    return note + "\n" + rest
