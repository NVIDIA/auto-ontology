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
from auto_ontology.dal.datasources import (
    fetch_all_schema_ids,
    fetch_schema_ids_for_database,
)
from auto_ontology.retrieval.data_access.graph_schemas import get_schemas_by_ids
from auto_ontology.retrieval.text_to_sql.agents.sql_execution import (
    _deduplicate_result_columns,
    _sanitize_sql_for_dialect,
)
from auto_ontology.retrieval.text_to_sql.chat_sql import execute_chat_sql
from auto_ontology.retrieval.text_to_sql.connector_routing import (
    databases_for_sql,
    resolve_connector_for_sql,
    wildcard_for,
)
from auto_ontology.retrieval.text_to_sql.react import config
from auto_ontology.retrieval.text_to_sql.react.context import ToolContext

logger = logging.getLogger(__name__)

_MAX_PROBLEMS = 10
_MAX_CELL_CHARS = 120

# Nodes that write or lock even under a SELECT root: Postgres runs DML inside a
# CTE, ``SELECT ... INTO`` creates a table, ``FOR UPDATE``/``FOR SHARE`` lock rows.
_WRITING_NODES: tuple[type[exp.Expression], ...] = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Merge,
    exp.Create,
    exp.Drop,
    exp.Command,
    exp.Into,
    exp.Lock,
)

# A read-only transaction does not stop these: session advisory locks outlive
# the transaction and return to the pool, backend signalling and server file
# access ignore it, dblink writes over a connection of its own, and
# query_to_xml runs SQL from a string this check never sees.
_FORBIDDEN_FUNCTIONS = frozenset(
    {
        "currval",
        "lastval",
        "nextval",
        "setval",
        "set_config",
        "query_to_xml",
        "query_to_xmlschema",
        "query_to_xml_and_xmlschema",
        "cursor_to_xml",
        "cursor_to_xmlschema",
    }
)
_FORBIDDEN_FUNCTION_PREFIXES = ("pg_", "dblink", "lo_")
_ALLOWED_PG_FUNCTIONS = frozenset({"pg_typeof", "pg_column_size", "pg_size_pretty"})


class CatalogNames(Protocol):
    """What the hallucination check needs to know about the catalog."""

    def columns(self, table: str, schema: str | None) -> list[str] | None:
        """Column names of *table*, or ``None`` when the catalog has no such table."""

    def schemas_with(self, table: str) -> list[str]:
        """The schemas that have a table named *table*."""


class SchemaCatalog:
    """:class:`CatalogNames` over the catalog's in-memory :class:`Schema` objects."""

    def __init__(self, schemas: dict[str, Any]):
        self._schemas = schemas

    def _owners(self, table: str, schema: str | None) -> list[Any]:
        return [
            candidate
            for key, candidate in self._schemas.items()
            if (not schema or (key or "").lower() == schema.lower())
            and candidate.table_exists(table)
        ]

    def columns(self, table: str, schema: str | None) -> list[str] | None:
        """Column names of *table*; of every copy when no *schema* picks one.

        Which copy an unqualified name reads is the database's search path to
        decide, so none is guessed here: a column any copy has is known, and
        :func:`tables_not_shown` asks for the schema where it matters.
        """
        owners = self._owners(table, schema)
        if not owners:
            return None
        return list(dict.fromkeys(c for o in owners for c in _table_columns(o, table)))

    def schemas_with(self, table: str) -> list[str]:
        return [
            key
            for key, candidate in self._schemas.items()
            if key and candidate.table_exists(table)
        ]


def _table_columns(owner: Any, table: str) -> list[str]:
    columns_map = getattr(owner, "columns_map", None)
    if columns_map is not None:
        return list(columns_map.get(table.lower(), {}))
    node = owner.get_table_node(table)
    return list(owner.tables_to_columns.get(node, []))

    def schemas_with(self, table: str) -> list[str]:
        return [
            key
            for key, candidate in self._schemas.items()
            if key and candidate.table_exists(table)
        ]


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


def catalog_table(name: str, schema: str | None, catalog: CatalogNames) -> str:
    """*name*, or the catalog's wildcard table when *name* reads some of its shards.

    BigQuery reads ``events_20210131`` and ``events_2021*`` from the same
    shards as ``events_*``, the one table the catalog holds for them; the
    query still runs as written.
    """
    if catalog.columns(name, schema) is not None:
        return name
    wildcard = wildcard_for(name)
    if wildcard and catalog.columns(wildcard, schema) is not None:
        return wildcard
    return name


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
        name = catalog_table(name, table.db or None, catalog)
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


def _function_name(func: exp.Func) -> str:
    name = func.name if isinstance(func, exp.Anonymous) else func.sql_name()
    return (name or "").lower()


def _is_forbidden_function(name: str) -> bool:
    if name in _ALLOWED_PG_FUNCTIONS:
        return False
    return name in _FORBIDDEN_FUNCTIONS or name.startswith(_FORBIDDEN_FUNCTION_PREFIXES)


def find_side_effects(statement: exp.Expression) -> list[str]:
    """Writes, locks and side-effecting calls anywhere in *statement*.

    A SELECT root is not enough to make a query read-only, so the whole tree is
    walked. This is the first line only: the database also runs agent SQL in a
    read-only transaction where it can (see ``execute_chat_sql``).
    """
    problems = [
        f"Not allowed in a read query: {type(node).__name__.upper()}"
        for node in statement.find_all(*_WRITING_NODES)
    ]
    problems += [
        f"Not allowed in a read query: function {name}()"
        for func in statement.find_all(exp.Func)
        if _is_forbidden_function(name := _function_name(func))
    ]
    return list(dict.fromkeys(problems))[:_MAX_PROBLEMS]


def check_sql(sql: str, dialect: str | None, catalog: CatalogNames) -> CheckResult:
    """Parse *sql*, refuse anything but one side-effect-free read, find unknown names."""
    try:
        statements = [s for s in sqlglot.parse(sql, read=dialect or None) if s]
    except sqlglot.errors.ParseError as error:
        return CheckResult([f"SQL does not parse: {str(error).splitlines()[0]}"])
    if len(statements) != 1:
        return CheckResult(["Send exactly one SQL statement."])
    statement = statements[0]
    if not isinstance(statement, exp.Query):
        return CheckResult(["Only read queries (SELECT or WITH … SELECT) can run."])
    side_effects = find_side_effects(statement)
    if side_effects:
        return CheckResult(side_effects)
    return CheckResult(find_unknown_names(statement, catalog), statement)


def catalog_for(ctx: ToolContext, connector: SQLDatabase) -> CatalogNames:
    """The catalog of *connector*'s database, loaded once per episode.

    Scoped to one database because the schema map is keyed by schema name
    alone: two databases that both have ``public`` would otherwise merge into
    one entry, and a query could pass on a table only the other one has.
    """
    database_name = getattr(connector, "database_name", None)
    if database_name not in ctx.schemas:
        if not database_name:
            schemas = get_schemas_by_ids(fetch_all_schema_ids())
        elif schema_ids := fetch_schema_ids_for_database(database_name):
            schemas = get_schemas_by_ids(schema_ids)
        else:
            # A named database with no schemas has no tables. It must not borrow
            # every other database's: the loader reads an empty id list as "all".
            schemas = {}
        ctx.schemas[database_name] = schemas
    return SchemaCatalog(ctx.schemas[database_name])


def tables_not_shown(
    statement: exp.Expression, ctx: ToolContext, catalog: CatalogNames
) -> list[str]:
    """Catalog tables *statement* reads that no ``retrieve`` result named.

    The prompt tells the model to use only what ``retrieve`` showed, and this
    is what holds it to that: a real table it never retrieved comes without
    its description, its term or the SIMILAR line that says it is one of two
    copies. A name is matched as the table it resolves to, so neither side can
    stand in for a same-named table in another schema.
    """
    derived = _derived_names(statement)
    problems = []
    for table in statement.find_all(exp.Table):
        if not table.name or table.name.lower() in derived:
            continue
        name = catalog_table(table.name, table.db or None, catalog)
        owners = catalog.schemas_with(name)
        if ctx.was_shown(table.db or None, name, owners):
            continue
        if not table.db and len(owners) > 1:
            problems.append(
                f"Ambiguous: {table.name} is in {', '.join(sorted(owners))}. "
                "Name its schema."
            )
            continue
        problems.append(
            f"Not in view: {'.'.join(p for p in (table.db, table.name) if p)}. "
            "Retrieve it (its exact name works) before using it."
        )
    return list(dict.fromkeys(problems))[:_MAX_PROBLEMS]


def bound_catalog(connector: SQLDatabase) -> str | None:
    """The catalog *connector* is bound to; ``None`` on a two-level engine.

    Read from the connector's own :meth:`qualify`, which is the one place that
    knows its naming rule. ``database_name`` cannot stand in for it: that is
    the catalog on Trino, Databricks and Kyuubi but the database everywhere else.
    """
    qualify = getattr(connector, "qualify", None)
    if qualify is None:
        return None
    try:
        reference = qualify("probe_schema", "probe_table")
        dialect = getattr(connector, "dialect", None) or None
        return exp.to_table(reference, dialect=dialect).catalog or None
    except (ValueError, sqlglot.errors.ParseError):
        return None


def foreign_catalogs(statement: exp.Expression, catalog: str) -> list[str]:
    """Tables *statement* names under a catalog other than *catalog*.

    The connection is bound to one catalog, so ``other.sales.orders`` would read
    a catalog the agent was never given, wherever the credential can reach.
    """
    problems = [
        f"Not allowed: catalog {table.catalog!r} in {table.catalog}.{table.db}."
        f"{table.name}. This database's catalog is {catalog!r}; leave the catalog "
        "out or use it."
        for table in statement.find_all(exp.Table)
        if table.catalog and table.catalog.lower() != catalog.lower()
    ]
    return list(dict.fromkeys(problems))[:_MAX_PROBLEMS]


def check_query(ctx: ToolContext, connector: SQLDatabase, sql: str) -> CheckResult:
    """:func:`check_sql` against *connector*'s catalog, then against what is in view."""
    catalog = catalog_for(ctx, connector)
    check = check_sql(sql, getattr(connector, "dialect", None), catalog)
    if check.problems or check.statement is None:
        return check
    if bound := bound_catalog(connector):
        if problems := foreign_catalogs(check.statement, bound):
            return CheckResult(problems, check.statement)
    return CheckResult(tables_not_shown(check.statement, ctx, catalog), check.statement)


def connector_for(ctx: ToolContext, sql: str | None = None) -> SQLDatabase | None:
    """The database the episode's SQL runs on.

    The target database when the caller named one; otherwise the only one
    configured, or the connector that owns the tables ``retrieve`` has shown.
    When those span several databases, *sql* decides: the one database
    holding every table it reads.
    """
    if ctx.database_name:
        for connector in ctx.connectors:
            if getattr(connector, "database_name", None) == ctx.database_name:
                return connector
    if len(ctx.connectors) == 1:
        return ctx.connectors[0]
    return resolve_connector_for_sql(
        list(ctx.seen_tables.values()), ctx.connectors, sql
    )


def no_database_error(ctx: ToolContext, sql: str) -> str:
    """Why :func:`connector_for` found no database for *sql*."""
    candidates = databases_for_sql(list(ctx.seen_tables.values()), ctx.connectors, sql)
    if len(candidates) > 1:
        return (
            "Error: the tables this query reads are in more than one database "
            f"({', '.join(candidates)}), and the query cannot say which. Use "
            "tables only one of them has."
        )
    shown = sorted(
        {
            str(name)
            for t in ctx.seen_tables.values()
            if (name := t.get("database_name"))
        }
    )
    if len(shown) > 1:
        return (
            "Error: no single shown database holds every table this query reads. "
            f"Shown databases: {', '.join(shown)}. Query one database at a time."
        )
    return "Error: no database to run on. Retrieve the tables you need first."


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
    connector = connector_for(ctx, sql)
    if connector is None:
        return no_database_error(ctx, sql)
    dialect = getattr(connector, "dialect", None)
    check = check_query(ctx, connector, sql)
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
