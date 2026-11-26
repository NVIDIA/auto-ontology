# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Find column references in SQL that the catalog says do not exist.

:func:`~auto_ontology.catalog.parsers.sqlglot_extractor.extract_tables_and_columns`
attributes the columns it can and drops the rest, which is right for mining a
query log but lets a saved definition name a column its table never had.  This
check reports those columns instead, statically: the SQL is never executed.

It only reports what it can prove.  A column is checked against a scope only
when every source that scope (and the scopes enclosing it) reads has a known
column list: a catalog table with ingested columns, or a CTE / derived table
whose outputs are explicit.  A table missing from the catalog, a table function
or a ``SELECT *`` source makes the scope unverifiable, and nothing in it is
reported.
"""

from __future__ import annotations

import sqlglot
from sqlglot import exp
from sqlglot.errors import OptimizeError, ParseError, TokenError
from sqlglot.optimizer.scope import Scope, traverse_scope

_CatalogColumns = dict[tuple[str, str], set[str]]


def _catalog_columns(all_schemas: dict) -> _CatalogColumns:
    """Return ``{(schema, table): {column, ...}}``, all lower-cased."""
    columns: _CatalogColumns = {}
    for schema_name, schema in all_schemas.items():
        df = getattr(schema, "columns_df", None)
        if df is None or df.empty:
            continue
        skey = str(schema_name).lower()
        for table_name, column_name in zip(df["table_name"], df["column_name"]):
            key = (skey, str(table_name).lower())
            columns.setdefault(key, set()).add(str(column_name).lower())
    return columns


def _table_columns(table: exp.Table, catalog: _CatalogColumns) -> set[str] | None:
    name = table.name.lower()
    if not name:
        return None
    db = table.db.lower() if table.db else None
    matches = [
        cols
        for (schema, tbl), cols in catalog.items()
        if tbl == name and (db is None or schema == db)
    ]
    if not matches:
        return None
    # A bare name present in several schemas may be any of them, so accept a
    # column found in any.
    return set().union(*matches)


def _has_star(query: exp.Query) -> bool:
    return any(
        isinstance(select, exp.Star)
        or (isinstance(select, exp.Column) and isinstance(select.this, exp.Star))
        for select in query.selects
    )


def _source_columns(source: object, catalog: _CatalogColumns) -> set[str] | None:
    """Return the column names *source* exposes, or ``None`` when unknown."""
    if isinstance(source, exp.Table):
        return _table_columns(source, catalog)
    if not isinstance(source, Scope):
        return None
    query = source.expression
    alias = query.parent.args.get("alias") if query.parent else None
    if isinstance(alias, exp.TableAlias) and alias.columns:
        return {column.name.lower() for column in alias.columns}
    if not isinstance(query, exp.Query) or _has_star(query):
        return None
    return {name.lower() for name in query.named_selects}


def _scope_sources(
    scope: Scope, catalog: _CatalogColumns
) -> dict[str, set[str] | None]:
    return {
        name.lower(): _source_columns(source, catalog)
        for name, (_node, source) in scope.selected_sources.items()
    }


def _visible_sources(scope: Scope, catalog: _CatalogColumns) -> list[set[str] | None]:
    """Sources a bare name in *scope* may come from, its correlated outers included."""
    visible: list[set[str] | None] = []
    current: Scope | None = scope
    while current is not None:
        visible.extend(_scope_sources(current, catalog).values())
        current = current.parent
    return visible


def _is_output_alias_reference(
    column: exp.Column, query: exp.Select, dialect: str | None
) -> bool:
    """Whether *column* occurs in a clause that may reference a SELECT alias."""
    alias_clauses: tuple[type[exp.Expression], ...] = (
        exp.Order,
        exp.Group,
        exp.Having,
        exp.Qualify,
    )
    if dialect and dialect.lower() == "sqlite":
        # SQLite extends standard alias visibility to WHERE and JOIN ... ON.
        alias_clauses += (exp.Where, exp.Join)
    current = column.parent
    while current is not None and current is not query:
        if isinstance(current, alias_clauses):
            return True
        current = current.parent
    return False


def find_unresolved_columns(
    sql: str, dialect: str | None, all_schemas: dict
) -> list[str]:
    """Return the column references in *sql* that the catalog rules out.

    Each entry is ``table.column`` for a qualified reference and ``column``
    for a bare one, in order of first appearance.  An empty list means none
    was proven missing, not that every column was verified.
    """
    catalog = _catalog_columns(all_schemas)
    if not catalog:
        return []
    try:
        statement = sqlglot.parse_one(sql, dialect=dialect or None)
        scopes = traverse_scope(statement)
    except (OptimizeError, ParseError, TokenError):
        return []

    unresolved: list[str] = []
    for scope in scopes:
        query = scope.expression
        if not isinstance(query, exp.Select):
            continue
        sources = _scope_sources(scope, catalog)
        visible = _visible_sources(scope, catalog)
        bare_verifiable = bool(visible) and all(cols is not None for cols in visible)
        aliases = {
            select.alias.lower()
            for select in query.selects
            if isinstance(select, exp.Alias)
        }
        for column in scope.columns:
            # sqlglot also lists a subquery's unqualified columns on its outer
            # scopes; each is checked once, by the SELECT it is written in.
            if column.find_ancestor(exp.Select) is not query:
                continue
            if isinstance(column.this, exp.Star) or column.args.get("db"):
                continue
            name = column.name.lower()
            table = column.table.lower()
            if table:
                known = sources.get(table)
                if known is None or name in known:
                    continue
                reference = f"{table}.{name}"
            else:
                if not bare_verifiable or (
                    name in aliases
                    and _is_output_alias_reference(column, query, dialect)
                ):
                    continue
                if any(name in cols for cols in visible if cols is not None):
                    continue
                reference = name
            if reference not in unresolved:
                unresolved.append(reference)
    return unresolved
