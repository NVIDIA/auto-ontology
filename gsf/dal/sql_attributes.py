# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SqlAttribute reads and writes.

Orchestration — SQL validation, connector resolution, VDB lifecycle — stays in
``gsf/server/sql_attributes/service.py``. This module only touches the store.

Two rules run through every read here and are the easiest things to lose:

* **A SqlAttribute is scoped by the tables its own SQL references**, not by its
  parent Term's tables. The two genuinely differ, and using the Term's would
  show a viewer an attribute querying data they cannot see.
* **The scope check is all-or-nothing.** One out-of-zone table hides the whole
  attribute. Partial visibility would mean rendering SQL a viewer is not
  allowed to run.
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import String, delete, func, literal, or_, select, update
from sqlalchemy.dialects.postgresql import insert

from gsf.dal import schema as s
from gsf.dal.session import store, write_transaction
from gsf.dal.sql_fragments import column_description_expr
from gsf.dal.users import resolve_accessible_catalog_ids
from gsf.server.sql_utils import SqlParseError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Domain errors
# ---------------------------------------------------------------------------


class SqlAttributeNameConflict(Exception):
    """Raised when a write would collide with another SqlAttribute name."""


class SqlAttributeExpressionConflict(Exception):
    """Raised when a Term already has another SqlAttribute with this SQL."""


SqlAttributeSqlError = SqlParseError


# ---------------------------------------------------------------------------
# Shared traversal
# ---------------------------------------------------------------------------


def _out_of_zone(table_ids: list[str]):
    """True for an attribute whose SQL touches a table outside *table_ids*.

    The all-or-nothing check, phrased in the negative on purpose: not "does it
    touch an allowed table" but "does it touch a disallowed one". The difference
    matters for an attribute joining an in-zone table to an out-of-zone one,
    which the positive form would happily show.
    """
    return (
        select(literal(1))
        .select_from(
            s.sql_attribute_sql.join(
                s.sql_query_table,
                s.sql_query_table.c.sql_query_id == s.sql_attribute_sql.c.sql_query_id,
            )
        )
        .where(
            s.sql_attribute_sql.c.attribute_id == s.sql_attribute.c.id,
            s.sql_query_table.c.table_id.notin_(table_ids),
        )
        .correlate(s.sql_attribute)
        .exists()
    )


def _zone_scope(
    zone_ids: list[str] | None,
    data_ids_by_zone: dict[str, set[str]] | None = None,
):
    """The zone predicate for a SqlAttribute read, or ``None`` for unscoped."""
    if zone_ids is None:
        return None
    resolved = resolve_accessible_catalog_ids(zone_ids, data_ids_by_zone)
    return ~_out_of_zone(list(resolved["table_ids"]))


#: The SQL text of an attribute, taken from its lowest-id statement.
#:
#: An attribute can be linked to several statements, so this collapses them to
#: one. Without the collapse the rows outnumber the count the pager is given,
#: and the last attributes of a term become unreachable — the page ends before
#: they do.
_SQL_TEXT = (
    select(s.sql_query.c.sql_full_query)
    .select_from(
        s.sql_attribute_sql.join(
            s.sql_query, s.sql_query.c.id == s.sql_attribute_sql.c.sql_query_id
        )
    )
    .where(s.sql_attribute_sql.c.attribute_id == s.sql_attribute.c.id)
    .order_by(s.sql_query.c.id)
    .limit(1)
    .correlate(s.sql_attribute)
    .scalar_subquery()
)


def _has_sql():
    """An attribute with no statement is invisible to every read here.

    The statement join is inner, so such an attribute never appears in a list —
    and the counts repeat the condition, so a badge cannot promise more rows
    than the list can show.
    """
    return (
        select(literal(1))
        .where(s.sql_attribute_sql.c.attribute_id == s.sql_attribute.c.id)
        .correlate(s.sql_attribute)
        .exists()
    )


def _attribute_select():
    """The shared projection: attribute, its Term, and its SQL text."""
    return (
        select(
            s.sql_attribute.c.id,
            s.sql_attribute.c.name,
            s.sql_attribute.c.description,
            s.sql_attribute.c.description_suggestion,
            s.sql_attribute.c.expression,
            s.sql_attribute.c.source,
            _SQL_TEXT.label("sql"),
            s.sql_attribute.c.certified,
            s.term.c.id.label("term_id"),
            s.term.c.name.label("term_name"),
        )
        .select_from(
            s.sql_attribute.join(
                s.sql_attribute_term,
                s.sql_attribute_term.c.attribute_id == s.sql_attribute.c.id,
            ).join(s.term, s.term.c.id == s.sql_attribute_term.c.term_id)
        )
        .where(_has_sql())
    )


def _query_sql_attributes(
    *,
    attr_id: str | None = None,
    term_id: str | None = None,
    zone_ids: list[str] | None = None,
    order_by: Any = None,
    skip: int = 0,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Every SqlAttribute read goes through here.

    Anchor on *attr_id* or *term_id* — mutually exclusive — or neither for all
    of them. *skip* and *limit* page *order_by*; only pass them with an
    *order_by* that fully determines row order, or a page's contents shift
    between requests.
    """
    statement = _attribute_select()
    if attr_id is not None:
        statement = statement.where(s.sql_attribute.c.id == attr_id)
    if term_id is not None:
        statement = statement.where(s.term.c.id == term_id)

    scope = _zone_scope(zone_ids)
    if scope is not None:
        statement = statement.where(scope)
    if order_by is not None:
        statement = statement.order_by(*order_by)
    if skip:
        statement = statement.offset(skip)
    if limit is not None:
        statement = statement.limit(limit)

    return [dict(r) for r in store().query_read(statement)]


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


def list_sql_attributes() -> list[dict[str, Any]]:
    """Every SqlAttribute, with its Term and SQL text."""
    return _query_sql_attributes(order_by=(s.sql_attribute.c.name,))


def fetch_sql_attribute_counts(
    zone_ids: list[str] | None = None,
    data_ids_by_zone: dict[str, set[str]] | None = None,
    term_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """``[{term_id, count}]``, under the same zone scoping as every read.

    *term_ids* narrows the scan — the paged Terms list passes the ids on the
    page it is about to render, so the response does not carry counts for the
    rest of the glossary. ``None`` counts every term.

    Terms with no SqlAttributes are omitted, and the ``_has_sql`` condition is
    repeated deliberately: a count including statement-less attributes would
    render a badge larger than the list behind it.
    """
    statement = (
        select(
            s.term.c.id.label("term_id"),
            func.count(func.distinct(s.sql_attribute.c.id)).label("count"),
        )
        .select_from(
            s.sql_attribute.join(
                s.sql_attribute_term,
                s.sql_attribute_term.c.attribute_id == s.sql_attribute.c.id,
            ).join(s.term, s.term.c.id == s.sql_attribute_term.c.term_id)
        )
        .where(_has_sql())
        .group_by(s.term.c.id)
    )
    scope = _zone_scope(zone_ids, data_ids_by_zone)
    if scope is not None:
        statement = statement.where(scope)
    if term_ids is not None:
        statement = statement.where(s.term.c.id.in_(list(term_ids)))

    return [
        {"term_id": r["term_id"], "count": int(r["count"])}
        for r in store().query_read(statement)
    ]


def get_full_sql_attribute_by_id(
    attr_id: str,
    zone_ids: list[str] | None = None,
) -> dict[str, Any] | None:
    """One SqlAttribute with its Term, SQL text, and the zones covering it.

    The zone chips come from the tables the attribute's **own SQL** references,
    reached either directly or through their schema or database — matching how
    the scope check resolves them, and not the parent Term's tables.

    With *zone_ids* supplied the attribute must pass the scope check (``None``
    otherwise, so a viewer cannot read an out-of-zone attribute) and disabled
    zones are dropped from the chips entirely. Pass ``None`` to skip both, which
    is how an admin sees a disabled zone at all — with ``enabled: False`` so the
    UI can render it as retired rather than active.
    """
    rows = _query_sql_attributes(attr_id=attr_id, zone_ids=zone_ids)
    if not rows:
        return None
    result = rows[0]

    # Zone -> the tables it covers, at any of the three grain levels a zone
    # target can name -- table, schema or database -- as three explicit
    # branches over the containment keys.
    covered = or_(
        s.zone_target.c.table_id == s.sql_query_table.c.table_id,
        s.zone_target.c.schema_id == s.catalog_table.c.schema_id,
        s.zone_target.c.database_id == s.catalog_schema.c.database_id,
    )
    statement = (
        select(
            s.zone.c.id,
            s.zone.c.name,
            s.zone.c.color,
            s.zone.c.enabled,
        )
        .select_from(
            s.sql_attribute_sql.join(
                s.sql_query_table,
                s.sql_query_table.c.sql_query_id == s.sql_attribute_sql.c.sql_query_id,
            )
            .join(s.catalog_table, s.catalog_table.c.id == s.sql_query_table.c.table_id)
            .join(
                s.catalog_schema, s.catalog_schema.c.id == s.catalog_table.c.schema_id
            )
            .join(s.zone_target, covered)
            .join(s.zone, s.zone.c.id == s.zone_target.c.zone_id)
        )
        .where(s.sql_attribute_sql.c.attribute_id == attr_id)
        .distinct()
        .order_by(s.zone.c.name)
    )
    if zone_ids is not None:
        statement = statement.where(
            s.zone.c.id.in_(list(zone_ids)), s.zone.c.enabled.is_(True)
        )

    result["zones"] = [dict(r) for r in store().query_read(statement)]
    return result


def fetch_sql_attributes_by_term_id(
    term_id: str,
    zone_ids: list[str] | None = None,
    *,
    skip: int = 0,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """One Term's SqlAttributes, each scoped independently.

    An attribute of an otherwise-visible term is still hidden when its own SQL
    touches a table outside *zone_ids*.

    Ordered by name then id — the id breaks ties between same-named attributes,
    without which a page boundary could repeat one and skip another. Pair with
    ``count_sql_attributes_by_term_id`` for the total.
    """
    return _query_sql_attributes(
        term_id=term_id,
        zone_ids=zone_ids,
        order_by=(s.sql_attribute.c.name, s.sql_attribute.c.id),
        skip=skip,
        limit=limit,
    )


def count_sql_attributes_by_term_id(
    term_id: str,
    zone_ids: list[str] | None = None,
) -> int:
    """The unpaged size of :func:`fetch_sql_attributes_by_term_id`.

    Applies the same ``_has_sql`` condition for the same reason: a total that
    counted statement-less attributes would exceed the rows the pager can reach.
    """
    statement = (
        select(func.count(func.distinct(s.sql_attribute.c.id)).label("total"))
        .select_from(
            s.sql_attribute.join(
                s.sql_attribute_term,
                s.sql_attribute_term.c.attribute_id == s.sql_attribute.c.id,
            )
        )
        .where(s.sql_attribute_term.c.term_id == term_id, _has_sql())
    )
    scope = _zone_scope(zone_ids)
    if scope is not None:
        statement = statement.where(scope)

    rows = store().query_read(statement)
    return int(rows[0]["total"]) if rows else 0


def find_attr_by_name(name: str, exclude_id: str | None) -> dict[str, str] | None:
    """Another SqlAttribute already using *name*, or ``None``.

    *exclude_id* is the attribute being edited — without it, renaming an
    attribute to its own name reports a conflict with itself.
    """
    statement = select(s.sql_attribute.c.id, s.sql_attribute.c.name).where(
        s.sql_attribute.c.name == name
    )
    if exclude_id is not None:
        statement = statement.where(s.sql_attribute.c.id != exclude_id)
    rows = store().query_read(statement.order_by(s.sql_attribute.c.id).limit(1))
    return {"id": rows[0]["id"], "name": rows[0]["name"]} if rows else None


def find_attr_by_expression(
    *,
    term_id: str,
    expression: str,
    exclude_id: str | None,
) -> dict[str, str] | None:
    """A same-term SqlAttribute with equivalent SQL, or ``None``.

    Equivalence ignores case and collapses whitespace, matching the legacy
    snippet validation. Compared in Python rather than SQL, as before: the
    normalisation is ``" ".join(x.split())``, which collapses *runs* of any
    whitespace to one space, and no SQL expression reproduces that — ``regexp_replace``
    on ``\\s+`` comes close but differs on the Unicode whitespace ``str.split``
    accepts. Matching the old behaviour exactly is worth one full scan of a
    single term's attributes.
    """
    normalized = " ".join(expression.split()).lower()
    statement = (
        select(
            s.sql_attribute.c.id,
            s.sql_attribute.c.name,
            s.sql_attribute.c.expression,
        )
        .select_from(
            s.sql_attribute.join(
                s.sql_attribute_term,
                s.sql_attribute_term.c.attribute_id == s.sql_attribute.c.id,
            )
        )
        .where(s.sql_attribute_term.c.term_id == term_id)
    )
    if exclude_id is not None:
        statement = statement.where(s.sql_attribute.c.id != exclude_id)

    for row in store().query_read(statement):
        candidate = row["expression"]
        if not isinstance(candidate, str):
            continue
        if " ".join(candidate.split()).lower() == normalized:
            return {"id": row["id"], "name": row["name"]}
    return None


def get_sql_attribute_by_id(attr_id: str) -> str | None:
    """The attribute's id if it exists, else ``None`` — an existence check."""
    rows = store().query_read(
        select(s.sql_attribute.c.id).where(s.sql_attribute.c.id == attr_id).limit(1)
    )
    return rows[0]["id"] if rows else None


# ---------------------------------------------------------------------------
# Writes
# ---------------------------------------------------------------------------


def detach_existing_sql_edges(attr_id: str) -> None:
    """Unlink every statement from the attribute, leaving the statements alone.

    The Sql rows survive on purpose — they are shared with query history and
    with other attributes, so deleting them here would take out rows this
    attribute does not own.
    """
    store().query_write(
        delete(s.sql_attribute_sql).where(s.sql_attribute_sql.c.attribute_id == attr_id)
    )


def link_to_term(attr_id: str, term_id: str) -> None:
    """Point the attribute at exactly one Term, replacing any previous link.

    Delete and insert in one transaction. Split across two autocommits there is
    a window in which the attribute belongs to no Term at all, and every read
    that runs in it sees an unlinked attribute; if the insert then fails, the
    window never closes.
    """
    with write_transaction():
        store().query_write(
            delete(s.sql_attribute_term).where(
                s.sql_attribute_term.c.attribute_id == attr_id,
                s.sql_attribute_term.c.term_id != term_id,
            )
        )
        store().query_write(
            insert(s.sql_attribute_term)
            .values(attribute_id=attr_id, term_id=term_id)
            .on_conflict_do_nothing()
        )


def update_sql_attribute(
    attr_id: str,
    *,
    name: str | None = None,
    description: str | None = None,
    expression: str | None = None,
    source: str | None = None,
    certified: bool | None = None,
) -> None:
    """Set properties on an existing attribute; omitted fields are untouched."""
    store().query_write(
        update(s.sql_attribute)
        .where(s.sql_attribute.c.id == attr_id)
        .values(
            name=func.coalesce(name, s.sql_attribute.c.name),
            description=func.coalesce(description, s.sql_attribute.c.description),
            expression=func.coalesce(expression, s.sql_attribute.c.expression),
            # Typed explicitly: with every argument None, Postgres cannot infer
            # a type for the parameter inside coalesce and rejects the statement
            # outright. `source` is the one that hits it first because the
            # column is nullable with no default.
            source=func.coalesce(literal(source, String), s.sql_attribute.c.source),
            certified=func.coalesce(certified, s.sql_attribute.c.certified),
        )
    )


def set_sql_attribute_description_suggestion(
    attr_id: str,
    description_suggestion: str,
) -> None:
    """Cache an LLM description suggestion on the attribute."""
    store().query_write(
        update(s.sql_attribute)
        .where(s.sql_attribute.c.id == attr_id)
        .values(description_suggestion=description_suggestion)
    )


def clear_sql_attribute_description_suggestion(attr_id: str) -> None:
    """Drop the cached suggestion."""
    store().query_write(
        update(s.sql_attribute)
        .where(s.sql_attribute.c.id == attr_id)
        .values(description_suggestion=None)
    )


def clear_sql_attribute_description_suggestions_for_term(term_id: str) -> None:
    """Drop cached suggestions from every SqlAttribute of a Term."""
    store().query_write(
        update(s.sql_attribute)
        .where(
            s.sql_attribute.c.id.in_(
                select(s.sql_attribute_term.c.attribute_id).where(
                    s.sql_attribute_term.c.term_id == term_id
                )
            )
        )
        .values(description_suggestion=None)
    )


def delete_sql_attribute_node(attr_id: str) -> None:
    """Delete the attribute. Its links go with it, by ``ON DELETE CASCADE``.

    Cascading in the schema rather than deleting links here means a new link
    table cannot be added later and forgotten.
    """
    store().query_write(delete(s.sql_attribute).where(s.sql_attribute.c.id == attr_id))


# ---------------------------------------------------------------------------
# Retrieval-time helpers
# ---------------------------------------------------------------------------


def fetch_sql_attributes_with_sql(attr_ids: list[str]) -> list[dict[str, str]]:
    """``id, name, description, expression, sql, term_name`` per attribute.

    One row per attribute even when several statements or terms match — the
    first of each id wins, ordered so "first" is reproducible.

    Returns ``[]`` on failure — this feeds retrieval context, where losing the
    context beats losing the answer.
    """
    if not attr_ids:
        return []
    try:
        rows = store().query_read(
            select(
                s.sql_attribute.c.id.label("attr_id"),
                s.sql_attribute.c.name,
                s.sql_attribute.c.description,
                s.sql_attribute.c.expression,
                _SQL_TEXT.label("sql_text"),
                s.term.c.name.label("term_name"),
            )
            .select_from(
                s.sql_attribute.join(
                    s.sql_attribute_term,
                    s.sql_attribute_term.c.attribute_id == s.sql_attribute.c.id,
                ).join(s.term, s.term.c.id == s.sql_attribute_term.c.term_id)
            )
            .where(s.sql_attribute.c.id.in_(list(attr_ids)), _has_sql())
            .order_by(s.sql_attribute.c.id, s.term.c.id)
        )
    except Exception:
        logger.warning("fetch_sql_attributes_with_sql: query failed", exc_info=True)
        return []

    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        attr_id = row["attr_id"] or ""
        if attr_id in seen:
            continue
        seen.add(attr_id)
        result.append(
            {
                "id": attr_id,
                "name": row["name"] or "",
                "description": row["description"] or "",
                "expression": row["expression"] or "",
                "sql": row["sql_text"] or "",
                "term_name": row["term_name"] or "",
            }
        )
    return result


def fetch_tables_from_sql_attributes(
    attr_ids: list[str],
) -> list[dict[str, Any]]:
    """The tables these attributes' SQL actually references, with column summaries.

    Deduplicated by table id, since several attributes commonly reference the
    same table. Returns ``[]`` on failure, as above.
    """
    if not attr_ids:
        return []
    try:
        rows = store().query_read(
            select(
                s.catalog_table.c.id,
                s.catalog_table.c.name,
                s.catalog_table.c.description,
                s.catalog_table.c.pk,
                s.catalog_database.c.name.label("database_name"),
                s.catalog_schema.c.name.label("schema_name"),
                s.catalog_column.c.name.label("column_name"),
                s.catalog_column.c.data_type,
                # In the select list because `SELECT DISTINCT` requires every
                # ORDER BY key to be there. Dropped again when the rows are
                # folded into nested column lists below.
                s.catalog_column.c.ordinal_position,
                column_description_expr().label("column_description"),
            )
            .select_from(
                s.sql_attribute_sql.join(
                    s.sql_query_table,
                    s.sql_query_table.c.sql_query_id
                    == s.sql_attribute_sql.c.sql_query_id,
                )
                .join(
                    s.catalog_table,
                    s.catalog_table.c.id == s.sql_query_table.c.table_id,
                )
                .join(
                    s.catalog_schema,
                    s.catalog_schema.c.id == s.catalog_table.c.schema_id,
                )
                .join(
                    s.catalog_database,
                    s.catalog_database.c.id == s.catalog_schema.c.database_id,
                )
                .join(
                    s.catalog_column,
                    s.catalog_column.c.table_id == s.catalog_table.c.id,
                )
            )
            .where(s.sql_attribute_sql.c.attribute_id.in_(list(attr_ids)))
            .distinct()
            .order_by(s.catalog_table.c.id, s.catalog_column.c.ordinal_position)
        )
    except Exception:
        logger.warning("fetch_tables_from_sql_attributes: query failed", exc_info=True)
        return []

    tables: dict[str, dict[str, Any]] = {}
    for row in rows:
        table = tables.setdefault(
            row["id"],
            {
                "id": row["id"],
                "name": row["name"] or "",
                "description": row["description"] or "",
                "database_name": row["database_name"] or "",
                "schema_name": row["schema_name"] or "",
                "label": "Table",
                # The prediction graph keys its entities on this: a table
                # that arrives without a pk reaches KumoRFM with no identity,
                # which costs it every edge and makes it unusable in
                # `FOR EACH`. It has to survive every path to relevant_tables.
                "pk": row.get("pk") or [],
                "columns": [],
            },
        )
        if row["column_name"]:
            table["columns"].append(
                {
                    "name": row["column_name"],
                    "data_type": row["data_type"],
                    "description": row["column_description"],
                }
            )
    return list(tables.values())


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


def fetch_sql_attribute_docs(attr_id: str) -> list[dict[str, Any]]:
    """One attribute as embedding-ready docs, or ``[]`` if it has no statement.

    The text format is load-bearing — it is what gets embedded, so changing the
    separators or the order of the parts silently invalidates every stored
    vector for these attributes. Reproduced literally, including the blank
    description being omitted rather than rendered as an empty clause.
    """
    rows = store().query_read(
        select(
            s.sql_attribute.c.id,
            s.sql_attribute.c.name,
            s.sql_attribute.c.description,
            s.sql_attribute.c.source,
            s.term.c.name.label("term_name"),
            _SQL_TEXT.label("sql_text"),
        )
        .select_from(
            s.sql_attribute.join(
                s.sql_attribute_term,
                s.sql_attribute_term.c.attribute_id == s.sql_attribute.c.id,
            ).join(s.term, s.term.c.id == s.sql_attribute_term.c.term_id)
        )
        .where(s.sql_attribute.c.id == attr_id, _has_sql())
        .order_by(s.term.c.id)
    )
    docs: list[dict[str, Any]] = []
    for row in rows:
        description = row["description"]
        text = f"sql_attribute: {row['name']}"
        if description is not None and str(description).strip():
            text += f", description: {description}"
        if row["term_name"] is not None:
            text += f", term: {row['term_name']}"
        text += f", sql: {row['sql_text']}"
        docs.append(
            {
                "text": text,
                "name": row["name"],
                "label": "SqlAttribute",
                "id": row["id"],
                "source": row["source"] or "",
            }
        )
    return docs


#: The module's contract, spelled out.
#:
#: Needed because ``SqlAttributeSqlError`` is an *alias* for ``SqlParseError``,
#: defined elsewhere — so its ``__module__`` points there and the surface freeze
#: cannot see it, even though ``router.py`` catches it as
#: ``dal.SqlAttributeSqlError`` to turn a bad snippet into a 422. The selector
#: used to carry this list; with the selector gone it belongs here.
__all__ = [
    "SqlAttributeExpressionConflict",
    "SqlAttributeNameConflict",
    "SqlAttributeSqlError",
    "clear_sql_attribute_description_suggestion",
    "clear_sql_attribute_description_suggestions_for_term",
    "count_sql_attributes_by_term_id",
    "delete_sql_attribute_node",
    "detach_existing_sql_edges",
    "fetch_sql_attribute_counts",
    "fetch_sql_attribute_docs",
    "fetch_sql_attributes_by_term_id",
    "fetch_sql_attributes_with_sql",
    "fetch_tables_from_sql_attributes",
    "find_attr_by_expression",
    "find_attr_by_name",
    "get_full_sql_attribute_by_id",
    "get_sql_attribute_by_id",
    "link_to_term",
    "list_sql_attributes",
    "set_sql_attribute_description_suggestion",
    "update_sql_attribute",
]
