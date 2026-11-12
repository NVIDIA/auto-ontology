# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reusable SQL expressions shared across the catalog and semantic reads.

Small, but disproportionately load-bearing: the description fallback here
decides what the UI and the SQL generator see as a column's description.

A paging note that belongs to callers rather than here: without a total
``ORDER BY``, consecutive pages can both repeat and drop rows. Every paged
query must order by something unique.
"""

from __future__ import annotations

from sqlalchemy import ColumnElement, func, select

from auto_ontology.dal import schema as s


def name_contains(column: ColumnElement, search: str) -> ColumnElement:
    """Case-insensitive substring match on *column*, wildcards escaped.

    The caller means a literal substring. Unescaped, ``_`` matches any
    character (so ``customer_id`` also finds ``customerXid``) and a lone ``%``
    returns every row.
    """
    needle = (
        search.strip().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    )
    return column.ilike(f"%{needle}%", escape="\\")


def _non_blank(column: ColumnElement) -> ColumnElement:
    """The column's value, or ``NULL`` when it is absent or whitespace.

    Blank is treated as missing throughout the fallback, so an empty string
    stored on a column does not mask a real description on the attribute behind
    it. Dropping the guard would make the fallback silently useless for every
    row someone had saved and cleared.
    """
    return func.nullif(func.trim(func.coalesce(column, "")), "")


def _attribute_description(column_id: ColumnElement, link_table) -> ColumnElement:
    """First non-blank ColumnAttribute description reachable via *link_table*."""
    attribute = s.column_attribute
    return (
        select(attribute.c.description)
        .select_from(
            link_table.join(attribute, attribute.c.id == link_table.c.attribute_id)
        )
        .where(
            link_table.c.column_id == column_id,
            _non_blank(attribute.c.description).isnot(None),
        )
        # Ordering by id keeps the answer the same between calls. Which
        # attribute wins when a column has several is still arbitrary, but it
        # is at least stably arbitrary.
        .order_by(attribute.c.id)
        .limit(1)
        .scalar_subquery()
    )


def column_description_expr(column=s.catalog_column) -> ColumnElement:
    """A column's description, falling back through its attributes.

    Prefers the column's own description; then a ``HAS_ATTRIBUTE`` attribute's;
    then a ``SEMANTIC_FK`` one's. The order matters and is not alphabetical: an
    attribute the column *is* an instance of describes it better than one it
    merely *references*.

    Takes the **table or alias** the caller is selecting from, not an id. Both
    are needed — ``description`` for the first branch and ``id`` to correlate
    the subqueries — and reading the description back out of a second
    ``SELECT`` on ``catalog_column`` would correlate the table with itself,
    turning the ``WHERE`` into a tautology that matches every row. Postgres
    catches that as a cardinality violation rather than returning a wrong
    answer, which is the good outcome; passing the alias avoids it entirely.
    """
    return func.coalesce(
        _non_blank(column.c.description),
        _attribute_description(column.c.id, s.column__has_attribute),
        _attribute_description(column.c.id, s.column__semantic_fk),
    )


def table_description_expr(table=s.catalog_table) -> ColumnElement:
    """A table's description, falling back to the Term it represents.

    Takes the table or alias, for the same reason as
    :func:`column_description_expr`.
    """
    table_id = table.c.id
    from_term = (
        select(s.term.c.description)
        .select_from(s.table__term.join(s.term, s.term.c.id == s.table__term.c.term_id))
        .where(
            s.table__term.c.table_id == table_id,
            _non_blank(s.term.c.description).isnot(None),
        )
        .order_by(s.term.c.id)
        .limit(1)
        .scalar_subquery()
    )
    return func.coalesce(_non_blank(table.c.description), from_term)


def _qualified_column(target) -> ColumnElement:
    return func.concat_ws(
        ".", s.catalog_schema.c.name, s.catalog_table.c.name, target.c.name
    )


def column_reference_expr(column=s.catalog_column) -> ColumnElement:
    """``schema.table.column`` a column's foreign key lands on, or ``NULL``.

    A declared foreign key first, then a ``SEMANTIC_FK``, whose target is the
    column that *owns* the referenced attribute. Takes the table or alias, for
    the same reason as :func:`column_description_expr`.
    """
    target = s.catalog_column.alias("reference_target")

    def _onto(link) -> ColumnElement:
        return (
            select(_qualified_column(target))
            .select_from(
                link.join(target, target.c.id == link.c.target_id)
                .join(s.catalog_table, s.catalog_table.c.id == target.c.table_id)
                .join(
                    s.catalog_schema,
                    s.catalog_schema.c.id == s.catalog_table.c.schema_id,
                )
            )
            .where(link.c.source_id == column.c.id)
            .order_by(target.c.id)
            .limit(1)
            .scalar_subquery()
        )

    declared = select(
        s.column__foreign_key.c.source_column_id.label("source_id"),
        s.column__foreign_key.c.target_column_id.label("target_id"),
    ).subquery("declared_reference")
    semantic = (
        select(
            s.column__semantic_fk.c.column_id.label("source_id"),
            s.column__has_attribute.c.column_id.label("target_id"),
        )
        .select_from(
            s.column__semantic_fk.join(
                s.column__has_attribute,
                s.column__has_attribute.c.attribute_id
                == s.column__semantic_fk.c.attribute_id,
            )
        )
        .subquery("semantic_reference")
    )
    return func.coalesce(_onto(declared), _onto(semantic))
