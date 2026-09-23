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
