# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reads and writes for ambiguity resolution.

Ambiguity resolution writes ``column_attribute.ambiguity_resolver``: when to
pick an attribute over others of the same name in other tables of its
database. Text-to-SQL reads it back for the columns of its candidate tables.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import select, update

from auto_ontology.dal import schema as s
from auto_ontology.dal.session import store, write_transaction
from auto_ontology.dal.sql_fragments import table_description_expr
from auto_ontology.semantic.constants import SEMANTIC_SOURCE


def _owned_attributes_join() -> Any:
    """ColumnAttribute → its owning column → table → schema → database."""
    return (
        s.column__has_attribute.join(
            s.column_attribute,
            s.column_attribute.c.id == s.column__has_attribute.c.attribute_id,
        )
        .join(
            s.catalog_column,
            s.catalog_column.c.id == s.column__has_attribute.c.column_id,
        )
        .join(s.catalog_table, s.catalog_table.c.id == s.catalog_column.c.table_id)
        .join(s.catalog_schema, s.catalog_schema.c.id == s.catalog_table.c.schema_id)
        .join(
            s.catalog_database,
            s.catalog_database.c.id == s.catalog_schema.c.database_id,
        )
    )


def fetch_attributes_for_ambiguity(database_name: str) -> list[dict[str, Any]]:
    """Every semantic ColumnAttribute of *database_name* with its owning column.

    One row per attribute. When an attribute owns several columns, the lowest
    column id wins, as in
    :func:`auto_ontology.dal.descriptions.fetch_table_for_enrichment`.
    """
    rows = store().query_read(
        select(
            s.column_attribute.c.id,
            s.column_attribute.c.name,
            s.column_attribute.c.description,
            s.catalog_column.c.id.label("column_id"),
            s.catalog_column.c.name.label("column_name"),
            s.catalog_column.c.description.label("column_description"),
            s.catalog_column.c.description_certified.label("column_certified"),
            s.catalog_column.c.value_description,
            s.catalog_column.c.sample_values,
            s.catalog_table.c.id.label("table_id"),
            s.catalog_table.c.name.label("table_name"),
            table_description_expr().label("table_description"),
            s.catalog_schema.c.name.label("schema_name"),
        )
        .select_from(_owned_attributes_join())
        .where(
            s.catalog_database.c.name == database_name,
            s.column_attribute.c.source == SEMANTIC_SOURCE,
        )
        .order_by(s.column_attribute.c.id, s.catalog_column.c.id)
    )
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        by_id.setdefault(str(row["id"]), {**dict(row), "id": str(row["id"])})
    return list(by_id.values())


def write_ambiguity_resolvers(database_name: str, resolvers: dict[str, str]) -> int:
    """Set each attribute's resolver and reset every other one in the database.

    *resolvers* maps attribute id to text. Every other attribute of
    *database_name* gets ``''``, so a name that is no longer ambiguous keeps no
    stale note. Returns the number of non-empty resolvers written.
    """
    in_database = (
        select(s.column__has_attribute.c.attribute_id)
        .select_from(_owned_attributes_join())
        .where(s.catalog_database.c.name == database_name)
    )
    written = 0
    with write_transaction():
        store().query_write(
            update(s.column_attribute)
            .where(
                s.column_attribute.c.id.in_(in_database),
                s.column_attribute.c.ambiguity_resolver != "",
            )
            .values(ambiguity_resolver="")
        )
        for attribute_id, text in resolvers.items():
            text = (text or "").strip()
            if not text:
                continue
            rows = store().query_write(
                update(s.column_attribute)
                .where(s.column_attribute.c.id == attribute_id)
                .values(ambiguity_resolver=text)
                .returning(s.column_attribute.c.id)
            )
            written += len(rows)
    return written


def fetch_ambiguity_resolvers(table_ids: list[str]) -> dict[tuple[str, str], str]:
    """Non-empty resolvers of the columns of *table_ids*, keyed by (table id, column)."""
    if not table_ids:
        return {}
    rows = store().query_read(
        select(
            s.catalog_column.c.table_id,
            s.catalog_column.c.name,
            s.column_attribute.c.ambiguity_resolver,
        )
        .select_from(
            s.column__has_attribute.join(
                s.column_attribute,
                s.column_attribute.c.id == s.column__has_attribute.c.attribute_id,
            ).join(
                s.catalog_column,
                s.catalog_column.c.id == s.column__has_attribute.c.column_id,
            )
        )
        .where(
            s.catalog_column.c.table_id.in_(table_ids),
            s.column_attribute.c.ambiguity_resolver != "",
        )
        .order_by(s.catalog_column.c.id, s.column_attribute.c.id)
    )
    resolvers: dict[tuple[str, str], str] = {}
    for row in rows:
        resolvers.setdefault(
            (str(row["table_id"]), row["name"]), row["ambiguity_resolver"]
        )
    return resolvers
