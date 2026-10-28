# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reads and writes for description enrichment.

Enrichment rewrites ``description`` on tables, columns and the columns'
semantic attributes. The ingested text is kept in ``source_description`` the
first time a row is rewritten, and every later run reads it from there, so the
pass never feeds on its own output.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import case, func, literal, select, update

from auto_ontology.dal import schema as s
from auto_ontology.dal.session import store, write_transaction
from auto_ontology.dal.sql_fragments import (
    column_description_expr,
    table_description_expr,
)
from auto_ontology.semantic.constants import SEMANTIC_SOURCE


def fetch_tables_with_term(table_ids: list[str]) -> set[str]:
    """The subset of *table_ids* that a Term now represents."""
    if not table_ids:
        return set()
    rows = store().query_read(
        select(s.table__term.c.table_id)
        .where(s.table__term.c.table_id.in_(table_ids))
        .distinct()
    )
    return {str(row["table_id"]) for row in rows}


def fetch_table_for_enrichment(table_id: str) -> dict[str, Any] | None:
    """One table, its columns, and each column's semantic attribute.

    ``source_description`` on the table and on each column is the text
    enrichment should treat as authoritative: the preserved original when the
    row was enriched before, otherwise the current description.

    ``attribute_id`` is set only for columns the taxonomy pass gave a semantic
    ColumnAttribute — exactly its non-FK columns, since suggested FKs are never
    stored. When a column carries several, the lowest id wins, matching
    :func:`auto_ontology.dal.attributes.find_column_attribute_by_column_id`.
    """
    tables = store().query_read(
        select(
            s.catalog_table.c.id,
            s.catalog_table.c.name,
            s.catalog_table.c.description,
            s.catalog_table.c.source_description,
            s.catalog_table.c.description_certified,
            s.catalog_schema.c.name.label("schema_name"),
            s.catalog_database.c.name.label("database_name"),
        )
        .select_from(
            s.catalog_table.join(
                s.catalog_schema, s.catalog_schema.c.id == s.catalog_table.c.schema_id
            ).join(
                s.catalog_database,
                s.catalog_database.c.id == s.catalog_schema.c.database_id,
            )
        )
        .where(s.catalog_table.c.id == table_id)
    )
    if not tables:
        return None
    table = dict(tables[0])
    table["source_description"] = _source_text(table)

    columns = [
        dict(row)
        for row in store().query_read(
            select(
                s.catalog_column.c.id,
                s.catalog_column.c.name,
                s.catalog_column.c.data_type,
                s.catalog_column.c.description,
                s.catalog_column.c.source_description,
                s.catalog_column.c.description_certified,
                s.catalog_column.c.sample_values,
                s.catalog_column.c.format,
                s.catalog_column.c.is_unique,
                s.catalog_column.c.is_nullable,
            )
            .where(s.catalog_column.c.table_id == table_id)
            .order_by(s.catalog_column.c.ordinal_position, s.catalog_column.c.id)
        )
    ]
    attributes: dict[str, dict[str, Any]] = {}
    for row in store().query_read(
        select(
            s.column__has_attribute.c.column_id,
            s.column_attribute.c.id,
            s.column_attribute.c.certified,
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
            s.catalog_column.c.table_id == table_id,
            s.column_attribute.c.source == SEMANTIC_SOURCE,
        )
        .order_by(s.column__has_attribute.c.column_id, s.column_attribute.c.id)
    ):
        attributes.setdefault(str(row["column_id"]), dict(row))

    for column in columns:
        column["source_description"] = _source_text(column)
        attribute = attributes.get(str(column["id"]))
        column["attribute_id"] = attribute["id"] if attribute else None
        column["attribute_certified"] = bool(attribute and attribute["certified"])
    table["columns"] = columns
    return table


def _source_text(row: dict[str, Any]) -> str:
    preserved = row.get("source_description")
    if preserved is not None:
        return preserved
    return row.get("description") or ""


def fetch_table_links(table_id: str) -> list[dict[str, Any]]:
    """Every column-level link between *table_id* and another table.

    Reads both declared FKs and SEMANTIC_FK edges, in both directions. Each row
    is ``{local_column, other_table_id, other_table_name, other_column,
    direction}``, where ``direction`` is ``"out"`` when the local column points
    at the other table and ``"in"`` when the other table points here.
    Self-references are dropped.
    """
    local = s.catalog_column.alias("local_column")
    other = s.catalog_column.alias("other_column")
    other_table = s.catalog_table.alias("other_table")

    def _select(
        direction: str, link_from: Any, local_id: Any, other_id: Any
    ) -> tuple[Any, str]:
        return (
            select(
                local.c.name.label("local_column"),
                other_table.c.id.label("other_table_id"),
                other_table.c.name.label("other_table_name"),
                other.c.name.label("other_column"),
            )
            .select_from(
                link_from.join(local, local.c.id == local_id)
                .join(other, other.c.id == other_id)
                .join(other_table, other_table.c.id == other.c.table_id)
            )
            .where(local.c.table_id == table_id, other.c.table_id != table_id)
        ), direction

    fk = s.column__foreign_key
    semantic_fk = s.column__semantic_fk
    owner = s.column__has_attribute
    semantic_link = semantic_fk.join(
        owner, owner.c.attribute_id == semantic_fk.c.attribute_id
    )
    statements = [
        _select("out", fk, fk.c.source_column_id, fk.c.target_column_id),
        _select("in", fk, fk.c.target_column_id, fk.c.source_column_id),
        _select("out", semantic_link, semantic_fk.c.column_id, owner.c.column_id),
        _select("in", semantic_link, owner.c.column_id, semantic_fk.c.column_id),
    ]

    seen: set[tuple[str, str, str, str]] = set()
    links: list[dict[str, Any]] = []
    for statement, direction in statements:
        for row in store().query_read(statement):
            key = (
                direction,
                row["local_column"],
                str(row["other_table_id"]),
                row["other_column"],
            )
            if key in seen:
                continue
            seen.add(key)
            links.append({**dict(row), "direction": direction})
    return links


def fetch_neighbour_tables(table_ids: list[str]) -> dict[str, dict[str, Any]]:
    """Name, description and columns of each table, keyed by table id.

    Descriptions use the same fallbacks the SQL prompt reads, so a neighbour is
    shown to the LLM the way the text-to-SQL agent will see it.
    """
    if not table_ids:
        return {}
    tables = {
        str(row["id"]): {**dict(row), "columns": []}
        for row in store().query_read(
            select(
                s.catalog_table.c.id,
                s.catalog_table.c.name,
                table_description_expr().label("description"),
            ).where(s.catalog_table.c.id.in_(table_ids))
        )
    }
    for row in store().query_read(
        select(
            s.catalog_column.c.table_id,
            s.catalog_column.c.name,
            s.catalog_column.c.data_type,
            column_description_expr().label("description"),
            s.catalog_column.c.sample_values,
            s.catalog_column.c.format,
        )
        .where(s.catalog_column.c.table_id.in_(table_ids))
        .order_by(s.catalog_column.c.ordinal_position, s.catalog_column.c.id)
    ):
        table = tables.get(str(row["table_id"]))
        if table is not None:
            table["columns"].append(dict(row))
    return tables


def write_enriched_descriptions(
    table_id: str,
    table_description: str | None,
    column_descriptions: dict[str, str],
    attribute_descriptions: dict[str, str],
) -> dict[str, Any]:
    """Write enriched text in one transaction; certified text is never replaced.

    * Table and column ``description`` is rewritten unless
      ``description_certified``. The previous text moves to
      ``source_description`` only if that is still NULL.
    * An attribute's ``description`` is rewritten unless it is certified, in
      which case the text goes to ``description_suggestion`` for a person to
      accept.

    *column_descriptions* and *attribute_descriptions* are keyed by column id
    and attribute id. Returns the ids actually rewritten, under ``table``,
    ``columns`` and ``attributes``.
    """
    written: dict[str, Any] = {"table": False, "columns": [], "attributes": []}
    with write_transaction():
        if table_description:
            rows = store().query_write(
                update(s.catalog_table)
                .where(
                    s.catalog_table.c.id == table_id,
                    s.catalog_table.c.description_certified.is_(False),
                )
                .values(
                    source_description=func.coalesce(
                        s.catalog_table.c.source_description,
                        s.catalog_table.c.description,
                        "",
                    ),
                    description=table_description,
                )
                .returning(s.catalog_table.c.id)
            )
            written["table"] = bool(rows)

        for column_id, text in column_descriptions.items():
            rows = store().query_write(
                update(s.catalog_column)
                .where(
                    s.catalog_column.c.id == column_id,
                    s.catalog_column.c.table_id == table_id,
                    s.catalog_column.c.description_certified.is_(False),
                )
                .values(
                    source_description=func.coalesce(
                        s.catalog_column.c.source_description,
                        s.catalog_column.c.description,
                        "",
                    ),
                    description=text,
                )
                .returning(s.catalog_column.c.id)
            )
            if rows:
                written["columns"].append(str(rows[0]["id"]))

        for attribute_id, text in attribute_descriptions.items():
            rows = store().query_write(
                update(s.column_attribute)
                .where(s.column_attribute.c.id == attribute_id)
                .values(
                    description=func.coalesce(
                        _unless_certified(text), s.column_attribute.c.description
                    ),
                    description_suggestion=func.coalesce(
                        _if_certified(text),
                        s.column_attribute.c.description_suggestion,
                    ),
                )
                .returning(s.column_attribute.c.id, s.column_attribute.c.certified)
            )
            if rows and not rows[0]["certified"]:
                written["attributes"].append(str(rows[0]["id"]))
    return written


def _unless_certified(text: str) -> Any:
    return case((s.column_attribute.c.certified.is_(False), literal(text)), else_=None)


def _if_certified(text: str) -> Any:
    return case((s.column_attribute.c.certified.is_(True), literal(text)), else_=None)


def fetch_attribute_embedding_rows(attribute_ids: list[str]) -> list[dict[str, Any]]:
    """The rows :meth:`SemanticEmbedder.embed_column_attributes` needs.

    The same fields the server's sample-value refresh embeds with, so a
    re-embedded attribute reads exactly like a freshly compiled one.
    """
    if not attribute_ids:
        return []
    rows = store().query_read(
        select(
            s.column_attribute.c.id,
            s.column_attribute.c.name,
            s.column_attribute.c.description,
            s.column_attribute.c.datatype,
            s.column_attribute.c.term_name,
            s.column_attribute.c.source_column,
            s.catalog_column.c.sample_values,
            s.catalog_column.c.is_unique,
            s.catalog_table.c.id.label("table_id"),
            s.catalog_table.c.name.label("table_name"),
            s.catalog_schema.c.name.label("schema_name"),
        )
        .select_from(
            s.column_attribute.join(
                s.column__has_attribute,
                s.column__has_attribute.c.attribute_id == s.column_attribute.c.id,
            )
            .join(
                s.catalog_column,
                s.catalog_column.c.id == s.column__has_attribute.c.column_id,
            )
            .join(s.catalog_table, s.catalog_table.c.id == s.catalog_column.c.table_id)
            .join(
                s.catalog_schema, s.catalog_schema.c.id == s.catalog_table.c.schema_id
            )
        )
        .where(s.column_attribute.c.id.in_(attribute_ids))
        .order_by(s.column_attribute.c.id, s.catalog_column.c.id)
    )
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        if row["id"] in seen:
            continue
        seen.add(row["id"])
        out.append(dict(row))
    return out
