# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Write a semantic FK in a test the way it was written before Concept keys.

A test that seeds catalog rows by hand has no Terms, so no Concepts for
``merge_semantic_fk`` to key. :func:`link_semantic_fk` keys the attribute's
owning column on a Concept of that column's database directly; deleting the
test's database removes the Concept and the FK with it.
"""

from __future__ import annotations

from sqlalchemy import func, select

from auto_ontology.dal import schema as s
from auto_ontology.dal.session import store


def _one(statement) -> dict | None:
    rows = store().query_read(statement)
    return rows[0] if rows else None


def column_key(column_id: str) -> str:
    """The Concept key *column_id* is a member column of, created if needed."""
    existing = _one(
        select(s.concept_key__column.c.key_id).where(
            s.concept_key__column.c.column_id == column_id
        )
    )
    if existing:
        return existing["key_id"]
    column = _one(
        select(
            s.catalog_column.c.name,
            s.catalog_column.c.table_id,
            s.catalog_schema.c.database_id,
        )
        .join(s.catalog_table, s.catalog_table.c.id == s.catalog_column.c.table_id)
        .join(s.catalog_schema, s.catalog_schema.c.id == s.catalog_table.c.schema_id)
        .where(s.catalog_column.c.id == column_id)
    )
    assert column, f"no column {column_id}"
    concept = _one(
        select(s.concept_key.c.concept_id)
        .join(
            s.concept_key__column,
            s.concept_key__column.c.key_id == s.concept_key.c.id,
        )
        .where(s.concept_key__column.c.table_id == column["table_id"])
    )
    concept_id = (
        concept["concept_id"]
        if concept
        else store().query_write(
            s.concept.insert()
            .values(
                database_id=column["database_id"],
                name=f"test-concept-{column['table_id']}",
            )
            .returning(s.concept.c.id)
        )[0]["id"]
    )
    key_id = store().query_write(
        s.concept_key.insert()
        .values(concept_id=concept_id, name=column["name"])
        .returning(s.concept_key.c.id)
    )[0]["id"]
    store().query_write(
        s.concept_key__column.insert().values(
            key_id=key_id, column_id=column_id, table_id=column["table_id"]
        )
    )
    return key_id


def link_semantic_fk(
    column_id: str, attribute_id: str, *, use_in_join_paths: bool = True
) -> str:
    """*column_id* references the key of the column owning *attribute_id*."""
    owner = _one(
        select(func.min(s.column__has_attribute.c.column_id).label("id")).where(
            s.column__has_attribute.c.attribute_id == attribute_id
        )
    )
    assert owner and owner["id"], f"attribute {attribute_id} owns no column"
    key_id = column_key(owner["id"])
    store().query_write(
        s.column__semantic_fk.insert().values(
            column_id=column_id, key_id=key_id, use_in_join_paths=use_in_join_paths
        )
    )
    return key_id
