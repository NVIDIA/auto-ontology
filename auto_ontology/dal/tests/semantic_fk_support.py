# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test helper: write a semantic FK in the column -> attribute shape.

A semantic FK references a concept. Tests built around attributes still say
"this column references that attribute", which this turns into a concept
keyed by the column(s) owning the attribute, and a reference to it. Unlike
:func:`auto_ontology.dal.attributes.merge_semantic_fk` it needs no Term,
so a fixture can stay minimal.
"""

from __future__ import annotations

from sqlalchemy import select

from auto_ontology.dal import schema as s
from auto_ontology.dal.concepts import (
    SEMANTIC_FK_ATTRIBUTES,
    concept_keys_containing,
    merge_concept_fk,
    set_concept_key,
)
from auto_ontology.dal.session import store


def concept_for_attribute(attribute_id: str) -> str:
    """The concept keyed by the attribute's owning columns, created if needed."""
    owners = store().query_read(
        select(s.catalog_column.c.id, s.catalog_column.c.table_id)
        .select_from(
            s.column__has_attribute.join(
                s.catalog_column,
                s.catalog_column.c.id == s.column__has_attribute.c.column_id,
            )
        )
        .where(s.column__has_attribute.c.attribute_id == attribute_id)
        .order_by(s.catalog_column.c.id)
    )
    assert owners, f"attribute {attribute_id} has no owning column"
    for owner in owners:
        keys = concept_keys_containing(owner["id"])
        if keys:
            concept_id = keys[0]["concept_id"]
            break
    else:
        concept_id = store().query_write(
            s.concept.insert()
            .values(name=f"test-concept-{attribute_id}", source="test")
            .returning(s.concept.c.id)
        )[0]["id"]
    for owner in owners:
        if not concept_keys_containing(owner["id"]):
            assert set_concept_key(concept_id, owner["table_id"], [owner["id"]])
    return concept_id


def reference_attribute(
    column_id: str, attribute_id: str, *, use_in_join_paths: bool = True
) -> str:
    """Make *column_id* reference the concept keyed by *attribute_id*'s owner."""
    fk_id = merge_concept_fk(concept_for_attribute(attribute_id), [column_id])
    assert fk_id, f"reference {column_id} -> {attribute_id} was refused"
    store().query_write(
        s.concept__fk.update()
        .where(s.concept__fk.c.id == fk_id)
        .values(use_in_join_paths=use_in_join_paths)
    )
    return fk_id


def references_of(attribute_id: str) -> list[dict]:
    """``{column_id, use_in_join_paths}`` for every column referencing the
    attribute's concept, ordered by column id."""
    return [
        dict(row)
        for row in store().query_read(
            select(
                SEMANTIC_FK_ATTRIBUTES.c.column_id,
                SEMANTIC_FK_ATTRIBUTES.c.use_in_join_paths,
            )
            .where(SEMANTIC_FK_ATTRIBUTES.c.attribute_id == attribute_id)
            .order_by(SEMANTIC_FK_ATTRIBUTES.c.column_id)
        )
    ]
