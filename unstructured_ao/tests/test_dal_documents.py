# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Document deletion against a migrated database, rolled back after each test."""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator

import pytest
from sqlalchemy import Connection, select, text

from auto_ontology.dal.schema import (
    unstructured_attribute_family,
    unstructured_attribute_family__attribute,
)
from unstructured_ao import dal


@pytest.fixture
def conn() -> Iterator[Connection]:
    if not os.environ.get("POSTGRES_USER"):
        pytest.skip("POSTGRES_* not set")
    from auto_ontology.dal.session import get_engine

    with get_engine().connect() as connection:
        transaction = connection.begin()
        try:
            yield connection
        finally:
            transaction.rollback()


def _attribute(conn: Connection, document_id: str) -> str:
    term = dal.get_document_term(conn, document_id)
    term_id = (
        term["id"]
        if term
        else dal.insert_term(conn, document_id=document_id, name="Term")
    )
    return dal.insert_attribute(
        conn,
        term_id=term_id,
        name=f"attr-{uuid.uuid4().hex[:6]}",
        description=None,
        attribute_type="ref",
    )


def _family(conn: Connection, attribute_ids: list[str]) -> str:
    family_id = str(uuid.uuid4())
    conn.execute(
        unstructured_attribute_family.insert(),
        {"id": family_id, "canonical": "Measure", "model": "test"},
    )
    conn.execute(
        unstructured_attribute_family__attribute.insert(),
        [{"family_id": family_id, "attribute_id": a} for a in attribute_ids],
    )
    return family_id


def _document(conn: Connection) -> str:
    suffix = uuid.uuid4().hex[:8]
    return dal.insert_document(
        conn, name=f"{suffix}.txt", path=f"/tmp/{suffix}.txt", file_type="txt"
    )


def test_list_documents_filters_and_orders(conn: Connection) -> None:
    tag = uuid.uuid4().hex[:8]
    first = dal.insert_document(
        conn, name=f"{tag}-a.txt", path=f"/tmp/{tag}-a.txt", file_type="txt"
    )
    second = dal.insert_document(
        conn, name=f"{tag}-b.txt", path=f"/tmp/{tag}-b.txt", file_type="txt"
    )
    conn.execute(
        text(
            "UPDATE unstructured_document SET ingested_at = now() - interval '1 day' "
            "WHERE id = :id"
        ),
        {"id": first},
    )
    dal.insert_term(conn, document_id=second, name="Term")

    def ids(**kwargs: object) -> list[str]:
        return [r["id"] for r in dal.list_documents(conn, search=tag.upper(), **kwargs)]

    assert ids() == [first, second]
    assert ids(newest_first=True) == [second, first]
    assert ids(without_term=True) == [first]
    assert ids(newest_first=True, limit=1) == [second]
    assert dal.count_documents(conn, search=tag, without_term=True) == 1
    assert dal.list_documents(conn, search=tag)[0]["path"] == f"/tmp/{tag}-a.txt"


def test_deleting_a_document_drops_only_families_it_leaves_too_small(
    conn: Connection,
) -> None:
    a, b, c = _document(conn), _document(conn), _document(conn)
    pair = _family(conn, [_attribute(conn, a), _attribute(conn, b)])
    trio = _family(
        conn, [_attribute(conn, a), _attribute(conn, b), _attribute(conn, c)]
    )
    untouched = _family(conn, [_attribute(conn, b), _attribute(conn, c)])

    dal.delete_document(conn, a)

    remaining = set(
        conn.execute(
            select(unstructured_attribute_family.c.id).where(
                unstructured_attribute_family.c.id.in_([pair, trio, untouched])
            )
        ).scalars()
    )
    assert remaining == {trio, untouched}
