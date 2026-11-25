# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``turn_count`` counts one memory's turns in the database, not in Python."""

from __future__ import annotations

import os
import uuid

import pytest
from sqlalchemy import delete, select

from auto_ontology.conversations.store import turn_count
from auto_ontology.dal import schema as s
from auto_ontology.dal.session import store


@pytest.fixture
def database_name():
    if not os.environ.get("POSTGRES_USER"):
        pytest.skip("POSTGRES_* not set")
    try:
        store().query_read("SELECT 1 FROM conversation LIMIT 1")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"auto_ontology schema unavailable (alembic upgrade head): {exc}")
    name = f"test-count-{uuid.uuid4().hex[:8]}"
    store().query_write(s.catalog_database.insert().values(name=name))
    yield name
    store().query_write(
        delete(s.catalog_database).where(s.catalog_database.c.name == name)
    )


def test_counts_only_this_memorys_turns(database_name: str) -> None:
    database_id = store().query_read(
        select(s.catalog_database.c.id).where(
            s.catalog_database.c.name == database_name
        )
    )[0]["id"]
    for index in range(3):
        store().query_write(
            s.conversation.insert().values(
                database_id=database_id,
                conversation_id="1",
                turn_index=index,
                question="q",
                answer="a",
            )
        )

    assert turn_count(database_name) == 3
    assert turn_count(f"{database_name}-other") == 0
