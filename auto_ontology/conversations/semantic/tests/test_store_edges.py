# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Knowledge edges that land on one primary key are merged, not rejected.

Two names can resolve to one node, and a single duplicate edge would fail the
key and roll back the whole transcript's write.
"""

from __future__ import annotations

import os
import uuid

import pytest
from sqlalchemy import delete, select

from auto_ontology.conversations.semantic import store as semantic_store
from auto_ontology.dal import schema as s
from auto_ontology.dal.session import store


@pytest.fixture
def database_name():
    if not os.environ.get("POSTGRES_USER"):
        pytest.skip("POSTGRES_* not set")
    try:
        store().query_read("SELECT 1 FROM conversation_edge LIMIT 1")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"auto_ontology schema unavailable (alembic upgrade head): {exc}")
    name = f"test-edges-{uuid.uuid4().hex[:8]}"
    store().query_write(s.catalog_database.insert().values(name=name))
    yield name
    store().query_write(
        delete(s.catalog_database).where(s.catalog_database.c.name == name)
    )


def _node(name: str) -> dict:
    return {"name": name, "normalized_name": name, "type": "Thing", "kinds": ["object"]}


def test_duplicate_edges_merge_their_turns(database_name: str) -> None:
    edge = {"source": "weather app", "target": "flask", "role": "built_with"}

    ids = semantic_store.replace_graph(
        database_name=database_name,
        conversation_id="1",
        nodes=[_node("weather app"), _node("flask")],
        edges=[
            {**edge, "turn_indexes": [3], "evidence": ""},
            {**edge, "turn_indexes": [7, 3], "evidence": "built it with Flask"},
        ],
    )

    rows = store().query_read(
        select(s.conversation_edge).where(
            s.conversation_edge.c.source_id == ids["weather app"]
        )
    )
    assert len(rows) == 1
    assert rows[0]["turn_indexes"] == [3, 7]
    assert rows[0]["evidence"] == "built it with Flask"
