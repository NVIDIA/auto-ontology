# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A rebuild drops the vectors of the graph it replaces.

``replace_graph`` gives every node a new id, so the next embed never overwrites
the previous build's vectors; they have to be found and deleted by id.
"""

from __future__ import annotations

import pytest

from auto_ontology import vdb
from auto_ontology.conversations.semantic import pipeline
from auto_ontology.conversations.semantic import store as semantic_store
from auto_ontology.conversations.semantic.paths import path_rows

NODES = [
    {"id": "n1", "name": "weather app", "description": None, "kinds": ["object"]},
    {"id": "n2", "name": "localstorage", "description": None, "kinds": ["object"]},
    {"id": "n3", "name": "cache on disk", "description": None, "kinds": ["fact"]},
]
EDGES = [
    {
        "source_id": "n1",
        "target_id": "n2",
        "type": "knowledge",
        "role": "built_with",
        "statement": "weather app built with localstorage",
        "turn_indexes": [0],
    },
    {
        "source_id": "n3",
        "target_id": "n2",
        "type": "semantic",
        "role": "unrelated",
        "statement": None,
        "turn_indexes": [1],
    },
]


class _VDB:
    def __init__(self, deleted: list[str], *, fail: bool = False) -> None:
        self.deleted = deleted
        self.fail = fail

    def delete_by_id(self, vector_id: str) -> int:
        if self.fail:
            raise RuntimeError("vector store down")
        self.deleted.append(vector_id)
        return 1


@pytest.fixture
def stored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(semantic_store, "fetch_nodes", lambda *_, **__: NODES)
    monkeypatch.setattr(semantic_store, "fetch_edges", lambda *_, **__: EDGES)


def test_finds_every_node_and_path_of_the_old_graph(stored: None) -> None:
    nodes, paths = pipeline._stale_vectors(database_name="beam", conversation_id="2")

    assert nodes == ["n1", "n2", "n3"]
    # ``unrelated`` is never embedded, so it has no vector to find.
    assert paths == [row["id"] for row in path_rows(NODES, EDGES)]
    assert len(paths) == 1


def test_nothing_stored_means_nothing_to_forget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(semantic_store, "fetch_nodes", lambda *_, **__: [])

    assert pipeline._stale_vectors(database_name="beam", conversation_id="2") == (
        [],
        [],
    )


def test_deletes_node_and_path_vectors_by_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node_deleted: list[str] = []
    path_deleted: list[str] = []
    monkeypatch.setattr(
        vdb, "get_conversation_semantic_vdb", lambda **_: _VDB(node_deleted)
    )
    monkeypatch.setattr(vdb, "get_one_hop_path_vdb", lambda **_: _VDB(path_deleted))

    pipeline._forget_vectors(
        database_name="beam", node_ids=["n1", "n2"], path_ids=["p1"]
    )

    assert node_deleted == ["n1", "n2"]
    assert path_deleted == ["p1"]


def test_a_failing_collection_does_not_stop_the_other(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path_deleted: list[str] = []
    monkeypatch.setattr(
        vdb, "get_conversation_semantic_vdb", lambda **_: _VDB([], fail=True)
    )
    monkeypatch.setattr(vdb, "get_one_hop_path_vdb", lambda **_: _VDB(path_deleted))

    pipeline._forget_vectors(database_name="beam", node_ids=["n1"], path_ids=["p1"])

    assert path_deleted == ["p1"]
