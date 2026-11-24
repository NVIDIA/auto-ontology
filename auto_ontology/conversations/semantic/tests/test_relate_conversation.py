# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Re-relating a transcript replaces its semantic edges in one step.

The model call sits between the delete and the write, and a failure there must
not leave the transcript with no semantic edges at all.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator

import pytest
from sqlalchemy import delete, select

from auto_ontology import vdb
from auto_ontology.conversations.semantic import pipeline
from auto_ontology.conversations.semantic import store as semantic_store
from auto_ontology.conversations.semantic.relate import SemanticEdge
from auto_ontology.dal import schema as s
from auto_ontology.dal.session import store


@pytest.fixture
def database_name() -> Iterator[str]:
    if not os.environ.get("POSTGRES_USER"):
        pytest.skip("POSTGRES_* not set")
    try:
        store().query_read("SELECT 1 FROM conversation_edge LIMIT 1")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"auto_ontology schema unavailable (alembic upgrade head): {exc}")
    name = f"test-relate-{uuid.uuid4().hex[:8]}"
    store().query_write(s.catalog_database.insert().values(name=name))
    yield name
    store().query_write(
        delete(s.catalog_database).where(s.catalog_database.c.name == name)
    )


def _fact(name: str, key: str) -> dict:
    return {
        "name": name,
        "normalized_name": name,
        "type": "Fact",
        "kinds": ["fact"],
        "attribute_key": key,
        "value": name,
    }


_EMBED_FOR_RELATE = pipeline._embed_for_relate


@pytest.fixture
def graph(database_name: str, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    ids = semantic_store.replace_graph(
        database_name=database_name,
        conversation_id="1",
        nodes=[_fact("cache in memory", "cache"), _fact("cache on disk", "storage")],
        edges=[],
    )
    semantic_store.write_edges(
        [
            SemanticEdge(
                source_id=ids["cache on disk"],
                target_id=ids["cache in memory"],
                role="contradicts",
                reason="old verdict",
                distance=None,
            )
        ]
    )

    def rekey(statements: list[dict]) -> int:
        for row in statements:
            row["attribute_key"] = "cache"
        return len(statements)

    monkeypatch.setattr(pipeline, "reconcile_keys", rekey)
    monkeypatch.setattr(pipeline, "derive_edges", lambda _: ([], set()))
    monkeypatch.setattr(pipeline, "_embed_for_relate", lambda _: True)
    monkeypatch.setattr(pipeline, "_embed_paths", lambda **_: 0)
    monkeypatch.setattr(vdb, "get_one_hop_path_vdb", lambda **_: _NoVDB())
    return ids


class _NoVDB:
    def delete_by_id(self, vector_id: str) -> int:
        return 1


def _semantic(ids: dict[str, str]) -> list[tuple[str, str | None]]:
    rows = store().query_read(
        select(s.conversation_edge).where(
            s.conversation_edge.c.type == "semantic",
            s.conversation_edge.c.source_id.in_(list(ids.values())),
        )
    )
    return sorted((row["role"], row["reason"]) for row in rows)


def _keys(ids: dict[str, str]) -> list[str]:
    rows = store().query_read(
        select(s.conversation_node.c.attribute_key).where(
            s.conversation_node.c.id.in_(list(ids.values()))
        )
    )
    return sorted(row["attribute_key"] for row in rows)


def test_a_failed_adjudication_keeps_the_old_edges(
    database_name: str, graph: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*_: object, **__: object) -> list:
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(pipeline, "relate_nodes", fail)

    with pytest.raises(RuntimeError):
        pipeline.relate_conversation(database_name=database_name, conversation_id="1")

    assert _semantic(graph) == [("contradicts", "old verdict")]
    assert _keys(graph) == ["cache", "storage"]


def test_a_failed_embedding_keeps_the_old_edges(
    database_name: str, graph: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from auto_ontology.utils import embedding

    def down(*_: object, **__: object) -> list:
        raise RuntimeError("embedding endpoint down")

    monkeypatch.setattr(pipeline, "_embed_for_relate", _EMBED_FOR_RELATE)
    monkeypatch.setattr(embedding, "embed_texts", down)
    monkeypatch.setattr(pipeline, "relate_nodes", lambda *_, **__: [])

    with pytest.raises(RuntimeError, match="could not be embedded"):
        pipeline.relate_conversation(database_name=database_name, conversation_id="1")

    assert _semantic(graph) == [("contradicts", "old verdict")]


def test_a_pair_with_no_verdict_keeps_the_old_edges(
    database_name: str, graph: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_verdict(*_: object, failed: list, **__: object) -> list:
        failed.append(frozenset((graph["cache on disk"], graph["cache in memory"])))
        return []

    monkeypatch.setattr(pipeline, "relate_nodes", no_verdict)

    with pytest.raises(RuntimeError, match="could not be adjudicated"):
        pipeline.relate_conversation(database_name=database_name, conversation_id="1")

    assert _semantic(graph) == [("contradicts", "old verdict")]
    assert _keys(graph) == ["cache", "storage"]


def test_replaces_the_edges_and_saves_the_reconciled_keys(
    database_name: str, graph: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    new = SemanticEdge(
        source_id=graph["cache on disk"],
        target_id=graph["cache in memory"],
        role="supersedes",
        reason="new verdict",
        distance=None,
    )
    monkeypatch.setattr(pipeline, "relate_nodes", lambda *_, **__: [new])

    written = pipeline.relate_conversation(
        database_name=database_name, conversation_id="1"
    )

    assert written == 1
    assert _semantic(graph) == [("supersedes", "new verdict")]
    assert _keys(graph) == ["cache", "cache"]
