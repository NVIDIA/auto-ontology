# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""QdrantVDB must behave like PostgresVDB for every caller.

Callers read ``metadata["id"]`` and ``label`` from the top level of a hit's
metadata and compare ``_distance`` against cosine-distance thresholds, so the
round trip below writes through the real :class:`QdrantVDB` and reads back
through the real :class:`Retriever`.

Every test runs twice: once in Qdrant's local in-memory mode, which needs no
server, and once against the server at ``QDRANT_URL`` (default
``http://localhost:6333``), which is skipped when no server answers. Only the
server builds payload indexes, so only the server run proves the filters use
them.
"""

from __future__ import annotations

import os
import uuid
from typing import Any, Iterator

import pytest
from qdrant_client import QdrantClient, models

from auto_ontology.utils.embedding import EmbedParams
from auto_ontology.utils.embedding_rows import build_embed_row
from auto_ontology.vdb import (
    BACKEND_ENV,
    get_semantic_vdb,
    get_vdb,
    get_vdb_backend,
)
from auto_ontology.vdb.base import VectorStore
from auto_ontology.vdb.postgres import PostgresVDB
from auto_ontology.vdb.qdrant import QdrantVDB, build_filter
from auto_ontology.vdb.records import write_embedded_rows

_DIMENSIONS = 2048

# Local mode ignores payload indexes and warns about it; the server uses them.
pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes have no effect")


def _unit(axis: int) -> list[float]:
    vector = [0.0] * _DIMENSIONS
    vector[axis] = 1.0
    return vector


def _row(
    node_id: str,
    embedding: Any,
    *,
    text: str = "orders table",
    label: str = "Table",
    database_name: str = "shop",
) -> dict:
    row = build_embed_row(
        text=text,
        node_id=node_id,
        label=label,
        name=node_id,
        schema_name="public",
        database_name=database_name,
    )
    if embedding is not None:
        row["metadata"]["embedding"] = embedding
    return row


@pytest.fixture(params=["memory", "server"])
def client(request: pytest.FixtureRequest) -> Iterator[QdrantClient]:
    if request.param == "memory":
        client = QdrantClient(":memory:")
    else:
        client = QdrantClient(
            url=os.environ.get("QDRANT_URL", "http://localhost:6333"),
            api_key=os.environ.get("QDRANT_API_KEY") or None,
            timeout=5,
            # The probe below reports a missing server; skip the version warning.
            check_compatibility=False,
        )
        try:
            client.get_collections()
        except Exception as exc:  # noqa: BLE001 -- no server, nothing to assert
            client.close()
            pytest.skip(f"qdrant not reachable: {exc}")
    try:
        yield client
    finally:
        client.close()


@pytest.fixture
def name(client: QdrantClient) -> Iterator[str]:
    """A collection name no other run uses, dropped afterwards."""
    name = f"probe_{uuid.uuid4().hex[:8]}"
    try:
        yield name
    finally:
        if client.collection_exists(name):
            client.delete_collection(name)


@pytest.fixture
def vdb(client: QdrantClient, name: str) -> QdrantVDB:
    return QdrantVDB(client=client, collection_name=name)


def _search(
    vdb: QdrantVDB, axis: int, where: dict | None = None, top_k: int = 10
) -> list[dict]:
    return vdb.retrieval([_unit(axis)], top_k=top_k, where=where)[0]


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------


def test_both_backends_implement_the_interface(vdb: QdrantVDB) -> None:
    assert isinstance(vdb, VectorStore)
    # The constructor does not connect, so no database is needed here.
    assert isinstance(PostgresVDB(connection_string="postgresql://u@h/d"), VectorStore)
    assert vdb.metadata_filter_format == "dict"


def test_reads_on_a_missing_collection_are_empty(vdb: QdrantVDB) -> None:
    assert vdb.retrieval([_unit(0), _unit(1)]) == [[], []]
    assert vdb.delete_by_id("x") == 0
    assert vdb.delete_by_ids(["x", "y"]) == 0
    assert vdb.delete_by_database("shop") == 0
    assert vdb.delete_all() == 0


def test_a_missing_collection_is_not_logged_as_an_error(
    vdb: QdrantVDB, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level("ERROR", logger="auto_ontology.vdb.qdrant"):
        assert vdb.retrieval([_unit(0)]) == [[]]
    assert caplog.records == []


def test_the_collection_uses_full_precision_cosine(
    vdb: QdrantVDB, client: QdrantClient
) -> None:
    assert vdb.create_index() == vdb.collection_name
    params = client.get_collection(vdb.collection_name).config.params.vectors
    assert params.size == _DIMENSIONS
    assert params.distance == models.Distance.COSINE
    # A second call finds the collection and leaves it alone.
    assert vdb.create_index() == vdb.collection_name


def test_the_server_indexes_the_filtered_keys(
    vdb: QdrantVDB, client: QdrantClient
) -> None:
    if client.init_options.get("location") == ":memory:":
        pytest.skip("local mode does not build payload indexes")
    vdb.create_index()
    schema = client.get_collection(vdb.collection_name).payload_schema
    assert {
        "metadata.database_name",
        "metadata.label",
        "metadata.schema_name",
        "metadata.id",
    } <= set(schema)
    assert schema["metadata.database_name"].params.is_tenant is True


def test_a_collection_without_indexes_gets_them_on_the_next_write(
    vdb: QdrantVDB, client: QdrantClient
) -> None:
    """A crash between create_collection and the index calls must heal."""
    if client.init_options.get("location") == ":memory:":
        pytest.skip("local mode does not build payload indexes")
    client.create_collection(
        vdb.collection_name,
        vectors_config=models.VectorParams(
            size=_DIMENSIONS, distance=models.Distance.COSINE
        ),
    )
    assert not client.get_collection(vdb.collection_name).payload_schema

    vdb.run([[_record("a", 0)]])

    schema = client.get_collection(vdb.collection_name).payload_schema
    assert {"metadata.database_name", "metadata.id"} <= set(schema)


def test_a_dropped_collection_is_created_again_on_write(
    vdb: QdrantVDB, client: QdrantClient
) -> None:
    """A long-lived store must not trust its cached state after a drop."""
    vdb.run([[_record("a", 0)]])
    client.delete_collection(vdb.collection_name)

    # The write comes first, so the store still believes the collection exists.
    assert vdb.write_to_index([[_record("b", 1)]]) == 1
    assert [hit["metadata"]["id"] for hit in _search(vdb, 1)] == ["b"]

    client.delete_collection(vdb.collection_name)
    assert _search(vdb, 0) == []


# ---------------------------------------------------------------------------
# Round trip
# ---------------------------------------------------------------------------


def test_written_rows_come_back_from_a_search(
    vdb: QdrantVDB, monkeypatch: pytest.MonkeyPatch
) -> None:
    from auto_ontology.utils import retriever as retriever_mod

    written = write_embedded_rows(
        vdb,
        [
            _row("orders", _unit(0), text="orders table"),
            _row("customers", _unit(1), text="customers table"),
            _row("failed", [], text="never embedded"),
        ],
    )
    assert written == 2

    monkeypatch.setattr(
        retriever_mod, "embed_texts", lambda texts, params, kind, **kw: [_unit(1)]
    )
    retriever = retriever_mod.Retriever(vdb, EmbedParams("m", "http://unused"))

    hits = retriever.query(
        "who buys", top_k=5, vdb_kwargs={"where": {"database_name": "shop"}}
    )

    assert [hit["metadata"]["id"] for hit in hits] == ["customers", "orders"]
    best = hits[0]
    assert best["text"] == "customers table"
    assert best["metadata"]["label"] == "Table"
    assert best["metadata"]["document_type"] == "text"
    assert "embedding" not in best["metadata"]
    # Cosine distance, the scale DEFAULT_MAX_DISTANCE is tuned for: an
    # identical vector is 0 and an orthogonal one is 1.
    assert best["_distance"] == pytest.approx(0.0, abs=1e-3)
    assert hits[1]["_distance"] == pytest.approx(1.0, abs=1e-3)
    # The filter is applied by the store, not after the fact.
    assert (
        retriever.query("who buys", vdb_kwargs={"where": {"database_name": "x"}}) == []
    )


def test_a_label_list_matches_any_label(vdb: QdrantVDB) -> None:
    vdb.run(
        [
            [
                _record("t", 0, label="Table"),
                _record("c", 1, label="Column"),
                _record("a", 2, label="CustomAnalysis"),
            ]
        ]
    )

    hits = _search(vdb, 0, where={"label": ["Table", "Column"]})
    assert sorted(hit["metadata"]["id"] for hit in hits) == ["c", "t"]
    hits = _search(vdb, 0, where={"label": "CustomAnalysis"})
    assert [hit["metadata"]["id"] for hit in hits] == ["a"]


def test_a_selective_filter_still_fills_top_k(vdb: QdrantVDB) -> None:
    """Many near rows from one database must not crowd out another's."""
    crowd = [_record(f"n{i}", 0, database_name="noisy") for i in range(200)]
    rare = [_record(f"q{i}", 1, database_name="quiet") for i in range(5)]
    vdb.run([crowd + rare])

    hits = _search(vdb, 0, where={"database_name": "quiet"}, top_k=5)
    assert sorted(hit["metadata"]["id"] for hit in hits) == [f"q{i}" for i in range(5)]


def test_several_queries_return_one_list_each(vdb: QdrantVDB) -> None:
    vdb.run([[_record("x", 0), _record("y", 1)]])
    first, second = vdb.retrieval([_unit(0), _unit(1)], top_k=1)
    assert first[0]["metadata"]["id"] == "x"
    assert second[0]["metadata"]["id"] == "y"


# ---------------------------------------------------------------------------
# Deletes
# ---------------------------------------------------------------------------


def _record(
    node_id: str, axis: int, *, label: str = "Table", database_name: str = "shop"
) -> dict:
    return {
        "document_type": "text",
        "metadata": {
            "id": node_id,
            "label": label,
            "database_name": database_name,
            "content": f"{node_id} text",
            "embedding": _unit(axis),
        },
    }


def test_delete_by_id_removes_every_copy(vdb: QdrantVDB) -> None:
    vdb.run([[_record("a", 0), _record("a", 1), _record("b", 2)]])

    assert vdb.delete_by_id("a") == 2
    assert vdb.delete_by_id("a") == 0
    assert [hit["metadata"]["id"] for hit in _search(vdb, 0)] == ["b"]


def test_delete_by_ids_removes_them_in_one_call(vdb: QdrantVDB) -> None:
    vdb.run([[_record("a", 0), _record("b", 1), _record("c", 2)]])

    assert vdb.delete_by_ids(iter(["a", "b", "a"])) == 2
    assert vdb.delete_by_ids([]) == 0
    assert [hit["metadata"]["id"] for hit in _search(vdb, 2)] == ["c"]


def test_delete_by_database_returns_the_count(vdb: QdrantVDB) -> None:
    vdb.run(
        [
            [
                _record("a", 0, database_name="shop"),
                _record("b", 1, database_name="shop"),
                _record("c", 2, database_name="other"),
            ]
        ]
    )

    assert vdb.delete_by_database("shop") == 2
    assert vdb.delete_by_database("shop") == 0
    assert [hit["metadata"]["id"] for hit in _search(vdb, 2)] == ["c"]


def test_reset_clears_the_database_before_ingest(
    client: QdrantClient, name: str
) -> None:
    first = QdrantVDB(client=client, collection_name=name)
    first.run([[_record("old", 0), _record("keep", 1, database_name="other")]])

    second = QdrantVDB(
        client=client, collection_name=name, database_name="shop", reset=True
    )
    assert sorted(hit["metadata"]["id"] for hit in _search(second, 0)) == ["keep"]


def test_delete_all_keeps_the_collection(vdb: QdrantVDB, client: QdrantClient) -> None:
    vdb.run([[_record("a", 0), _record("b", 1)]])

    assert vdb.delete_all() == 2
    assert client.collection_exists(vdb.collection_name)
    assert _search(vdb, 0) == []


def test_close_leaves_a_shared_client_open(
    vdb: QdrantVDB, client: QdrantClient
) -> None:
    vdb.create_index()
    vdb.close()
    assert client.collection_exists(vdb.collection_name)
    assert vdb.create_index() == vdb.collection_name


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


def test_build_filter_maps_flat_entries_to_metadata_paths() -> None:
    query_filter = build_filter({"label": ["Table", "Column"], "database_name": "x"})
    assert query_filter == models.Filter(
        must=[
            models.FieldCondition(
                key="metadata.label", match=models.MatchAny(any=["Table", "Column"])
            ),
            models.FieldCondition(
                key="metadata.database_name", match=models.MatchValue(value="x")
            ),
        ]
    )
    assert build_filter(None) is None
    assert build_filter({}) is None


def test_build_filter_rejects_a_sql_string() -> None:
    with pytest.raises(TypeError):
        build_filter("metadata LIKE '%x%'")


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------


def test_postgres_is_the_default_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(BACKEND_ENV, raising=False)
    assert get_vdb_backend() == "postgres"


def test_an_unknown_backend_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(BACKEND_ENV, "milvus")
    with pytest.raises(ValueError, match="milvus"):
        get_vdb_backend()


def test_the_qdrant_backend_shares_one_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import auto_ontology.vdb as vdb_mod

    client = QdrantClient(":memory:")
    monkeypatch.setenv(BACKEND_ENV, " Qdrant ")
    monkeypatch.setattr(vdb_mod, "_qdrant_client", client)

    data = get_vdb()
    semantic = get_semantic_vdb()
    assert isinstance(data, QdrantVDB) and isinstance(semantic, QdrantVDB)
    assert data.client is semantic.client is client
    assert (data.collection_name, semantic.collection_name) == (
        "data_objects_layer",
        "semantic_layer",
    )


# ---------------------------------------------------------------------------
# Second review
# ---------------------------------------------------------------------------


def test_stores_on_one_client_share_the_ready_state(
    vdb: QdrantVDB, client: QdrantClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """get_vdb() builds a store per call; only the first may pay for setup."""
    vdb.create_index()
    calls: list[str] = []
    for method in ("collection_exists", "get_collection"):
        original = getattr(client, method)
        monkeypatch.setattr(
            client,
            method,
            lambda *a, _m=method, _o=original, **kw: calls.append(_m) or _o(*a, **kw),
        )

    fresh = QdrantVDB(client=client, collection_name=vdb.collection_name)
    fresh.run([[_record("a", 0)]])
    fresh.delete_by_id("a")
    assert calls == []


def test_a_collection_with_other_vectors_is_refused(
    client: QdrantClient, name: str
) -> None:
    client.create_collection(
        name,
        vectors_config=models.VectorParams(size=8, distance=models.Distance.DOT),
    )
    store = QdrantVDB(client=client, collection_name=name)
    with pytest.raises(ValueError, match="EMBED_MODEL"):
        store.create_index()


def test_an_upsert_error_is_not_replaced_by_the_check(
    vdb: QdrantVDB, client: QdrantClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    vdb.create_index()

    def _upsert(*args: Any, **kwargs: Any) -> None:
        raise ConnectionError("upsert timed out")

    def _exists(*args: Any, **kwargs: Any) -> bool:
        raise TimeoutError("check timed out too")

    monkeypatch.setattr(client, "upsert", _upsert)
    monkeypatch.setattr(client, "collection_exists", _exists)
    with pytest.raises(ConnectionError, match="upsert timed out"):
        vdb.write_to_index([[_record("a", 0)]])


def test_a_dropped_collection_reads_as_empty_on_delete(
    vdb: QdrantVDB, client: QdrantClient
) -> None:
    vdb.run([[_record("a", 0)]])
    client.delete_collection(vdb.collection_name)
    assert vdb.delete_by_id("a") == 0
    assert vdb.write_to_index([[_record("b", 1)]]) == 1


@pytest.mark.parametrize(
    ("raw", "expected"), [("", 60), ("30", 30), ("7.5", 8), (" 5 ", 5)]
)
def test_qdrant_timeout_accepts_seconds(
    monkeypatch: pytest.MonkeyPatch, raw: str, expected: int
) -> None:
    from auto_ontology.vdb import _qdrant_timeout

    monkeypatch.setenv("QDRANT_TIMEOUT", raw)
    assert _qdrant_timeout() == expected


@pytest.mark.parametrize("raw", ["soon", "0", "-1", "inf", "nan"])
def test_qdrant_timeout_names_the_setting(
    monkeypatch: pytest.MonkeyPatch, raw: str
) -> None:
    from auto_ontology.vdb import _qdrant_timeout

    monkeypatch.setenv("QDRANT_TIMEOUT", raw)
    with pytest.raises(ValueError, match="QDRANT_TIMEOUT"):
        _qdrant_timeout()
