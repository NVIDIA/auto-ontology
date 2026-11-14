# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Elasticsearch store against a real cluster, through the app's own write and read paths.

Writes go through :func:`~auto_ontology.utils.embedding.embed_rows` and
:func:`~auto_ontology.vdb.records.write_embedded_rows`, and reads through
:class:`~auto_ontology.utils.retriever.Retriever` -- the functions the services
call -- so the switch to "the store embeds" is under test, not bypassed.

Elasticsearch embeds rows and queries itself, by calling its ``nvidia``
inference endpoint over HTTP, so a monkeypatch in this process cannot stand in
for the model. The endpoint is pointed at :mod:`dev_tools.fake_embedder`
instead. The Postgres store needs none of this: the app embeds for it, and
``test_records.py`` covers it with hand-made vectors.

Needs a reachable Elasticsearch (``ES_URL`` plus ``ES_API_KEY`` or
``ES_USERNAME``/``ES_PASSWORD``). The node calls back into this process for
embeddings, at ``ES_TEST_EMBEDDER_HOST`` -- ``host.docker.internal`` by
default, which is how a container reaches the host on Docker Desktop and on a
Linux runner started with ``--add-host=host.docker.internal:host-gateway``.
"""

from __future__ import annotations

import os
import uuid
from typing import Iterator

import pytest

from auto_ontology.infra.elasticsearch import (
    EmbedEndpoint,
    get_elasticsearch_client,
    get_elasticsearch_config,
)
from auto_ontology.retrieval.data_access.hit_scores import passes_relevance_gate
from auto_ontology.retrieval.data_access.semantic_search import (
    _build_metadata_where_clause,
    search_semantic_index,
)
from auto_ontology.utils.embedding import EmbedParams, embed_rows
from auto_ontology.utils.embedding_rows import build_embed_row
from auto_ontology.utils.retriever import Retriever
from auto_ontology.vdb.base import VectorStore
from auto_ontology.vdb.elasticsearch import ElasticsearchVDB
from auto_ontology.vdb.records import replace_embedded_rows, write_embedded_rows
from dev_tools.fake_embedder import FakeEmbedder, embed

#: The model name each side would send, so the recorded calls say who embedded.
_APP_MODEL = "fake/app-side"
_ES_MODEL = "fake/es-side"

_ROWS = [
    # (id, label, database, text)
    ("c_email", "Column", "pagila", "customer email address"),
    ("c_rate", "Column", "pagila", "film rental rate in dollars"),
    ("t_customer", "Table", "pagila", "customer table with first and last names"),
    ("c_billing", "Column", "chinook", "invoice billing address"),
    ("c_length", "Column", "pagila", "film length in minutes"),
]


def _rows(rows: list[tuple[str, str, str, str]] = _ROWS) -> list[dict]:
    return [
        build_embed_row(
            text=text,
            node_id=node_id,
            label=label,
            name=node_id,
            schema_name="public",
            database_name=database,
        )
        for node_id, label, database, text in rows
    ]


@pytest.fixture(scope="module")
def fake() -> Iterator[FakeEmbedder]:
    # 0.0.0.0, not loopback: the Elasticsearch container calls in from outside.
    with FakeEmbedder(host="0.0.0.0") as server:
        yield server


def _elasticsearch(fake: FakeEmbedder) -> Iterator[tuple[VectorStore, EmbedParams]]:
    try:
        config = get_elasticsearch_config()
        client = get_elasticsearch_client(config)
        client.info()
    except Exception as exc:  # noqa: BLE001 -- no cluster, nothing to assert
        pytest.skip(f"elasticsearch not reachable: {exc}")
    name = f"ao-contract-{uuid.uuid4().hex[:8]}"
    host = os.environ.get("ES_TEST_EMBEDDER_HOST", "host.docker.internal")
    endpoint = EmbedEndpoint(
        inference_id=name,
        model_id=_ES_MODEL,
        api_key="unused",
        url=f"{fake.url(host)}/embeddings",
    )
    store = ElasticsearchVDB(client=client, index_name=name, endpoint=endpoint)
    try:
        # The app-side params must never be used on this backend; pointing them
        # at a dead port makes a regression fail loudly instead of passing.
        yield (
            store,
            EmbedParams(
                model_name=_APP_MODEL, embed_invoke_url="http://127.0.0.1:9/v1"
            ),
        )
    finally:
        client.options(ignore_status=404).indices.delete(index=name)
        client.options(ignore_status=404).inference.delete(
            inference_id=name, force=True
        )


@pytest.fixture
def backend(
    fake: FakeEmbedder, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[VectorStore, EmbedParams]]:
    monkeypatch.setenv("VDB_BACKEND", "elasticsearch")
    fake.reset()
    yield from _elasticsearch(fake)


def _write(store: VectorStore, params: EmbedParams, rows: list[dict]) -> int:
    return write_embedded_rows(store, embed_rows(rows, params, what="contract"))


def _search(
    store: VectorStore,
    params: EmbedParams,
    text: str,
    top_k: int = 3,
    **filters: str,
) -> list[dict]:
    where = _build_metadata_where_clause(
        labels=[filters["label"]] if "label" in filters else None,
        database_name=filters.get("database_name"),
        fmt="dict",
    )
    return Retriever(store, params).query(
        text, top_k=top_k, vdb_kwargs={"where": where} if where else None
    )


def _ids(hits: list[dict]) -> list[str]:
    return [hit["metadata"]["id"] for hit in hits]


def test_the_nearest_row_comes_first_at_cosine_distance(
    backend: tuple[VectorStore, EmbedParams],
) -> None:
    store, params = backend
    assert _write(store, params, _rows()) == len(_ROWS)

    hits = _search(store, params, "customer email")

    assert _ids(hits)[0] == "c_email"
    expected = 1 - sum(
        a * b for a, b in zip(embed("customer email"), embed(_ROWS[0][3]))
    )
    # The index quantises (BBQ with rescoring), so close, not equal.
    assert hits[0]["_distance"] == pytest.approx(expected, abs=0.02)
    assert hits[0]["text"] == "customer email address"
    assert hits[0]["metadata"]["label"] == "Column"


def test_rows_are_passages_and_searches_are_queries(
    backend: tuple[VectorStore, EmbedParams], fake: FakeEmbedder
) -> None:
    store, params = backend
    _write(store, params, _rows())
    write_types = set(fake.input_types())
    _search(store, params, "customer email")

    assert write_types == {"passage"}
    assert fake.input_types()[-1] == "query"
    # And Elasticsearch embedded them, not the app.
    assert {call.model for call in fake.calls} == {_ES_MODEL}


def test_filters_narrow_by_label_and_database(
    backend: tuple[VectorStore, EmbedParams],
) -> None:
    store, params = backend
    _write(store, params, _rows())

    tables = _search(store, params, "customer", label="Table")
    chinook = _search(store, params, "address", database_name="chinook")

    assert _ids(tables) == ["t_customer"]
    assert _ids(chinook) == ["c_billing"]


def test_deleting_by_id_removes_only_that_row(
    backend: tuple[VectorStore, EmbedParams],
) -> None:
    store, params = backend
    _write(store, params, _rows())

    assert store.delete_by_id("c_email") == 1

    assert "c_email" not in _ids(_search(store, params, "customer email", top_k=10))
    assert len(_search(store, params, "customer email", top_k=10)) == len(_ROWS) - 1


def test_deleting_everything_leaves_an_empty_searchable_store(
    backend: tuple[VectorStore, EmbedParams],
) -> None:
    store, params = backend
    _write(store, params, _rows())

    assert store.delete_all() == len(_ROWS)

    assert _search(store, params, "customer email") == []


def test_deleting_a_database_returns_the_deleted_rows_ids(
    backend: tuple[VectorStore, EmbedParams],
) -> None:
    # dal.reset counts them with len(); Elasticsearch once returned an int.
    store, params = backend
    _write(store, params, _rows())

    deleted = store.delete_by_database("pagila")

    assert isinstance(deleted, list)
    assert len(set(deleted)) == sum(1 for row in _ROWS if row[2] == "pagila")
    assert _ids(_search(store, params, "address", top_k=10)) == ["c_billing"]


def _replace(store: VectorStore, params: EmbedParams, node_id: str, text: str) -> int:
    """Re-embed one row as a catalog edit does (``_refresh_vdb_embeddings``)."""
    rows = _rows([(node_id, "Column", "pagila", text)])
    return replace_embedded_rows(
        store, [node_id], embed_rows(rows, params, what="contract")
    )


def test_replacing_a_row_swaps_it_for_the_new_text(
    backend: tuple[VectorStore, EmbedParams],
) -> None:
    store, params = backend
    _write(store, params, _rows())

    assert _replace(store, params, "c_email", "customer phone number") == 1

    hits = _search(store, params, "customer phone number", top_k=10)
    assert _ids(hits).count("c_email") == 1
    assert len(hits) == len(_ROWS)
    assert next(h for h in hits if h["metadata"]["id"] == "c_email")["text"] == (
        "customer phone number"
    )


def test_a_replacement_that_cannot_be_embedded_keeps_the_old_row(
    backend: tuple[VectorStore, EmbedParams], fake: FakeEmbedder
) -> None:
    # Elasticsearch embeds in the write, so the store has to write before it
    # deletes.
    store, params = backend
    _write(store, params, _rows())
    fake.fail_next(401, times=20)

    with pytest.raises(RuntimeError):
        _replace(store, params, "c_email", "customer phone number")
    fake.reset()

    hits = _search(store, params, "customer email", top_k=10)
    assert _ids(hits).count("c_email") == 1
    assert len(hits) == len(_ROWS)


def test_searching_an_empty_store_finds_nothing(
    backend: tuple[VectorStore, EmbedParams],
) -> None:
    store, params = backend

    assert _search(store, params, "customer email") == []


def test_a_reset_replaces_one_database_and_keeps_the_others(
    backend: tuple[VectorStore, EmbedParams],
) -> None:
    store, params = backend
    _write(store, params, _rows())

    assert isinstance(store, ElasticsearchVDB)
    fresh = ElasticsearchVDB(
        client=store._client,
        index_name=store.index_name,
        endpoint=store._endpoint,
        database_name="pagila",
        reset=True,
    )
    _write(fresh, params, _rows([("c_new", "Column", "pagila", "customer email")]))

    remaining = _ids(_search(store, params, "customer email address", top_k=10))
    assert sorted(remaining) == ["c_billing", "c_new"]


# ---------------------------------------------------------------------------
# Hybrid search
# ---------------------------------------------------------------------------

#: The same rows under column-style names. Hybrid's BM25 branch searches names
#: (split into words: ``BillingAddress`` matches "billing address"), so these
#: give it something to match.
_NAMED_ROWS = [
    # (id, label, database, name, text)
    ("c_email", "Column", "pagila", "customer_email", "customer email address"),
    ("c_rate", "Column", "pagila", "rental_rate", "film rental rate in dollars"),
    ("t_customer", "Table", "pagila", "customer", "customer table with names"),
    ("c_billing", "Column", "chinook", "BillingAddress", "invoice billing address"),
    ("c_length", "Column", "pagila", "length", "film length in minutes"),
]


def _named_rows() -> list[dict]:
    return [
        build_embed_row(
            text=text,
            node_id=node_id,
            label=label,
            name=name,
            schema_name="public",
            database_name=database,
        )
        for node_id, label, database, name, text in _NAMED_ROWS
    ]


@pytest.fixture
def hybrid(
    fake: FakeEmbedder, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[ElasticsearchVDB, EmbedParams]]:
    monkeypatch.setenv("VDB_BACKEND", "elasticsearch")
    monkeypatch.setenv("VDB_SEARCH_MODE", "hybrid")
    for store, params in _elasticsearch(fake):
        assert isinstance(store, ElasticsearchVDB)
        store.search_mode = "hybrid"
        _write(store, params, _named_rows())
        yield store, params


def _rows_for(
    store: VectorStore, params: EmbedParams, text: str, **kwargs: object
) -> list[dict]:
    """Search rows as the clarification flow gets them."""
    return search_semantic_index(Retriever(store, params), text, **kwargs)


def test_hybrid_puts_the_row_both_searches_rank_first_at_the_top(
    hybrid: tuple[ElasticsearchVDB, EmbedParams],
) -> None:
    store, params = hybrid

    [top, *_] = _rows_for(store, params, "customer email")

    assert top["id"] == "c_email"
    assert top["fused_score"] == pytest.approx(1.0)
    assert top["score"] == pytest.approx(0.0)


def test_a_hybrid_rows_vector_distance_is_the_semantic_searchs(
    hybrid: tuple[ElasticsearchVDB, EmbedParams],
) -> None:
    store, params = hybrid
    hybrid_rows = {r["id"]: r for r in _rows_for(store, params, "film rental")}

    store.search_mode = "semantic"
    semantic_rows = {r["id"]: r for r in _rows_for(store, params, "film rental")}

    for row_id, row in hybrid_rows.items():
        assert row["semantic_distance"] == pytest.approx(
            semantic_rows[row_id]["score"], abs=1e-6
        )


def test_the_hybrid_gate_reads_the_vector_distance_not_the_rank(
    hybrid: tuple[ElasticsearchVDB, EmbedParams],
) -> None:
    store, params = hybrid

    rows = {r["id"]: r for r in _rows_for(store, params, "film length")}
    top = min(rows.values(), key=lambda r: r["score"])

    # Ranked first, yet held to the same max_distance as a semantic hit.
    assert top["id"] == "c_length"
    assert passes_relevance_gate(top, max_distance=top["semantic_distance"])
    assert not passes_relevance_gate(top, max_distance=top["semantic_distance"] - 0.01)
    # Shares no word with the query, so only its vector distance can admit it.
    assert "lexical_score" not in rows["c_email"]
    assert passes_relevance_gate(rows["c_email"], max_distance=2.0)


def test_hybrid_filters_narrow_both_searches(
    hybrid: tuple[ElasticsearchVDB, EmbedParams],
) -> None:
    store, params = hybrid

    tables = _rows_for(store, params, "customer", label_filter=["Table"])
    chinook = _rows_for(store, params, "address", database_name="chinook")

    assert [r["id"] for r in tables] == ["t_customer"]
    assert [r["id"] for r in chinook] == ["c_billing"]


def test_a_hybrid_search_embeds_the_query_once_as_a_query(
    hybrid: tuple[ElasticsearchVDB, EmbedParams], fake: FakeEmbedder
) -> None:
    store, params = hybrid
    fake.reset()

    _rows_for(store, params, "customer email", label_filter=["Column"])

    assert fake.input_types() == ["query"]


def test_a_hybrid_search_of_an_empty_store_finds_nothing(
    fake: FakeEmbedder, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VDB_BACKEND", "elasticsearch")
    for store, params in _elasticsearch(fake):
        assert isinstance(store, ElasticsearchVDB)
        store.search_mode = "hybrid"
        assert _search(store, params, "customer email") == []
