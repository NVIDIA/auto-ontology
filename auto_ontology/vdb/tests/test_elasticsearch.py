# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the Elasticsearch store sends, checked without a cluster.

The behaviour that matters end to end -- ranking, filtering, deletes -- is in
``test_elasticsearch_cluster.py``, against a real cluster. These pin the parts
that a working cluster would not catch: a distance on the wrong scale still ranks
correctly, a retry loop that retries a bad key still eventually fails, and a
reset that fires before the first successful write still leaves a populated
index *most* of the time.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest
from elastic_transport import ApiResponseMeta, HttpHeaders, NodeConfig
from elasticsearch import ApiError, BadRequestError
from elasticsearch import ConnectionError as ESConnectionError
from elasticsearch import NotFoundError

from auto_ontology.infra.elasticsearch import (
    EmbedEndpoint,
    ensure_embed_endpoint,
    get_embed_endpoint,
)
from auto_ontology.retrieval.data_access.semantic_search import _is_retryable
from auto_ontology.utils.embedding_rows import build_embed_row
from auto_ontology.vdb.backend import vdb_search_mode
from auto_ontology.vdb.elasticsearch import ElasticsearchVDB, score_to_distance
from auto_ontology.vdb.records import to_vdb_records

_ENDPOINT = EmbedEndpoint(
    inference_id="ao-embed",
    model_id="nvidia/nemotron-3.5-embed-1b",
    api_key="nvapi-test",
    url="https://integrate.api.nvidia.com/v1/embeddings",
)


def _meta(status: int) -> ApiResponseMeta:
    return ApiResponseMeta(
        status=status,
        http_version="1.1",
        headers=HttpHeaders(),
        duration=0.0,
        node=NodeConfig("http", "localhost", 9200),
    )


def _not_found() -> NotFoundError:
    return NotFoundError("not found", _meta(404), {})


class _Namespace:
    def __init__(self, **methods: Any) -> None:
        self.__dict__.update(methods)


class FakeClient:
    """Records every call; ``bulk_statuses`` scripts each bulk's per-row status."""

    def __init__(self, *, index_exists: bool = True, endpoint: dict | None = None):
        self.calls: list[tuple[str, dict]] = []
        self.bulk_statuses: list[list[int]] = []
        self.search_hits: list[dict] = []
        self.index_missing_on_search = False
        self.default_similarity = "cosine"
        #: The ``error`` of each failed row, per bulk; a generic one if unset.
        self.bulk_errors: list[dict] = []
        self._index_exists = index_exists
        self._endpoint = endpoint
        self.indices = _Namespace(exists=self._exists, create=self._create)
        self.inference = _Namespace(get=self._get_endpoint, put=self._put_endpoint)
        self.esql = _Namespace(query=self._esql)
        #: ES|QL result rows: (_id, _score, lexical_score, semantic_score).
        self.esql_rows: list[tuple] = []
        self.esql_error: Exception | None = None
        #: _source per _id for mget; ids absent here come back not found.
        self.sources: dict[str, dict] = {}

    def options(self, **kwargs: Any) -> "FakeClient":
        return self

    def _exists(self, **kwargs: Any) -> bool:
        return self._index_exists

    def _create(self, **kwargs: Any) -> dict:
        self.calls.append(("create", kwargs))
        self._index_exists = True
        return {"acknowledged": True}

    def _get_endpoint(self, **kwargs: Any) -> dict:
        if self._endpoint is None:
            raise _not_found()
        return {"endpoints": [self._endpoint]}

    def _put_endpoint(self, **kwargs: Any) -> dict:
        self.calls.append(("put_endpoint", kwargs))
        settings = {
            "similarity": self.default_similarity,
            **kwargs["body"]["service_settings"],
        }
        self._endpoint = {**kwargs["body"], "service_settings": settings}
        return self._endpoint

    def bulk(self, **kwargs: Any) -> dict:
        self.calls.append(("bulk", kwargs))
        docs = kwargs["operations"][1::2]
        statuses = self.bulk_statuses.pop(0) if self.bulk_statuses else []
        statuses = statuses + [201] * (len(docs) - len(statuses))
        error = (
            self.bulk_errors.pop(0)
            if self.bulk_errors
            else {"type": "inference_exception"}
        )
        return {
            "items": [
                {"index": {"status": s} if s < 300 else {"status": s, "error": error}}
                for s in statuses
            ]
        }

    def delete_by_query(self, **kwargs: Any) -> dict:
        self.calls.append(("delete_by_query", kwargs))
        if not self._index_exists:
            raise _not_found()
        return {"deleted": 3}

    def search(self, **kwargs: Any) -> dict:
        self.calls.append(("search", kwargs))
        if self.index_missing_on_search:
            raise _not_found()
        return {"hits": {"hits": self.search_hits}}

    def _esql(self, **kwargs: Any) -> dict:
        self.calls.append(("esql", kwargs))
        if self.esql_error is not None:
            raise self.esql_error
        return {
            "columns": [
                {"name": n}
                for n in ("_id", "_score", "lexical_score", "semantic_score")
            ],
            "values": [list(row) for row in self.esql_rows],
        }

    def mget(self, **kwargs: Any) -> dict:
        self.calls.append(("mget", kwargs))
        return {
            "docs": [
                {"_id": i, "found": True, "_source": self.sources[i]}
                if i in self.sources
                else {"_id": i, "found": False}
                for i in kwargs["ids"]
            ]
        }

    def named(self, name: str) -> list[dict]:
        return [kwargs for call, kwargs in self.calls if call == name]


def _store(client: FakeClient, **kwargs: Any) -> tuple[ElasticsearchVDB, list[float]]:
    slept: list[float] = []
    store = ElasticsearchVDB(
        client=client,  # type: ignore[arg-type]
        # Unique per store: the module caches indices it has already checked,
        # and id(client) can repeat once an earlier client is collected.
        index_name=f"ao-test-{uuid.uuid4().hex}",
        endpoint=_ENDPOINT,
        sleep=slept.append,
        **kwargs,
    )
    return store, slept


def _records(*ids: str, database_name: str = "shop") -> list:
    rows = [
        build_embed_row(
            text=f"{node_id} table",
            node_id=node_id,
            label="Table",
            name=node_id,
            schema_name="public",
            database_name=database_name,
        )
        for node_id in ids
    ]
    for row in rows:
        row["metadata"]["embedding"] = [0.1, 0.2]
    return to_vdb_records(rows)


# ---------------------------------------------------------------------------
# Index and endpoint
# ---------------------------------------------------------------------------


def test_the_index_embeds_through_the_endpoint_and_keeps_text_lexical() -> None:
    client = FakeClient(index_exists=False, endpoint=None)
    store, _ = _store(client)

    store.create_index()

    [create] = client.named("create")
    properties = create["mappings"]["properties"]
    assert properties["content"] == {
        "type": "semantic_text",
        "inference_id": "ao-embed",
    }
    assert properties["text"] == {"type": "text", "copy_to": "content"}
    meta = properties["metadata"]["properties"]
    assert all(meta[k] == {"type": "keyword"} for k in ("id", "label", "database_name"))


def test_the_endpoint_is_unchunked_and_truncating() -> None:
    client = FakeClient(endpoint=None)

    ensure_embed_endpoint(client, _ENDPOINT)  # type: ignore[arg-type]

    [put] = client.named("put_endpoint")
    body = put["body"]
    assert body["service"] == "nvidia"
    # Left to the service: setting it crashes 9.5's nvidia service.
    assert "similarity" not in body["service_settings"]
    assert body["service_settings"]["url"] == _ENDPOINT.url
    assert body["chunking_settings"] == {"strategy": "none"}
    assert body["task_settings"] == {"truncate": "END"}


@pytest.mark.parametrize("similarity", ["cosine", "dot_product"])
def test_a_cosine_scored_similarity_is_accepted(similarity: str) -> None:
    client = FakeClient(endpoint=None)
    client.default_similarity = similarity

    ensure_embed_endpoint(client, _ENDPOINT)  # type: ignore[arg-type]


def test_an_l2_scored_endpoint_is_refused() -> None:
    # 1 / (1 + d^2) is not (1 + cos) / 2: every distance would be wrong.
    client = FakeClient(endpoint=None)
    client.default_similarity = "l2_norm"

    with pytest.raises(RuntimeError, match="l2_norm"):
        ensure_embed_endpoint(client, _ENDPOINT)  # type: ignore[arg-type]


def test_an_endpoint_on_another_model_is_refused_not_reused() -> None:
    client = FakeClient(
        endpoint={
            "service_settings": {
                "model_id": "nvidia/some-other-embedder",
                "similarity": "cosine",
            }
        }
    )

    with pytest.raises(RuntimeError, match="some-other-embedder"):
        ensure_embed_endpoint(client, _ENDPOINT)  # type: ignore[arg-type]
    assert client.named("put_endpoint") == []


def test_the_endpoint_url_is_the_full_embeddings_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EMBED_ENDPOINT", "https://integrate.api.nvidia.com/v1/")
    monkeypatch.setenv("EMBED_MODEL", "nvidia/nemotron-3.5-embed-1b")
    monkeypatch.delenv("ES_EMBED_URL", raising=False)

    endpoint = get_embed_endpoint("ao-embed")

    assert endpoint.url == "https://integrate.api.nvidia.com/v1/embeddings"


def test_a_model_that_needs_a_text_prefix_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EMBED_MODEL", "nvidia/nvidia/nemotron-3.5-embed-1b-nvfp4")

    with pytest.raises(ValueError, match="prefix"):
        get_embed_endpoint("ao-embed")


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------


def test_rows_are_written_as_text_without_their_vectors() -> None:
    client = FakeClient()
    store, _ = _store(client)

    assert store.run(_records("t1")) == 1

    [bulk] = client.named("bulk")
    [doc] = bulk["operations"][1::2]
    assert doc["text"] == "t1 table"
    assert "embedding" not in doc["metadata"]
    assert doc["metadata"]["id"] == "t1"
    assert doc["metadata"]["document_type"] == "text"


def test_only_the_last_batch_waits_for_visibility() -> None:
    client = FakeClient()
    store, _ = _store(client)

    store.write_to_index(_records("a", "b", "c"), batch_size=2)

    assert [b["refresh"] for b in client.named("bulk")] == [False, "wait_for"]


def test_a_rate_limited_row_is_retried_alone_after_a_wait() -> None:
    client = FakeClient()
    client.bulk_statuses = [[201, 429]]
    store, slept = _store(client)

    assert store.run(_records("a", "b")) == 2

    first, second = client.named("bulk")
    assert len(first["operations"]) == 4
    assert [d["metadata"]["id"] for d in second["operations"][1::2]] == ["b"]
    assert slept == [15.0]


def test_an_upstream_server_error_reported_as_400_is_retried() -> None:
    client = FakeClient()
    client.bulk_statuses = [[400]]
    client.bulk_errors = [
        {
            "type": "inference_exception",
            "caused_by": {
                "type": "status_exception",
                "reason": "Received a server error status code for request "
                "from inference entity id [ao-embed] status [500].",
            },
        }
    ]
    store, slept = _store(client)

    assert store.run(_records("a")) == 1
    assert len(client.named("bulk")) == 2
    assert slept == [15.0]


def test_a_rejected_key_is_not_retried_and_the_write_fails() -> None:
    client = FakeClient()
    client.bulk_statuses = [[401, 401]]
    store, slept = _store(client)

    with pytest.raises(RuntimeError, match="0/2"):
        store.run(_records("a", "b"))
    assert len(client.named("bulk")) == 1
    assert slept == []


def test_a_reset_waits_for_the_first_successful_write() -> None:
    client = FakeClient()
    client.bulk_statuses = [[401]]
    store, _ = _store(client, database_name="shop", reset=True)

    with pytest.raises(RuntimeError):
        store.run(_records("a"))
    assert client.named("delete_by_query") == [], "a failed write must not reset"

    store.run(_records("b"))

    [delete] = client.named("delete_by_query")
    query = delete["query"]["bool"]
    assert query["filter"] == [{"term": {"metadata.database_name": "shop"}}]
    generation = client.named("bulk")[-1]["operations"][1]["generation"]
    assert query["must_not"] == [{"term": {"generation": generation}}]

    store.run(_records("c"))
    assert len(client.named("delete_by_query")) == 1, "reset only once"


# ---------------------------------------------------------------------------
# Delete and search
# ---------------------------------------------------------------------------


def test_deletes_on_a_missing_index_delete_nothing() -> None:
    store, _ = _store(FakeClient(index_exists=False))

    assert store.delete_by_id("t1") == 0
    assert store.delete_all() == 0


def test_delete_by_id_matches_the_metadata_id() -> None:
    client = FakeClient()
    store, _ = _store(client)

    store.delete_by_id("t1")

    [delete] = client.named("delete_by_query")
    assert delete["query"] == {"term": {"metadata.id": "t1"}}
    assert delete["refresh"] is True


def test_search_filters_inside_knn_and_returns_cosine_distance() -> None:
    client = FakeClient()
    client.search_hits = [
        {"_score": 0.9, "_source": {"text": "orders", "metadata": {"id": "t1"}}}
    ]
    store, _ = _store(client)

    [[hit]] = store.retrieval(
        ["orders"],
        top_k=3,
        where={"label": ["Table", "Column"], "database_name": "shop"},
    )

    [search] = client.named("search")
    knn = search["query"]["knn"]
    assert knn["query_vector_builder"] == {"text_embedding": {"model_text": "orders"}}
    assert knn["k"] == 3
    assert knn["filter"] == [
        {"terms": {"metadata.label": ["Table", "Column"]}},
        {"term": {"metadata.database_name": "shop"}},
    ]
    assert hit["metadata"] == {"id": "t1"}
    assert hit["_distance"] == pytest.approx(0.2)


@pytest.mark.parametrize(
    "score, distance", [(1.0, 0.0), (0.5, 1.0), (0.0, 2.0), (0.9, 0.2)]
)
def test_scores_map_onto_pgvector_cosine_distance(
    score: float, distance: float
) -> None:
    # (1 + cos) / 2 -> 1 - cos: identical, orthogonal, opposite.
    assert score_to_distance(score) == pytest.approx(distance)


def test_searching_before_anything_was_ingested_finds_nothing() -> None:
    client = FakeClient()
    client.index_missing_on_search = True
    store, _ = _store(client)

    assert store.retrieval(["a", "b"]) == [[], []]


@pytest.mark.parametrize(
    "exc, retryable",
    [
        (ApiError("bad gateway", _meta(502), {}), True),
        (ApiError("rate limited", _meta(429), {}), True),
        (ApiError("bad request", _meta(400), {}), False),
        # How a 5xx from NVIDIA arrives: a 400 quoting the real status.
        (
            ApiError(
                "status_exception",
                _meta(400),
                {
                    "error": {
                        "reason": "Received a server error status code for "
                        "request from inference entity id [x] status [500]."
                    }
                },
            ),
            True,
        ),
        (
            ApiError(
                "status_exception",
                _meta(400),
                {"error": {"reason": "... inference entity id [x] status [401]."}},
            ),
            False,
        ),
        (ESConnectionError("refused"), True),
    ],
)
def test_cluster_errors_are_classified_for_the_query_retry(
    exc: Exception, retryable: bool
) -> None:
    assert _is_retryable(exc) is retryable


# ---------------------------------------------------------------------------
# Hybrid search
# ---------------------------------------------------------------------------


def _hybrid_store(client: FakeClient) -> ElasticsearchVDB:
    store, _ = _store(client, search_mode="hybrid")
    return store


def test_hybrid_forks_bm25_and_vector_search_and_fuses_on_the_server() -> None:
    client = FakeClient()
    store = _hybrid_store(client)

    store.retrieval(["film length"], top_k=3, where={"label": "Column"})

    [call] = client.named("esql")
    query = call["query"]
    # BM25 reads names, not descriptions; the vector search reads the text.
    assert "MATCH(name_text, ?query)" in query and "MATCH(content, ?query)" in query
    assert (
        '| FUSE LINEAR WITH {"normalizer": "minmax", '
        '"weights": {"fork1": 0.3, "fork2": 0.7}}' in query
    )
    # The filter is on both branches, as parameters.
    assert query.count("AND metadata.label == ?f0") == 2
    assert {"query": "film length"} in call["params"]
    assert {"f0": "Column"} in call["params"]
    assert {"top_k": 3} in call["params"]
    # Each branch contributes more than top_k rows to the fusion.
    assert {"window": 50} in call["params"]


def test_hybrid_filters_on_several_values_with_in() -> None:
    client = FakeClient()
    store = _hybrid_store(client)

    store.retrieval(
        ["x"], where={"label": ["Table", "Column"], "database_name": "shop"}
    )

    query = client.named("esql")[0]["query"]
    assert "metadata.label IN (?f0, ?f1)" in query
    assert "metadata.database_name == ?f2" in query


def test_hybrid_refuses_to_put_an_unindexed_key_into_the_query() -> None:
    store = _hybrid_store(FakeClient())

    with pytest.raises(ValueError, match="content_metadata"):
        store.retrieval(["x"], where={"content_metadata": "y"})


def test_hybrid_hits_carry_the_fused_score_and_each_branch_score() -> None:
    client = FakeClient()
    client.esql_rows = [
        ("d1", 1.0, 4.2, 0.9),  # first in both
        ("d2", 0.35, None, 0.6),  # only the vector search found it
    ]
    client.sources = {
        "d1": {"text": "film length", "metadata": {"id": "c1", "extra": 1}},
        "d2": {"text": "film rate", "metadata": {"id": "c2"}},
    }
    store = _hybrid_store(client)

    [[first, second]] = store.retrieval(["film length"], top_k=2)

    assert first["metadata"] == {"id": "c1", "extra": 1}, "full metadata, from mget"
    assert first["_fused_score"] == pytest.approx(1.0)
    assert first["_distance"] == pytest.approx(0.0)
    assert first["_lexical_score"] == 4.2
    assert first["_semantic_distance"] == pytest.approx(0.2)
    assert second["_fused_score"] == pytest.approx(0.35)
    assert second["_distance"] == pytest.approx(0.65)
    assert second["_lexical_score"] is None
    assert second["_semantic_distance"] == pytest.approx(0.8)


def test_a_hybrid_row_deleted_before_its_metadata_was_read_is_dropped() -> None:
    client = FakeClient()
    client.esql_rows = [("gone", 1.0, 1.0, 0.9), ("d2", 0.35, None, 0.6)]
    client.sources = {"d2": {"text": "t", "metadata": {"id": "c2"}}}
    store = _hybrid_store(client)

    [hits] = store.retrieval(["x"])

    assert [h["metadata"]["id"] for h in hits] == ["c2"]


def test_a_hybrid_search_before_anything_was_ingested_finds_nothing() -> None:
    client = FakeClient()
    client.esql_error = BadRequestError(
        "verification_exception", _meta(400), {"error": "Unknown index [ao-x]"}
    )
    store = _hybrid_store(client)

    assert store.retrieval(["x"]) == [[]]
    assert client.named("mget") == []


def test_the_lexical_weight_sets_the_fusion_weights() -> None:
    client = FakeClient()
    store, _ = _store(client, search_mode="hybrid", lexical_weight=0.5)

    store.retrieval(["film length"])

    assert '"weights": {"fork1": 0.5, "fork2": 0.5}' in client.named("esql")[0]["query"]


def test_a_lexical_weight_outside_zero_to_one_is_refused() -> None:
    with pytest.raises(ValueError, match="lexical_weight"):
        _store(FakeClient(), search_mode="hybrid", lexical_weight=1.5)


def test_names_are_indexed_split_into_words() -> None:
    client = FakeClient(index_exists=False)
    store, _ = _store(client)

    store.create_index()

    [call] = client.named("create")
    assert call["mappings"]["properties"]["metadata"]["properties"]["name"] == {
        "type": "keyword",
        "copy_to": "name_text",
    }
    analyzer = call["settings"]["analysis"]["analyzer"]["name_words"]
    assert analyzer["filter"] == ["name_parts", "lowercase"]


def test_hybrid_search_needs_elasticsearch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VDB_SEARCH_MODE", "hybrid")
    monkeypatch.setenv("VDB_BACKEND", "postgres")

    with pytest.raises(ValueError, match="requires VDB_BACKEND=elasticsearch"):
        vdb_search_mode()

    monkeypatch.setenv("VDB_BACKEND", "elasticsearch")
    assert vdb_search_mode() == "hybrid"
