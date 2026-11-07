# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Elasticsearch embedding through the real NVIDIA endpoint, end to end.

``test_contract.py`` proves the plumbing against a fake embedder; this proves
the one thing it cannot -- that the ``nvidia`` inference service, given the
app's ``EMBED_*`` settings, produces vectors that rank a real question
sensibly. A model that ignored ``input_type`` would still return vectors, and
only a ranking like this one would notice.

Opt-in, because it needs a key and a network: set
``AUTO_ONTOLOGY_LIVE_NVIDIA_ES=1`` along with ``ES_URL`` (and credentials) and
``EMBED_API_KEY``.
"""

from __future__ import annotations

import os
import uuid
from typing import Iterator

import pytest

from auto_ontology.infra.elasticsearch import (
    get_elasticsearch_client,
    get_elasticsearch_config,
    get_embed_endpoint,
)
from auto_ontology.utils.embedding import get_embed_params
from auto_ontology.utils.embedding_rows import build_embed_row
from auto_ontology.utils.retriever import Retriever
from auto_ontology.vdb.elasticsearch import ElasticsearchVDB
from auto_ontology.vdb.records import write_embedded_rows

pytestmark = pytest.mark.skipif(
    os.environ.get("AUTO_ONTOLOGY_LIVE_NVIDIA_ES", "").lower() not in ("1", "true"),
    reason="set AUTO_ONTOLOGY_LIVE_NVIDIA_ES=1 to embed through the real endpoint",
)


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> Iterator[ElasticsearchVDB]:
    monkeypatch.setenv("VDB_BACKEND", "elasticsearch")
    config = get_elasticsearch_config()
    client = get_elasticsearch_client(config)
    name = f"ao-live-{uuid.uuid4().hex[:8]}"
    try:
        yield ElasticsearchVDB(
            client=client, index_name=name, endpoint=get_embed_endpoint(name)
        )
    finally:
        client.options(ignore_status=404).indices.delete(index=name)
        client.options(ignore_status=404).inference.delete(
            inference_id=name, force=True
        )


# One parametrised test rather than two: CI's skip guard counts skip *lines*,
# and pytest reports a parametrised test's skips as one.
@pytest.mark.parametrize(
    "mode, question",
    [("semantic", "how long is the movie"), ("hybrid", "running time of the film")],
)
def test_a_real_question_finds_the_column_that_answers_it(
    store: ElasticsearchVDB, mode: str, question: str
) -> None:
    store.search_mode = mode  # type: ignore[assignment]
    rows = [
        build_embed_row(
            text=text,
            node_id=node_id,
            label="Column",
            name=node_id,
            schema_name="public",
            database_name="pagila",
        )
        for node_id, text in [
            ("email", "customer.email: the customer's contact email address"),
            ("rate", "film.rental_rate: price charged to rent the film"),
            ("length", "film.length: running time of the film in minutes"),
        ]
    ]
    assert write_embedded_rows(store, rows) == 3

    hits = Retriever(store, get_embed_params()).query(question, top_k=3)

    assert hits[0]["metadata"]["id"] == "length"
    assert 0.0 <= hits[0]["_distance"] < hits[-1]["_distance"]
    if mode == "semantic":
        assert hits[-1]["_distance"] <= 2.0
    else:
        # Words and meaning agree on this one: first in both searches.
        assert hits[0]["_rrf"] == pytest.approx(1.0)
        assert 0.0 <= hits[0]["_semantic_distance"] < 1.0
