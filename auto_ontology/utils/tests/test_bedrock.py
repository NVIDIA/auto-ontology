# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``bedrock/`` embedding and rerank paths.

Bedrock is stubbed at the boto3 client, so these pin the request bodies each
model family needs and how the answers map back onto the inputs.
"""

from __future__ import annotations

import io
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from botocore.exceptions import ClientError

from auto_ontology.utils import bedrock, embedding, rerank
from auto_ontology.utils.embedding import EmbedParams
from auto_ontology.vdb.postgres import DEFAULT_VECTOR_SIZE

_COHERE = "bedrock/global.cohere.embed-v4:0"
_TITAN = "bedrock/amazon.titan-embed-text-v2:0"
_TOKYO_RERANK = "arn:aws:bedrock:ap-northeast-1::foundation-model/cohere.rerank-v3-5:0"


class _FakeBedrock:
    """Answers every text with ``[len(text), 1.0]``, and ranks longer text higher."""

    def __init__(self) -> None:
        self.meta = SimpleNamespace(region_name="ap-northeast-2")
        self.calls: list[Any] = []
        self.clients: list[tuple[str, str | None, float, int]] = []
        # Results per rerank response, the rest behind a nextToken; None sends all.
        self.rerank_page_size: int | None = None

    def invoke_model(self, *, modelId: str, body: str, **_: Any) -> dict[str, Any]:
        request = json.loads(body)
        self.calls.append(request)
        if "texts" in request:
            payload: dict[str, Any] = {
                "embeddings": {
                    "float": [[float(len(t)), 1.0] for t in request["texts"]]
                }
            }
        else:
            payload = {"embedding": [float(len(request["inputText"])), 1.0]}
        return {"body": io.BytesIO(json.dumps(payload).encode())}

    def rerank(self, **request: Any) -> dict[str, Any]:
        self.calls.append(request)
        texts = [
            s["inlineDocumentSource"]["textDocument"]["text"]
            for s in request["sources"]
        ]
        results = sorted(
            (
                {"index": i, "relevanceScore": float(len(t))}
                for i, t in enumerate(texts)
            ),
            key=lambda r: -r["relevanceScore"],
        )
        if self.rerank_page_size is None:
            return {"results": results}
        start = int(request.get("nextToken", "0"))
        end = start + self.rerank_page_size
        page: dict[str, Any] = {"results": results[start:end]}
        if end < len(results):
            page["nextToken"] = str(end)
        return page


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> _FakeBedrock:
    client = _FakeBedrock()

    def _client(
        service: str, region: str | None, *, timeout: float, attempts: int
    ) -> Any:
        client.clients.append((service, region, timeout, attempts))
        return client

    monkeypatch.setattr(bedrock, "_client", _client)
    return client


def test_the_prefix_routes_and_only_an_arn_names_a_region() -> None:
    assert bedrock.is_bedrock_model(_COHERE)
    assert not bedrock.is_bedrock_model("nvidia/nemotron-3.5-embed-1b")
    assert bedrock.model_id(_COHERE) == "global.cohere.embed-v4:0"
    assert bedrock.arn_region(_COHERE) is None
    assert bedrock.arn_region(f"bedrock/{_TOKYO_RERANK}") == "ap-northeast-1"


def test_clients_call_aws_region_like_the_chat_client(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """botocore skips ``AWS_REGION``; langchain-aws reads it before the profile."""
    config = tmp_path / "config"
    config.write_text("[default]\nregion = eu-west-1\n")
    monkeypatch.setenv("AWS_CONFIG_FILE", str(config))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "credentials"))
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.setattr(bedrock, "_clients", {})

    def region(arn_region: str | None) -> str:
        client = bedrock._client("bedrock-runtime", arn_region, timeout=5, attempts=1)
        return client.meta.region_name

    assert region(None) == "us-east-1"
    assert region("ap-northeast-1") == "ap-northeast-1"
    monkeypatch.delenv("AWS_REGION")
    assert region(None) == "eu-west-1"


def test_cohere_gets_typed_batches_of_at_most_96(fake: _FakeBedrock) -> None:
    texts = [f"t{i}" for i in range(100)]

    vectors = bedrock.embed(texts, _COHERE, "query", timeout=5, retries=2)

    assert vectors == [[float(len(t)), 1.0] for t in texts]
    assert [len(call["texts"]) for call in fake.calls] == [96, 4]
    assert {call["input_type"] for call in fake.calls} == {"search_query"}
    assert {call["output_dimension"] for call in fake.calls} == {1536}
    assert fake.clients == [("bedrock-runtime", None, 5, 3)]


def test_titan_embeds_one_text_per_request(fake: _FakeBedrock) -> None:
    vectors = bedrock.embed(["a", "bb"], _TITAN, "passage", timeout=5, retries=0)

    assert vectors == [[1.0, 1.0], [2.0, 1.0]]
    assert fake.calls == [
        {"inputText": "a", "dimensions": 1024, "normalize": True},
        {"inputText": "bb", "dimensions": 1024, "normalize": True},
    ]


def test_an_unsupported_embedding_model_fails_before_any_request(
    fake: _FakeBedrock,
) -> None:
    with pytest.raises(bedrock.UnsupportedModelError):
        bedrock.embed(
            ["a"], "bedrock/amazon.nova-embed-v1:0", "passage", timeout=5, retries=0
        )
    assert fake.clients == []
    assert embedding._is_fatal(bedrock.UnsupportedModelError("x"))


def test_embed_texts_pads_bedrock_vectors_to_the_store_width(
    fake: _FakeBedrock,
) -> None:
    params = EmbedParams(model_name=_COHERE, embed_invoke_url="ignored")

    vectors = embedding.embed_texts(["abc", "  ", "de"], params, "passage")

    assert vectors[1] is None
    assert [len(v or []) for v in (vectors[0], vectors[2])] == [DEFAULT_VECTOR_SIZE] * 2
    assert vectors[0][:2] == [3.0, 1.0] and set(vectors[0][2:]) == {0.0}
    assert fake.calls[0] == {
        "texts": ["abc", "de"],
        "input_type": "search_document",
        "embedding_types": ["float"],
        "output_dimension": 1536,
    }


@pytest.mark.parametrize(
    ("code", "status", "fatal"),
    [
        ("ValidationException", 400, True),
        ("AccessDeniedException", 403, True),
        ("ThrottlingException", 429, False),
        ("ServiceUnavailableException", 503, False),
    ],
)
def test_only_a_wrong_bedrock_request_is_fatal(
    code: str, status: int, fatal: bool
) -> None:
    exc = ClientError(
        {
            "Error": {"Code": code, "Message": "x"},
            "ResponseMetadata": {"HTTPStatusCode": status},
        },
        "InvokeModel",
    )
    assert embedding._is_fatal(exc) is fatal


def test_rerank_scores_by_index_and_skips_blank_documents(fake: _FakeBedrock) -> None:
    scores = bedrock.rerank_scores(
        "q", ["aaa", "  ", "b"], f"bedrock/{_TOKYO_RERANK}", timeout=30
    )

    assert scores == [3.0, -math.inf, 1.0]
    (request,) = fake.calls
    assert [
        s["inlineDocumentSource"]["textDocument"]["text"] for s in request["sources"]
    ] == [
        "aaa",
        "b",
    ]
    config = request["rerankingConfiguration"]["bedrockRerankingConfiguration"]
    assert config == {
        "numberOfResults": 2,
        "modelConfiguration": {"modelArn": _TOKYO_RERANK},
    }
    assert fake.clients == [("bedrock-agent-runtime", "ap-northeast-1", 30, 1)]


def test_rerank_follows_next_token_through_every_page(fake: _FakeBedrock) -> None:
    fake.rerank_page_size = 2

    scores = bedrock.rerank_scores(
        "q", ["a", "bbb", "cc", "dddd", " "], "bedrock/cohere.rerank-v3-5:0", timeout=30
    )

    assert scores == [1.0, 3.0, 2.0, 4.0, -math.inf]
    assert [call.get("nextToken") for call in fake.calls] == [None, "2"]
    assert fake.calls[0]["sources"] == fake.calls[1]["sources"]


def test_a_plain_rerank_id_gets_an_arn_in_the_configured_region(
    fake: _FakeBedrock,
) -> None:
    bedrock.rerank_scores("q", ["a"], "bedrock/cohere.rerank-v3-5:0", timeout=30)

    config = fake.calls[0]["rerankingConfiguration"]["bedrockRerankingConfiguration"]
    assert config["modelConfiguration"]["modelArn"] == (
        "arn:aws:bedrock:ap-northeast-2::foundation-model/cohere.rerank-v3-5:0"
    )


def test_rerank_hits_ranks_with_bedrock_and_ignores_the_url(fake: _FakeBedrock) -> None:
    hits = [{"id": "a", "text": "a"}, {"id": "b", "text": "bbb"}]

    ranked = rerank.rerank_hits(
        "q",
        hits,
        rerank_invoke_url="https://unused.example",
        model_name=f"bedrock/{_TOKYO_RERANK}",
    )

    assert [hit["id"] for hit in ranked] == ["b", "a"]
    assert ranked[0]["_rerank_score"] == 3.0
