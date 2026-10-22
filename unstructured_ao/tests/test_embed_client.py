# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Request shaping and row alignment, without a gateway."""

from __future__ import annotations

from typing import Any

import pytest

from unstructured_ao.embed_client import EmbedClient


def _client(captured: list[dict[str, Any]], vectors: list[list[float]]) -> EmbedClient:
    """An ``EmbedClient`` that records payloads instead of sending them."""
    client = object.__new__(EmbedClient)
    client._endpoint = "https://gateway/v1"
    client._api_key = "sk-test"
    client._model = "test-embed-model"

    def post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
        captured.append({"path": path, **payload})
        return {
            "data": [
                {"index": i, "embedding": vectors[i % len(vectors)]}
                for i in range(len(payload["input"]))
            ]
        }

    client._post = post  # type: ignore[method-assign]
    return client


def test_text_only_payload_matches_the_retrievers_shape() -> None:
    sent: list[dict[str, Any]] = []
    _client(sent, [[0.1, 0.2]]).embed(["posted workers"])
    assert sent[0]["path"] == "embeddings"
    assert sent[0]["input"] == ["posted workers"]
    assert sent[0]["modality"] == "text"
    assert sent[0]["input_type"] == "passage"
    # Not the retriever's "NONE": a page-grained row can exceed the model's
    # 4096 tokens, and that 422s the whole batch.
    assert sent[0]["truncate"] == "END"
    assert sent[0]["encoding_format"] == "float"


def test_text_and_image_are_one_newline_joined_string() -> None:
    """The retriever concatenates them rather than sending two fields."""
    sent: list[dict[str, Any]] = []
    _client(sent, [[0.1]]).embed(
        ["a chart of wages"], images_b64=["QUJD"], modality="text_image"
    )
    assert sent[0]["input"] == ["a chart of wages\ndata:image/png;base64,QUJD"]
    assert sent[0]["modality"] == "text_image"


def test_an_image_with_no_text_sends_only_the_data_url() -> None:
    sent: list[dict[str, Any]] = []
    _client(sent, [[0.1]]).embed([""], images_b64=["QUJD"], modality="image")
    assert sent[0]["input"] == ["data:image/png;base64,QUJD"]


def test_queries_are_encoded_as_queries() -> None:
    """The model encodes queries and passages asymmetrically."""
    sent: list[dict[str, Any]] = []
    _client(sent, [[0.3]]).embed_query("how many posted workers?")
    assert sent[0]["input_type"] == "query"


def test_empty_rows_keep_their_place_as_none() -> None:
    """The retriever drops them, which shifts every later vector onto the wrong row."""
    sent: list[dict[str, Any]] = []
    vectors = _client(sent, [[1.0], [2.0]]).embed(["first", "   ", "third"])
    assert len(vectors) == 3
    assert vectors[1] is None
    assert vectors[0] is not None and vectors[2] is not None
    assert sent[0]["input"] == ["first", "third"]


def test_everything_empty_sends_no_request() -> None:
    sent: list[dict[str, Any]] = []
    assert _client(sent, [[1.0]]).embed(["", "  "]) == [None, None]
    assert sent == []


def test_mismatched_image_count_is_refused() -> None:
    """Silently zipping to the shorter list is the alignment bug this avoids."""
    with pytest.raises(ValueError, match="one-to-one"):
        _client([], [[1.0]]).embed(["a", "b"], images_b64=["QUJD"])


def test_batches_are_split_and_reassembled_in_order() -> None:
    sent: list[dict[str, Any]] = []
    texts = [f"row {i}" for i in range(20)]
    vectors = _client(sent, [[float(i)] for i in range(20)]).embed(texts)
    assert len(sent) == 2, "20 rows should go in two requests of 16 and 4"
    assert [len(call["input"]) for call in sent] == [16, 4]
    assert all(vector is not None for vector in vectors)


def test_out_of_order_responses_are_placed_by_index() -> None:
    client = object.__new__(EmbedClient)
    client._endpoint, client._api_key, client._model = "https://g/v1", "k", "m"
    client._post = lambda path, payload: {  # type: ignore[method-assign]
        "data": [
            {"index": 1, "embedding": [2.0]},
            {"index": 0, "embedding": [1.0]},
        ]
    }
    assert client.embed(["first", "second"]) == [[1.0], [2.0]]


def _vision_client(replies: dict[str, str | Exception]) -> EmbedClient:
    """A client whose single-crop vision call is keyed by the crop's bytes."""
    client = object.__new__(EmbedClient)

    def describe_image(_prompt: str, b64: str) -> str:
        reply = replies[b64]
        if isinstance(reply, Exception):
            raise reply
        return reply

    client.describe_image = describe_image  # type: ignore[method-assign]
    return client


def test_many_crops_come_back_in_input_order() -> None:
    """Order carries the page number, so a shuffled result mislabels every row."""
    client = _vision_client({"a": "first", "b": "second", "c": "third"})
    assert client.describe_images("read this", ["a", "b", "c"]) == [
        "first",
        "second",
        "third",
    ]


def test_one_unreadable_crop_costs_only_its_own_row() -> None:
    client = _vision_client({"a": "chart text", "b": RuntimeError("gateway down")})
    assert client.describe_images("read this", ["a", "b"]) == ["chart text", ""]


def test_no_crops_sends_no_request() -> None:
    client = _vision_client({})
    assert client.describe_images("read this", []) == []
