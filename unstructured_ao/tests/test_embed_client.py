# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Request shaping and row alignment, without a gateway."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from unstructured_ao.embed_client import EmbedClient, post_with_retry


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


def test_figures_are_read_by_the_vision_triplet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A text-only ``MODEL_NAME`` 400s on images, silently dropping every figure."""
    from unstructured_ao import embed_client

    monkeypatch.setenv("UNSTRUCTURED_VISION_API_KEY", "nvapi-test")
    for name in ("UNSTRUCTURED_VISION_ENDPOINT", "UNSTRUCTURED_VISION_MODEL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("BASE_URL", "https://integrate.api.nvidia.com/v1")
    monkeypatch.setenv("MODEL_NAME", "nvidia/text-only")
    sent: dict[str, Any] = {}

    def fake_post(_http: Any, url: str, **kwargs: Any) -> dict[str, Any]:
        sent.update(url=url, **kwargs)
        return {"choices": [{"message": {"content": "a chart"}}]}

    monkeypatch.setattr(embed_client, "post_with_retry", fake_post)
    client = object.__new__(EmbedClient)
    client._http = None

    assert client.describe_image("read this", "QUJD") == "a chart"
    assert sent["url"] == "https://integrate.api.nvidia.com/v1/chat/completions"
    assert sent["json"]["model"] == "nvidia/nemotron-3.5-super-vl-120b-a12b"
    assert sent["headers"] == {"Authorization": "Bearer nvapi-test"}


def test_no_crops_sends_no_request() -> None:
    client = _vision_client({})
    assert client.describe_images("read this", []) == []


class _Http:
    """An ``httpx.Client`` stand-in answering with one status code per call."""

    def __init__(
        self,
        statuses: list[int | Exception],
        headers: dict[str, str] | None = None,
    ) -> None:
        self._statuses = statuses
        self._headers = headers or {}
        self.calls = 0

    def post(self, url: str, **_kwargs: Any) -> httpx.Response:
        status = self._statuses[self.calls]
        self.calls += 1
        if isinstance(status, Exception):
            raise status
        return httpx.Response(
            status,
            json={"ok": status},
            headers=self._headers,
            request=httpx.Request("POST", url),
        )


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    import time

    slept: list[float] = []
    monkeypatch.setattr(time, "sleep", slept.append)
    return slept


@pytest.mark.usefixtures("no_sleep")
@pytest.mark.parametrize("status", [429, 500, 502, 503])
def test_transient_failures_are_retried(status: int) -> None:
    http = _Http([status, 200])
    assert post_with_retry(http, "https://g/v1/embeddings") == {"ok": 200}
    assert http.calls == 2


@pytest.mark.usefixtures("no_sleep")
def test_a_rejected_request_is_not_retried() -> None:
    http = _Http([400, 200])
    with pytest.raises(httpx.HTTPStatusError):
        post_with_retry(http, "https://g/v1/embeddings")
    assert http.calls == 1


@pytest.mark.usefixtures("no_sleep")
def test_retries_give_up_and_raise() -> None:
    from auto_ontology.utils.llm_invoke import RETRY_MAX_ATTEMPTS

    http = _Http([429] * RETRY_MAX_ATTEMPTS)
    with pytest.raises(httpx.HTTPStatusError):
        post_with_retry(http, "https://g/v1/embeddings")
    assert http.calls == RETRY_MAX_ATTEMPTS


@pytest.mark.usefixtures("no_sleep")
def test_a_failed_connect_is_retried() -> None:
    http = _Http([httpx.ConnectError("refused"), 200])
    assert post_with_retry(http, "https://g/v1/embeddings") == {"ok": 200}
    assert http.calls == 2


@pytest.mark.usefixtures("no_sleep")
def test_a_read_timeout_is_not_retried() -> None:
    """The server likely did the work; a repeat pays for it twice."""
    http = _Http([httpx.ReadTimeout("timed out"), 200])
    with pytest.raises(httpx.ReadTimeout):
        post_with_retry(http, "https://g/v1/embeddings")
    assert http.calls == 1


@pytest.mark.parametrize(("header", "expected"), [("7", 7.0), ("3600", 60.0)])
def test_retry_after_is_honoured_and_capped(
    no_sleep: list[float], header: str, expected: float
) -> None:
    http = _Http([429, 200], headers={"Retry-After": header})
    post_with_retry(http, "https://g/v1/embeddings")
    assert no_sleep == [expected]


def test_a_dated_retry_after_falls_back_to_backoff(no_sleep: list[float]) -> None:
    http = _Http([429, 200], headers={"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"})
    post_with_retry(http, "https://g/v1/embeddings")
    assert len(no_sleep) == 1 and 2.0 <= no_sleep[0] < 3.0
