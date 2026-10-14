# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for :mod:`auto_ontology.utils.embedding`.

The endpoint is replaced by an ``httpx.MockTransport`` under the real OpenAI
client, so what is asserted is the request that actually goes over the wire:
the NVIDIA fields riding in the body, the auth header, batching, ordering and
which failures are retried.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable

import httpx
import pandas as pd
import pytest
from openai import AuthenticationError, OpenAI, RateLimitError

from auto_ontology.utils import embedding
from auto_ontology.utils.embedding import (
    EmbedParams,
    _embed_frame,
    _is_fatal,
    _embed_with_retry,
    batch_embed,
    embed_rows,
    embed_texts,
)

Handler = Callable[[httpx.Request], httpx.Response]


def _params(**overrides: Any) -> EmbedParams:
    fields: dict[str, Any] = {
        "model_name": "nvidia/embed",
        "embed_invoke_url": "https://embed.example/v1",
        "api_key": "sk-test",
    }
    fields.update(overrides)
    return EmbedParams(**fields)


def _vector(text: str) -> list[float]:
    """A distinct, recognisable vector per input text."""
    return [float(len(text)), float(ord(text[0]))]


def _ok(request: httpx.Request, *, reverse: bool = False) -> httpx.Response:
    body = json.loads(request.content)
    data = [
        {"object": "embedding", "index": i, "embedding": _vector(text)}
        for i, text in enumerate(body["input"])
    ]
    if reverse:
        data.reverse()
    return httpx.Response(200, json={"object": "list", "data": data, "model": "m"})


@pytest.fixture
def endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[Handler], list[httpx.Request]]:
    """Route the module's client through *handler*; returns the requests seen."""

    def _install(handler: Handler) -> list[httpx.Request]:
        seen: list[httpx.Request] = []

        def _record(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return handler(request)

        def _client(params: EmbedParams) -> OpenAI:
            transport = httpx.MockTransport(_record)
            return embedding._new_client(params, httpx.Client(transport=transport))

        monkeypatch.setattr(embedding, "_client", _client)
        return seen

    return _install


# ---------------------------------------------------------------------------
# The request
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("input_type", ["passage", "query"])
def test_the_request_carries_the_nvidia_fields(endpoint, input_type: str) -> None:
    seen = endpoint(_ok)

    vectors = embed_texts(["  orders  "], _params(), input_type)

    assert vectors == [_vector("orders")]
    [request] = seen
    assert request.url.path == "/v1/embeddings"
    assert request.headers["authorization"] == "Bearer sk-test"
    assert json.loads(request.content) == {
        "model": "nvidia/embed",
        "input": ["orders"],
        "encoding_format": "float",
        "input_type": input_type,
        "truncate": "END",
        "modality": "text",
    }


@pytest.mark.parametrize(
    "configured",
    [
        "https://embed.example/v1",
        "https://embed.example/v1/",
        "https://embed.example/v1/embeddings",
    ],
)
def test_both_endpoint_spellings_reach_the_same_url(endpoint, configured: str) -> None:
    seen = endpoint(_ok)
    embed_texts(["x"], _params(embed_invoke_url=configured), "passage")
    assert str(seen[0].url) == "https://embed.example/v1/embeddings"


def test_no_key_sends_no_authorization_header(endpoint) -> None:
    seen = endpoint(_ok)
    embed_texts(["x"], _params(api_key=""), "passage")
    assert "authorization" not in seen[0].headers


def test_the_key_stays_out_of_repr() -> None:
    assert "sk-test" not in repr(_params())


def test_batches_and_reassembles_in_input_order(endpoint) -> None:
    """Out-of-order ``data`` and concurrent batches still line up with the input."""
    seen = endpoint(lambda request: _ok(request, reverse=True))
    texts = ["a", "bb", "ccc", "dddd", "eeeee"]

    vectors = embed_texts(texts, _params(inference_batch_size=2), "passage")

    assert vectors == [_vector(t) for t in texts]
    assert sorted(len(json.loads(r.content)["input"]) for r in seen) == [1, 2, 2]


def test_blank_texts_are_not_sent(endpoint) -> None:
    seen = endpoint(_ok)

    vectors = embed_texts(["a", "  ", None, "b"], _params(), "passage")  # type: ignore[list-item]

    assert vectors == [_vector("a"), None, None, _vector("b")]
    assert json.loads(seen[0].content)["input"] == ["a", "b"]


def test_nothing_to_send_makes_no_request(endpoint) -> None:
    seen = endpoint(_ok)
    assert embed_texts(["", " "], _params(), "passage") == [None, None]
    assert seen == []


def test_a_rate_limit_is_retried_by_the_client(endpoint) -> None:
    responses = [
        httpx.Response(429, headers={"retry-after-ms": "1"}, json={"error": "slow"}),
    ]

    def _handler(request: httpx.Request) -> httpx.Response:
        return responses.pop(0) if responses else _ok(request)

    seen = endpoint(_handler)
    assert embed_texts(["a"], _params(), "passage") == [_vector("a")]
    assert len(seen) == 2


def test_retries_zero_leaves_retrying_to_the_caller(endpoint) -> None:
    seen = endpoint(
        lambda request: httpx.Response(
            429, headers={"retry-after-ms": "1"}, json={"error": "slow"}
        )
    )
    with pytest.raises(RateLimitError):
        embed_texts(["a"], _params(), "query", retries=0)
    assert len(seen) == 1


def test_the_timeout_reaches_the_request(endpoint) -> None:
    seen = endpoint(_ok)
    embed_texts(["a"], _params(), "query", timeout=7.0)
    assert seen[0].extensions["timeout"]["read"] == 7.0


@pytest.mark.parametrize(
    "status, fatal",
    [
        (400, True),
        (401, True),
        (403, True),
        (404, True),
        (422, True),
        (408, False),
        (409, False),
        (429, False),
        (500, False),
        (503, False),
    ],
)
def test_only_a_wrong_request_is_fatal(status: int, fatal: bool) -> None:
    from openai import APIStatusError

    request = httpx.Request("POST", "https://embed.example/v1/embeddings")
    exc = APIStatusError(
        "x", response=httpx.Response(status, request=request), body=None
    )
    assert _is_fatal(exc) is fatal


def test_a_bad_key_fails_at_once(endpoint) -> None:
    seen = endpoint(lambda request: httpx.Response(401, json={"error": "bad key"}))
    with pytest.raises(AuthenticationError):
        embed_texts(["a"], _params(), "passage")
    assert len(seen) == 1


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------


def _rows(*texts: str) -> list[dict]:
    return [{"text": t, "metadata": {"id": t}} for t in texts]


def test_embed_frame_writes_the_embedding_without_touching_the_input(endpoint) -> None:
    endpoint(_ok)
    rows = _rows("a", "b")

    out = _embed_frame(pd.DataFrame(rows), _params())

    assert [m["embedding"] for m in out["metadata"]] == [_vector("a"), _vector("b")]
    assert "embedding" not in rows[0]["metadata"]


def test_embed_frame_reports_a_failure_instead_of_raising(endpoint) -> None:
    endpoint(lambda request: httpx.Response(401, json={"error": "bad key"}))

    out = _embed_frame(pd.DataFrame(_rows("a")), _params())

    assert not any(m.get("embedding") for m in out["metadata"])
    assert "AuthenticationError" in out.attrs["embed_error"]
    assert out.attrs["embed_error_fatal"] is True


def test_empty_rows_short_circuit_without_calling_the_endpoint(endpoint) -> None:
    seen = endpoint(_ok)
    assert batch_embed([], _params()).empty
    assert batch_embed(pd.DataFrame(), _params()).empty
    assert seen == []


def test_batch_embed_returns_every_row(endpoint) -> None:
    endpoint(_ok)
    out = batch_embed(_rows("a", "b", "c"), _params())
    assert [m["id"] for m in out["metadata"]] == ["a", "b", "c"]
    assert all(m["embedding"] for m in out["metadata"])


def test_embed_rows_keeps_only_the_embedded_rows(endpoint) -> None:
    endpoint(_ok)
    out = embed_rows(_rows("a", " "), _params(), what="Thing")
    assert [r["metadata"]["id"] for r in out] == ["a"]


def test_embed_rows_raises_when_the_endpoint_fails(endpoint) -> None:
    endpoint(lambda request: httpx.Response(401, json={"error": "bad key"}))
    with pytest.raises(RuntimeError, match="Thing"):
        embed_rows(_rows("a"), _params(), what="Thing")


def test_embed_rows_is_bounded_for_interactive_callers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A save in a request handler must fail in about a minute, not an hour."""
    seen: list[dict] = []

    def _embed(texts: list[str], params: Any, input_type: str, **kw: Any) -> list:
        seen.append(kw)
        return [[0.5] for _ in texts]

    monkeypatch.setattr(embedding, "embed_texts", _embed)
    embed_rows(_rows("a"), _params(), what="Thing")
    assert seen == [{"timeout": 60.0, "retries": 1}]


def test_embed_rows_raises_when_nothing_embeds(endpoint) -> None:
    endpoint(_ok)
    with pytest.raises(RuntimeError, match="0/1 Thing"):
        embed_rows(_rows(" "), _params(), what="Thing")


# ---------------------------------------------------------------------------
# Chunk retry
# ---------------------------------------------------------------------------
#
# The endpoint rate-limits: a sustained ingest gets 429 once its quota is gone,
# for longer than the client's own retries wait. The embed step reports that as
# rows without embeddings rather than an exception, so without a chunk-level
# retry those rows are silently dropped and the catalog is quietly incomplete.

_RATE_LIMITED = "RateLimitError: Error code: 429 - Too Many Requests"


def _embedded(rows: int = 2) -> pd.DataFrame:
    return pd.DataFrame({"metadata": [{"embedding": [0.1, 0.2]}] * rows})


def _refused(error: str = _RATE_LIMITED, *, fatal: bool = False) -> pd.DataFrame:
    frame = pd.DataFrame({"metadata": [{}] * 2})
    frame.attrs["embed_error"] = error
    frame.attrs["embed_error_fatal"] = fatal
    return frame


def _part() -> pd.DataFrame:
    return pd.DataFrame({"text": ["a", "b"]})


def test_retry_recovers_a_rate_limited_chunk() -> None:
    """Two 429s then success must yield the embeddings, not an empty frame."""
    attempts = [_refused(), _refused(), _embedded()]
    slept: list[float] = []

    result = _embed_with_retry(
        lambda part: attempts.pop(0),
        _part(),
        label="chunk 1/1",
        sleep=slept.append,
    )

    assert result is not None and not result.empty
    assert attempts == []
    # Backoff must grow: an immediate retry just hits the same closed window.
    assert slept == [15.0, 30.0]


def test_retry_gives_up_and_reports_the_endpoint_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """After the last attempt the endpoint's own error must reach the log."""
    calls: list[int] = []

    def _always_refused(part: pd.DataFrame) -> pd.DataFrame:
        calls.append(1)
        return _refused()

    with caplog.at_level(logging.WARNING):
        result = _embed_with_retry(
            _always_refused,
            _part(),
            label="chunk 1/1",
            attempts=3,
            sleep=lambda _seconds: None,
        )

    assert len(calls) == 3
    assert result is not None
    text = "\n".join(record.getMessage() for record in caplog.records)
    assert "giving up after 3 attempt(s)" in text
    assert "429 - Too Many Requests" in text


def test_a_rejected_request_is_not_retried() -> None:
    """Waiting does not fix a wrong key; four minutes of backoff would only hide it."""
    calls: list[int] = []
    slept: list[float] = []

    def _unauthorised(part: pd.DataFrame) -> pd.DataFrame:
        calls.append(1)
        return _refused("AuthenticationError: 401", fatal=True)

    _embed_with_retry(_unauthorised, _part(), label="chunk 1/1", sleep=slept.append)

    assert calls == [1]
    assert slept == []


def test_first_attempt_success_does_not_sleep() -> None:
    """The happy path must not pay any backoff."""
    slept: list[float] = []

    result = _embed_with_retry(
        lambda part: _embedded(),
        _part(),
        label="chunk 1/1",
        sleep=slept.append,
    )

    assert result is not None and not result.empty
    assert slept == []
