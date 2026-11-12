# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import httpx
import openai
import pytest

from auto_ontology.utils import retriever as retriever_mod
from auto_ontology.utils.embedding import EmbedParams

_REQUEST = httpx.Request("POST", "http://embed/v1/embeddings")


def _status(cls: type, code: int) -> Exception:
    return cls(
        f"Error code: {code}",
        response=httpx.Response(code, request=_REQUEST),
        body=None,
    )


@pytest.mark.parametrize(
    "exc, transient",
    [
        (openai.APITimeoutError(request=_REQUEST), True),
        (_status(openai.InternalServerError, 502), True),
        (_status(openai.RateLimitError, 429), True),
        (_status(openai.AuthenticationError, 401), False),
        (ValueError("Cannot search for a blank query."), False),
        (RuntimeError("upstream said 503 Service Unavailable"), True),
    ],
    ids=["timeout", "502", "429", "401", "blank", "untyped-text"],
)
def test_classification(exc: Exception, transient: bool) -> None:
    assert retriever_mod._is_transient(exc) is transient


class _Store:
    def retrieval(self, vectors: list, top_k: int, where: Any) -> list[list[dict]]:
        return [[{"text": "orders", "metadata": {"id": "t1"}, "_distance": 0.1}]]


@pytest.fixture
def slept(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    waits: list[float] = []
    monkeypatch.setattr(retriever_mod.time, "sleep", waits.append)
    return waits


def _retriever(monkeypatch: pytest.MonkeyPatch, outcomes: list[Any]) -> Any:
    def _embed(texts: list[str], params: Any, input_type: str, **kw: Any) -> list:
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return [outcome]

    monkeypatch.setattr(retriever_mod, "embed_texts", _embed)
    return retriever_mod.Retriever(_Store(), EmbedParams("m", "http://embed"))


def test_query_retries_a_timeout(
    monkeypatch: pytest.MonkeyPatch, slept: list[float]
) -> None:
    outcomes = [openai.APITimeoutError(request=_REQUEST), [1.0, 0.0]]
    retriever = _retriever(monkeypatch, outcomes)

    hits = retriever.query("orders")

    assert [hit["metadata"]["id"] for hit in hits] == ["t1"]
    assert len(slept) == 1


def test_query_raises_once_attempts_run_out(
    monkeypatch: pytest.MonkeyPatch, slept: list[float]
) -> None:
    outcomes: list[Any] = [
        openai.APIConnectionError(request=_REQUEST) for _ in range(3)
    ]
    retriever = _retriever(monkeypatch, outcomes)

    with pytest.raises(openai.APIConnectionError):
        retriever.query("orders")
    assert len(slept) == 2


def test_query_does_not_retry_a_client_error(
    monkeypatch: pytest.MonkeyPatch, slept: list[float]
) -> None:
    outcomes = [_status(openai.AuthenticationError, 401), [1.0, 0.0]]
    retriever = _retriever(monkeypatch, outcomes)

    with pytest.raises(openai.AuthenticationError):
        retriever.query("orders")
    assert slept == []
