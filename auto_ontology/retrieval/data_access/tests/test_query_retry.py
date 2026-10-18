# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A transient embedding failure is retried; a problem with the request is not.

Without the retry a 502 from the embedding endpoint silently turns one entity
search into zero candidates. The client's errors are classified by type: a
timeout's message ("Request timed out.") names none of the status tokens the
fallback matches on.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from openai import APITimeoutError, AuthenticationError, InternalServerError

from auto_ontology.retrieval.data_access import semantic_search
from auto_ontology.retrieval.data_access.semantic_search import (
    _is_retryable,
    _query_with_retry,
)

_REQUEST = httpx.Request("POST", "https://embed.example/v1/embeddings")


def _status(cls: type, code: int) -> Exception:
    return cls(
        f"Error code: {code}",
        response=httpx.Response(code, request=_REQUEST),
        body=None,
    )


@pytest.mark.parametrize(
    "exc, retryable",
    [
        (APITimeoutError(request=_REQUEST), True),
        (_status(InternalServerError, 502), True),
        (_status(AuthenticationError, 401), False),
        (ValueError("Cannot search for a blank query."), False),
        (RuntimeError("upstream said 503 Service Unavailable"), True),
    ],
    ids=["timeout", "502", "401", "blank", "legacy-text"],
)
def test_classification(exc: Exception, retryable: bool) -> None:
    assert _is_retryable(exc) is retryable


class _Retriever:
    def __init__(self, *outcomes: Any) -> None:
        self.outcomes = list(outcomes)

    def query(self, text: str, top_k: int, vdb_kwargs: Any) -> list[dict]:
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def test_a_timeout_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(semantic_search.time, "sleep", lambda _s: None)
    retriever = _Retriever(APITimeoutError(request=_REQUEST), [{"text": "hit"}])

    assert _query_with_retry(retriever, "x", 3, None) == [{"text": "hit"}]


def test_a_bad_key_raises_at_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(semantic_search.time, "sleep", lambda _s: None)
    retriever = _Retriever(_status(AuthenticationError, 401), [{"text": "unused"}])

    with pytest.raises(AuthenticationError):
        _query_with_retry(retriever, "x", 3, None)
    assert len(retriever.outcomes) == 1
