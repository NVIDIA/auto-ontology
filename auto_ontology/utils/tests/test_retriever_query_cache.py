# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A query string is embedded once, and what comes back is exactly that vector.

One question searches for the same entity several times across retrieval,
preparation and reconstruction. Each search used to be its own request to the
embedding endpoint. Reusing the answer is only acceptable if it is the same
answer, and if a failed or blank request is never mistaken for one.
"""

from __future__ import annotations

import threading
from typing import Any

import pytest

from auto_ontology.utils import retriever as retriever_mod
from auto_ontology.utils.embedding import EmbedParams
from auto_ontology.utils.retriever import Retriever, _QueryEmbeddingCache


class _Store:
    """Records the vectors it is searched with."""

    def __init__(self) -> None:
        self.searched: list[list[float]] = []

    def retrieval(self, queries: list, top_k: int = 10, **kwargs: Any) -> list:
        self.searched.extend(queries)
        return [[] for _ in queries]


class _Embedder:
    """Stands in for ``embed_texts``, counting requests per text."""

    def __init__(self) -> None:
        self.requests: list[str] = []
        self.fail_with: Exception | None = None
        self.gate: threading.Event | None = None

    def __call__(
        self, texts: list[str], params: Any, input_type: str, **kw: Any
    ) -> list:
        self.requests.extend(texts)
        if self.gate is not None:
            assert self.gate.wait(5)
        if self.fail_with is not None:
            raise self.fail_with
        if not texts[0].strip():
            return [None]
        # Values that do not survive float32, so a lossy cache would show.
        return [[len(texts[0]) / 3.0, 0.1 + 0.2, 1e-17]]


@pytest.fixture
def embedder(monkeypatch: pytest.MonkeyPatch) -> _Embedder:
    embedder = _Embedder()
    monkeypatch.setattr(retriever_mod, "embed_texts", embedder)
    monkeypatch.setattr(retriever_mod, "_query_embeddings", _QueryEmbeddingCache(8))
    return embedder


def _retriever(model: str = "model-a") -> tuple[Retriever, _Store]:
    store = _Store()
    return Retriever(store, EmbedParams(model, "http://embed.test/v1")), store  # type: ignore[arg-type]


def test_repeated_query_is_embedded_once(embedder: _Embedder) -> None:
    retriever, store = _retriever()

    retriever.query("total revenue")
    retriever.query("total revenue", top_k=3)
    retriever.query("  total revenue ")

    assert embedder.requests == ["total revenue"]
    assert store.searched[0] == store.searched[1] == store.searched[2]


def test_cached_vector_is_bit_identical(embedder: _Embedder) -> None:
    retriever, store = _retriever()

    retriever.query("margin")
    retriever.query("margin")

    first, second = store.searched
    assert first == [len("margin") / 3.0, 0.1 + 0.2, 1e-17]
    assert [value.hex() for value in second] == [value.hex() for value in first]
    assert isinstance(second, list)


def test_different_text_or_model_is_a_different_request(embedder: _Embedder) -> None:
    retriever_a, _ = _retriever("model-a")
    retriever_b, _ = _retriever("model-b")

    retriever_a.query("orders")
    retriever_a.query("Orders")
    retriever_b.query("orders")

    assert embedder.requests == ["orders", "Orders", "orders"]


def test_failure_is_not_cached(embedder: _Embedder) -> None:
    retriever, _ = _retriever()
    embedder.fail_with = RuntimeError("502 Bad Gateway")

    with pytest.raises(RuntimeError, match="502"):
        retriever.query("orders")

    embedder.fail_with = None
    retriever.query("orders")

    assert embedder.requests == ["orders", "orders"]


def test_blank_query_still_raises_and_is_not_cached(embedder: _Embedder) -> None:
    retriever, _ = _retriever()

    for _ in range(2):
        with pytest.raises(ValueError, match="blank query"):
            retriever.query("   ")

    assert embedder.requests == ["   ", "   "]


def test_concurrent_requests_for_one_string_share_a_call(embedder: _Embedder) -> None:
    """Retrieval searches the question under two labels at the same moment."""
    retriever, store = _retriever()
    embedder.gate = threading.Event()
    threads = [
        threading.Thread(target=retriever.query, args=("how many orders",))
        for _ in range(4)
    ]
    for thread in threads:
        thread.start()
    while not embedder.requests:
        pass
    embedder.gate.set()
    for thread in threads:
        thread.join(5)

    assert embedder.requests == ["how many orders"]
    assert len(store.searched) == 4
    assert all(vector == store.searched[0] for vector in store.searched)


def test_concurrent_waiters_see_the_owners_failure(embedder: _Embedder) -> None:
    retriever, _ = _retriever()
    embedder.gate = threading.Event()
    embedder.fail_with = RuntimeError("503 Service Unavailable")
    errors: list[BaseException] = []

    def search() -> None:
        try:
            retriever.query("orders")
        except BaseException as exc:  # noqa: BLE001 — recorded for the assertion
            errors.append(exc)

    threads = [threading.Thread(target=search) for _ in range(3)]
    for thread in threads:
        thread.start()
    while not embedder.requests:
        pass
    embedder.gate.set()
    for thread in threads:
        thread.join(5)

    assert len(errors) == 3
    assert all("503" in str(error) for error in errors)


def test_a_vector_that_cannot_be_packed_is_passed_on_and_strands_nobody() -> None:
    """Whatever the endpoint returned still reaches the store, and every waiter."""
    cache = _QueryEmbeddingCache(8)
    gate = threading.Event()
    odd = [[0.1, 0.2]]  # nested: not a flat list of numbers
    calls: list[int] = []

    def compute() -> list:
        calls.append(1)
        assert gate.wait(5)
        return odd

    results: list[object] = []
    threads = [
        threading.Thread(target=lambda: results.append(cache.get(("k",), compute)))
        for _ in range(3)
    ]
    for thread in threads:
        thread.start()
    while not calls:
        pass
    gate.set()
    for thread in threads:
        thread.join(5)

    assert not any(thread.is_alive() for thread in threads)
    assert results == [odd, odd, odd]
    # Not kept, so the next request asks again instead of reusing it.
    assert cache.get(("k",), compute) == odd
    assert len(calls) == 2


def test_oldest_entry_is_evicted_first(embedder: _Embedder) -> None:
    retriever, _ = _retriever()
    retriever_mod._query_embeddings = _QueryEmbeddingCache(2)

    retriever.query("a")
    retriever.query("b")
    retriever.query("a")  # a is now the most recently used
    retriever.query("c")  # evicts b
    retriever.query("a")
    retriever.query("b")

    assert embedder.requests == ["a", "b", "c", "b"]


def test_size_zero_disables_the_cache(embedder: _Embedder) -> None:
    retriever, _ = _retriever()
    retriever_mod._query_embeddings = _QueryEmbeddingCache(0)

    retriever.query("orders")
    retriever.query("orders")

    assert embedder.requests == ["orders", "orders"]
