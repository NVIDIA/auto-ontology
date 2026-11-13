# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Batch retrieval isolates per-query failures.

A multi-query ``retrieval`` call where one query errors keeps the hits
already gathered, still attempts the remaining queries, and reports the
failed query as an empty list. Regression test for the ``except`` block
that used to discard every result in the batch with an unconditional
``return [[] for _ in queries]``.

These tests need no database: the vector store is faked, so the batch
loop's error isolation is exercised directly.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from auto_ontology.vdb.postgres import PostgresVDB


class _FlakyStore:
    """Fake store that records calls and fails one specific call."""

    def __init__(self, fail_on_call: int) -> None:
        self.fail_on_call = fail_on_call
        self.calls = 0

    def similarity_search_with_score_by_vector(
        self, **kwargs: Any
    ) -> list[tuple[Any, float]]:
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise RuntimeError("similarity search failed")
        return [
            (
                SimpleNamespace(
                    page_content=f"hit {self.calls}",
                    metadata={"call": self.calls},
                ),
                float(self.calls),
            )
        ]


class _AlwaysFailingStore:
    def similarity_search_with_score_by_vector(
        self, **kwargs: Any
    ) -> list[tuple[Any, float]]:
        raise RuntimeError("similarity search failed")


def _make_vdb(store: Any) -> PostgresVDB:
    # ``__new__`` skips ``__init__`` on purpose: no connection string or
    # engine is needed. With a live store, ``retrieval`` only touches the
    # engine for the debug pool-status read, which tolerates a bare object.
    vdb = PostgresVDB.__new__(PostgresVDB)
    vdb._store = store
    vdb._engine = SimpleNamespace()
    return vdb


def test_failed_query_is_isolated_to_empty_list() -> None:
    vdb = _make_vdb(_FlakyStore(fail_on_call=2))
    results = vdb.retrieval(["q1", "q2", "q3"], top_k=5)

    assert len(results) == 3
    assert [hit["text"] for hit in results[0]] == ["hit 1"]
    assert results[1] == []
    assert [hit["text"] for hit in results[2]] == ["hit 3"]
    assert vdb._store.calls == 3  # remaining queries still attempted


def test_every_query_failing_yields_all_empty() -> None:
    vdb = _make_vdb(_AlwaysFailingStore())
    assert vdb.retrieval(["q1", "q2"]) == [[], []]


def test_successful_batch_is_untouched() -> None:
    vdb = _make_vdb(_FlakyStore(fail_on_call=0))  # 0 never matches: no failure
    results = vdb.retrieval(["q1", "q2"], top_k=5)
    assert [[hit["text"] for hit in batch] for batch in results] == [
        ["hit 1"],
        ["hit 2"],
    ]
