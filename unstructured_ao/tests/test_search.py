# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Passage annotation, without a database."""

from __future__ import annotations

import contextlib
import types
from typing import Any

import pytest

from unstructured_ao import search


@pytest.fixture
def linked_to_a(monkeypatch: pytest.MonkeyPatch) -> None:
    """Chunk ``a`` reads as attribute ``X``; nothing else is linked."""
    import auto_ontology.dal.session as session

    rows = [
        {
            "chunk_id": "a",
            "attribute_id": "x",
            "attribute_name": "X",
            "source": "vector",
            "score": 1.0,
        }
    ]

    def attribute_rows(_conn: Any, chunk_ids: list[str]) -> list[dict[str, Any]]:
        return [row for row in rows if row["chunk_id"] in chunk_ids]

    monkeypatch.setattr(search, "_chunk_attribute_rows", attribute_rows)
    monkeypatch.setattr(search.ontology, "ontology_note", lambda _ids, _conn: "")
    monkeypatch.setattr(
        session,
        "get_engine",
        lambda: types.SimpleNamespace(connect=contextlib.nullcontext),
    )


@pytest.mark.usefixtures("linked_to_a")
@pytest.mark.parametrize(
    ("chunk_ids", "expected"),
    [
        (["a", "b"], [["X"], []]),
        ([None, "a"], [[], ["X"]]),
        (["b", "b", "a"], [[], [], ["X"]]),
    ],
)
def test_labels_stay_aligned_with_the_passages(
    chunk_ids: list[str | None], expected: list[list[str]]
) -> None:
    """One label list per passage, even for a missing or repeated id."""
    assert search.chunk_ontology(chunk_ids).labels == expected


def test_passages_without_ids_get_empty_labels() -> None:
    assert search.chunk_ontology([None, None]).labels == [[], []]
