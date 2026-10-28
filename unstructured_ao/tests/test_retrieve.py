# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from typing import Any

import pytest

import unstructured_ao.retrieve as retrieve


def _record_relink(
    monkeypatch: pytest.MonkeyPatch, failing: set[str]
) -> list[tuple[str, str]]:
    """Stub the DB and vector store; ``events`` logs every clear and insert."""
    events: list[tuple[str, str]] = []
    monkeypatch.setattr(
        retrieve,
        "get_document_attributes",
        lambda _c, _d: [
            {"id": "attr-ok", "name": "ok", "description": ""},
            {"id": "attr-down", "name": "down", "description": ""},
        ],
    )
    monkeypatch.setattr(retrieve, "get_document_chunks", lambda _c, _d: [{"id": "c1"}])
    monkeypatch.setattr(retrieve, "get_chunk_store", lambda: object())

    def search(_vdb: Any, query: str, _k: int, _d: str) -> list[tuple[Any, float]]:
        if query.split(":")[0] in failing:
            raise RuntimeError("endpoint saturated")
        return [(SimpleNamespace(id="c1", metadata={}), 0.9)]

    monkeypatch.setattr(retrieve, "_search_with_retry", search)
    monkeypatch.setattr(
        retrieve,
        "_clear_existing_links",
        lambda _c, attribute_id: events.append(("clear", attribute_id)),
    )
    monkeypatch.setattr(
        retrieve,
        "insert_attribute_chunk_link",
        lambda _c, *, attribute_id, **_k: events.append(("insert", attribute_id)),
    )
    return events


def test_a_failed_search_keeps_the_existing_links(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = _record_relink(monkeypatch, failing={"down"})

    results = retrieve.link_document_attributes("doc-1", object())

    assert events == [("clear", "attr-ok"), ("insert", "attr-ok")]
    assert results["attr-down"] == []
