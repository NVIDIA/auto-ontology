# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Term search box: name matches blended with matches by meaning."""

from __future__ import annotations

from typing import Any

import pytest

from auto_ontology.server.terms import service

TERMS = {
    "t-cust": {"id": "t-cust", "name": "Customer"},
    "t-cat": {"id": "t-cat", "name": "Customer Category"},
    "t-buy": {"id": "t-buy", "name": "Buying Group"},
    "t-inv": {"id": "t-inv", "name": "Invoice Line"},
}


@pytest.fixture
def fake_reads(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Patch the DAL read and the vector search; record what they were asked."""
    calls: dict[str, Any] = {"hits": [], "fail": False, "term_ids": None}

    def fetch_all_terms(
        search: str | None = None,
        limit: int | None = None,
        term_ids: list[str] | None = None,
        **_: Any,
    ) -> list[dict[str, Any]]:
        if term_ids is not None:
            calls["term_ids"] = list(term_ids)
            return [dict(TERMS[tid]) for tid in term_ids if tid in TERMS]
        needle = (search or "").lower()
        rows = [dict(t) for t in TERMS.values() if needle in t["name"].lower()]
        return sorted(rows, key=lambda r: r["name"].lower())[:limit]

    def search_semantic_index(*_: Any, **__: Any) -> list[dict[str, Any]]:
        if calls["fail"]:
            raise RuntimeError("embedding endpoint down")
        return calls["hits"]

    monkeypatch.setattr(service.terms_dal, "fetch_all_terms", fetch_all_terms)
    monkeypatch.setattr(service, "search_semantic_index", search_semantic_index)
    monkeypatch.setattr(service, "get_semantic_objects_retriever", lambda: object())
    return calls


def test_exact_name_ranks_before_prefix_and_meaning(fake_reads: dict[str, Any]) -> None:
    fake_reads["hits"] = [
        {"id": "t-buy", "score": 0.2},
        {"id": "t-cust", "score": 0.1},
    ]

    out = service.search_terms("customer", limit=10)

    assert [item["id"] for item in out["data"]] == ["t-cust", "t-cat", "t-buy"]
    assert [item["match"] for item in out["data"]] == ["name", "name", "meaning"]
    assert out["data"][0]["similarity"] == pytest.approx(0.9)
    assert out["data"][1]["similarity"] is None
    assert out["semantic"] is True
    # Only the vector hit that wasn't already a name match is hydrated.
    assert fake_reads["term_ids"] == ["t-buy"]


def test_meaning_hits_are_ordered_by_distance(fake_reads: dict[str, Any]) -> None:
    fake_reads["hits"] = [
        {"id": "t-inv", "score": 0.15},
        {"id": "t-buy", "score": 0.4},
        {"id": "t-inv", "score": 0.3},  # a second, worse hit for the same term
    ]

    out = service.search_terms("revenue", limit=10)

    assert [item["id"] for item in out["data"]] == ["t-inv", "t-buy"]
    assert out["data"][0]["similarity"] == pytest.approx(0.85)


def test_hits_the_glossary_cannot_see_are_dropped(fake_reads: dict[str, Any]) -> None:
    fake_reads["hits"] = [
        {"id": "t-hidden", "score": 0.05},
        {"id": "t-buy", "score": 0.3},
    ]

    out = service.search_terms("group", limit=10)

    assert [item["id"] for item in out["data"]] == ["t-buy"]


def test_embedding_failure_still_returns_name_matches(
    fake_reads: dict[str, Any],
) -> None:
    fake_reads["fail"] = True

    out = service.search_terms("cust", limit=10)

    assert [item["id"] for item in out["data"]] == ["t-cust", "t-cat"]
    assert out["semantic"] is False


def test_limit_caps_the_blend(fake_reads: dict[str, Any]) -> None:
    fake_reads["hits"] = [{"id": "t-buy", "score": 0.2}, {"id": "t-inv", "score": 0.3}]

    out = service.search_terms("customer", limit=3)

    assert out["count"] == 3
    assert [item["id"] for item in out["data"]] == ["t-cust", "t-cat", "t-buy"]


def test_blank_query_returns_nothing(fake_reads: dict[str, Any]) -> None:
    assert service.search_terms("   ")["data"] == []
