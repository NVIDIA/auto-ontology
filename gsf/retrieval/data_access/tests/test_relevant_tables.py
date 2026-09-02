# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest
from gsf.catalog.constants import Labels

from gsf.retrieval.data_access import relevant_tables


def test_get_relevant_tables_uses_default_limit_when_k_is_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_search(
        _retriever: object,
        question: str,
        *,
        label_filter: list[str],
        per_label_k: int,
        database_name: str | None,
        schema_name: str | None,
    ) -> list[dict]:
        captured.update(
            {
                "question": question,
                "label_filter": label_filter,
                "per_label_k": per_label_k,
                "database_name": database_name,
                "schema_name": schema_name,
            }
        )
        return [
            {
                "id": "table-1",
                "name": "orders",
                "label": Labels.TABLE,
                "text": "orders",
                "database_name": "retail",
            }
        ]

    monkeypatch.setattr(relevant_tables, "search_semantic_index", fake_search)

    result = relevant_tables.get_relevant_tables(
        object(),
        "show orders",
        database_name="retail",
    )

    assert captured["per_label_k"] == 1
    assert captured["label_filter"] == [Labels.TABLE]
    assert result[0]["name"] == "orders"


def test_normalize_keeps_the_key_when_applied_twice() -> None:
    """The candidate path normalizes once per module, so the shape must be stable.

    A rename here (``pk`` -> something else) silently loses the key on the second
    pass, which costs the prediction graph its entity and every edge with it.
    """
    table = {
        "id": "table-1",
        "name": "Store Locations",
        "label": Labels.TABLE,
        "text": "table_name: Store Locations",
        "pk": ["StoreID"],
    }

    once = relevant_tables._normalize_table_to_relevant_shape(table)
    twice = relevant_tables._normalize_table_to_relevant_shape(once)

    assert once["pk"] == ["StoreID"]
    assert twice["pk"] == ["StoreID"]


def test_get_relevant_tables_from_candidates_keeps_an_already_normalized_key() -> None:
    """``_get_candidates_information`` normalizes before this function sees the dict."""
    normalized = relevant_tables._normalize_table_to_relevant_shape(
        {
            "id": "table-1",
            "name": "Store Locations",
            "label": Labels.TABLE,
            "pk": ["StoreID"],
        }
    )

    result = relevant_tables.get_relevant_tables_from_candidates(
        [{"id": "col-1", "relevant_tables": [normalized]}]
    )

    assert result[0]["pk"] == ["StoreID"]


def test_dedupe_merge_keeps_the_key_of_an_already_normalized_table() -> None:
    """``dedupe_merge_relevant_tables`` normalizes again after merging."""
    normalized = relevant_tables._normalize_table_to_relevant_shape(
        {
            "id": "table-1",
            "name": "Store Locations",
            "label": Labels.TABLE,
            "pk": ["StoreID"],
        }
    )

    merged = relevant_tables.dedupe_merge_relevant_tables([normalized, normalized])

    assert merged[0]["pk"] == ["StoreID"]
