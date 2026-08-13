# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest
from nemo_retriever.tabular_data.ingestion.model.reserved_words import Labels

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
