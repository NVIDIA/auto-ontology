# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for bounded semantic-preparation tools."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from auto_ontology.retrieval.text_to_sql.agents import semantic_preparation_tools
from auto_ontology.retrieval.text_to_sql.agents.semantic_preparation_tools import (
    find_attribute_join_path,
    probe_value,
    search_semantic_layer,
)
from auto_ontology.semantic.constants import LABEL_COLUMN_ATTRIBUTE


class _Connector:
    database_name = "cards"
    dialect = "sqlite"

    def __init__(self, responses: list[list[dict[str, Any]]]) -> None:
        self.responses = iter(responses)
        self.queries: list[str] = []

    def execute(
        self,
        sql: str,
        parameters: list[Any] | None = None,
        **_: Any,
    ) -> pd.DataFrame:
        self.queries.append(sql)
        return pd.DataFrame(next(self.responses))


def _contexts(_ids: list[str], **_: Any) -> dict[str, dict[str, str]]:
    return {
        "attr-1": {
            "col_id": "col-1",
            "col_name": "name",
            "table_id": "table-1",
            "table_name": "cards",
            "schema_name": "main",
            "database_name": "cards",
            "datatype": "TEXT",
            "attr_name": "Card name",
        },
        "attr-2": {
            "col_id": "col-2",
            "col_name": "translation",
            "table_id": "table-2",
            "table_name": "foreign_data",
            "schema_name": "main",
            "database_name": "cards",
            "datatype": "VARCHAR",
            "attr_name": "Translated text",
        },
    }


def test_semantic_search_is_scoped_and_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def search(_retriever: object, query: str, **kwargs: Any) -> list[dict]:
        calls.append({"query": query, **kwargs})
        return [{"id": "attr-1", "label": LABEL_COLUMN_ATTRIBUTE}]

    monkeypatch.setattr(semantic_preparation_tools, "search_semantic_index", search)

    result = search_semantic_layer(
        query="card artist",
        labels=[LABEL_COLUMN_ATTRIBUTE],
        database_name="cards",
        limit=99,
        semantic_retriever=object(),
    )

    assert result["ok"] is True
    assert calls == [
        {
            "query": "card artist",
            "label_filter": [LABEL_COLUMN_ATTRIBUTE],
            "per_label_k": 10,
            "database_name": "cards",
        }
    ]


def test_probe_checks_all_equal_suspects_before_like(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        semantic_preparation_tools,
        "fetch_attr_column_contexts",
        _contexts,
    )
    connector = _Connector(
        [
            [],
            [],
            [],
            [{"matched_value": "Das perfekte Gegenmittel"}],
        ]
    )

    result = probe_value(
        value="perfekte gegenmittel",
        attribute_ids=["attr-1", "attr-2"],
        database_name="cards",
        connectors=[connector],
    )

    assert result["data"]["match_type"] == "contains"
    assert result["data"]["column_attribute_id"] == "attr-2"
    assert len(connector.queries) == 4
    assert all(" LIKE " not in sql for sql in connector.queries[:2])
    assert all(" LIKE " in sql for sql in connector.queries[2:])


def test_probe_stops_after_case_insensitive_equal_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        semantic_preparation_tools,
        "fetch_attr_column_contexts",
        _contexts,
    )
    connector = _Connector(
        [
            [],
            [{"matched_value": "Angel of Mercy"}],
        ]
    )

    result = probe_value(
        value="angel OF mercy",
        attribute_ids=["attr-1", "attr-2"],
        database_name="cards",
        connectors=[connector],
    )

    assert result["data"]["stored_value"] == "Angel of Mercy"
    assert result["data"]["match_type"] == "equal"
    assert len(connector.queries) == 2
    assert all(" LIKE " not in sql for sql in connector.queries)


def test_find_join_path_resolves_attribute_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        semantic_preparation_tools,
        "fetch_attr_column_contexts",
        _contexts,
    )
    monkeypatch.setattr(
        semantic_preparation_tools,
        "find_join_path",
        lambda anchor, destination: [
            {"source_column": anchor, "target_column": destination}
        ],
    )

    result = find_attribute_join_path(
        anchor_attribute_id="attr-1",
        destination_attribute_id="attr-2",
        database_name="cards",
    )

    assert result["ok"] is True
    assert result["data"]["path"] == [
        {"source_column": "col-1", "target_column": "col-2"}
    ]
