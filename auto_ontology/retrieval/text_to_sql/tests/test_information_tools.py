# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded read-only metadata tools for information questions."""

from __future__ import annotations

import pytest

from auto_ontology.retrieval.text_to_sql.agents import information_tools
from auto_ontology.semantic.constants import REL_SEMANTIC_FK


def test_get_dataset_returns_scoped_hierarchy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        information_tools,
        "fetch_databases",
        lambda: [{"id": "db-1", "name": "sales"}],
    )
    monkeypatch.setattr(
        information_tools,
        "fetch_schemas_for_database",
        lambda _database_id: [{"id": "schema-1", "name": "public"}],
    )
    monkeypatch.setattr(
        information_tools,
        "fetch_tables_for_schema",
        lambda _schema_id, **_kwargs: [{"id": "table-1", "name": "orders"}],
    )

    result = information_tools.get_dataset(dataset_name="SALES")

    assert result["ok"] is True
    assert result["data"]["dataset"]["id"] == "db-1"
    assert result["data"]["schemas"][0]["tables"][0]["id"] == "table-1"
    assert result["truncated"] is False


def test_get_dataset_reports_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(information_tools, "fetch_databases", lambda: [])

    result = information_tools.get_dataset(dataset_name="missing")

    assert result["ok"] is False
    assert result["error"]["code"] == "not_found"


def test_get_dataset_reports_ambiguous_identifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        information_tools,
        "fetch_databases",
        lambda: [
            {"id": "same", "name": "first"},
            {"id": "db-2", "name": "same"},
        ],
    )

    result = information_tools.get_dataset(dataset_name="same")

    assert result["ok"] is False
    assert result["error"]["code"] == "ambiguous"


def test_get_dataset_caps_table_results(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        information_tools,
        "fetch_databases",
        lambda: [{"id": "db-1", "name": "sales"}],
    )
    monkeypatch.setattr(
        information_tools,
        "fetch_schemas_for_database",
        lambda _database_id: [{"id": "schema-1", "name": "public"}],
    )
    monkeypatch.setattr(
        information_tools,
        "fetch_tables_for_schema",
        lambda _schema_id, **_kwargs: [
            {"id": f"table-{index}", "name": f"table_{index}"}
            for index in range(information_tools.MAX_TABLES + 1)
        ],
    )

    result = information_tools.get_dataset(dataset_name="sales")

    assert result["truncated"] is True
    assert len(result["data"]["schemas"][0]["tables"]) == information_tools.MAX_TABLES


def test_get_table_returns_columns_semantics_and_connections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        information_tools,
        "fetch_table_by_id",
        lambda _table_id: {"id": "table-1", "name": "orders"},
    )
    monkeypatch.setattr(
        information_tools,
        "fetch_columns_for_table",
        lambda *_args, **_kwargs: {
            "columns": [{"id": "column-1", "column_name": "revenue"}]
        },
    )
    monkeypatch.setattr(
        information_tools,
        "fetch_join_neighbors",
        lambda _table_id: [{"id": "table-2", "name": "customers"}],
    )
    monkeypatch.setattr(
        information_tools,
        "fetch_terms_and_attributes_for_table",
        lambda _table_id: (
            [{"id": "term-1", "name": "Order"}],
            [{"id": "attr-1", "name": "Revenue"}],
        ),
    )

    result = information_tools.get_table(table_id="table-1")

    assert result["ok"] is True
    assert result["data"]["columns"][0]["column_name"] == "revenue"
    assert result["data"]["connected_tables"][0]["name"] == "customers"
    assert result["data"]["column_attributes"][0]["id"] == "attr-1"


def test_get_column_returns_calculation_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        information_tools,
        "fetch_parent_table_id_for_column",
        lambda _column_id: "table-1",
    )
    monkeypatch.setattr(
        information_tools,
        "fetch_columns_for_table",
        lambda *_args, **_kwargs: {
            "columns": [{"id": "column-1", "column_name": "revenue"}]
        },
    )
    monkeypatch.setattr(
        information_tools,
        "fetch_column_exploration_details",
        lambda _column_id: {
            "column_attribute": {
                "id": "attr-1",
                "description": "Gross sales minus refunds",
            },
            "sql_queries": [{"statement": "SUM(gross_sales - refunds)"}],
        },
    )
    monkeypatch.setattr(
        information_tools,
        "fetch_table_by_id",
        lambda _table_id: {"id": "table-1", "name": "orders"},
    )
    monkeypatch.setattr(
        information_tools,
        "fetch_join_neighbors",
        lambda _table_id: [{"id": "table-2", "name": "customers"}],
    )

    result = information_tools.get_column(column_id="column-1")

    assert result["ok"] is True
    assert result["data"]["semantic_details"]["column_attribute"]["id"] == "attr-1"
    assert result["data"]["table"]["name"] == "orders"
    assert result["data"]["connected_tables"][0]["id"] == "table-2"


def test_semantic_fk_tool_only_returns_semantic_fk_attributes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        information_tools,
        "fetch_table_by_id",
        lambda _table_id: {"id": "table-1", "name": "orders"},
    )
    monkeypatch.setattr(
        information_tools,
        "fetch_table_context",
        lambda _table_id: {
            "columns": [
                {"id": "column-1", "name": "customer_id"},
                {"id": "column-2", "name": "amount"},
            ],
            "fks": [],
        },
    )

    def details(column_id: str) -> dict:
        if column_id == "column-1":
            return {
                "column_attribute": {
                    "id": "attr-1",
                    "relationship_type": REL_SEMANTIC_FK,
                }
            }
        return {
            "column_attribute": {
                "id": "attr-2",
                "relationship_type": "HAS_ATTRIBUTE",
            }
        }

    monkeypatch.setattr(information_tools, "fetch_column_exploration_details", details)
    monkeypatch.setattr(
        information_tools,
        "fetch_column_attribute_exploration_details",
        lambda _attribute_id: {
            "columns": [
                {
                    "id": "customer-pk",
                    "table_id": "customers",
                    "table_name": "customers",
                }
            ]
        },
    )

    result = information_tools.get_table_semantic_fks(table_id="table-1")

    semantic_fks = result["data"]["semantic_foreign_keys"]
    assert len(semantic_fks) == 1
    assert semantic_fks[0]["source_column"]["id"] == "column-1"
    assert semantic_fks[0]["connected_columns"][0]["table_name"] == "customers"


def test_dispatch_rejects_unknown_tool() -> None:
    result = information_tools.run_information_tool("delete_table", {})

    assert result["ok"] is False
    assert result["error"]["code"] == "unknown_tool"
