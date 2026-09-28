# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded read-only metadata tools for information questions."""

from __future__ import annotations

import pytest

from auto_ontology.retrieval.text_to_sql.agents import information_tools
from auto_ontology.semantic.constants import REL_SEMANTIC_FK


def test_lazy_catalog_tools_mirror_catalog_endpoints(
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
        lambda _database_id: {
            "schemas_count": 1,
            "schemas": [{"id": "schema-1", "schema_name": "public"}],
        },
    )
    monkeypatch.setattr(
        information_tools,
        "fetch_tables_for_schema",
        lambda _schema_id, **_kwargs: [{"id": "table-1", "name": "orders"}],
    )
    monkeypatch.setattr(
        information_tools,
        "fetch_columns_for_table",
        lambda _table_id, **_kwargs: {
            "table_name": "orders",
            "columns": [{"id": "column-1", "column_name": "revenue"}],
        },
    )
    monkeypatch.setattr(
        information_tools,
        "count_columns_for_table",
        lambda _table_id: 1,
    )

    databases = information_tools.list_databases()
    schemas = information_tools.list_schemas(database_id="db-1")
    tables = information_tools.list_tables(schema_id="schema-1")
    columns = information_tools.list_columns(table_id="table-1")

    assert databases["data"]["databases"][0]["id"] == "db-1"
    assert schemas["data"]["schemas"][0]["id"] == "schema-1"
    assert tables["data"]["tables"][0]["id"] == "table-1"
    assert columns["data"]["columns"][0]["id"] == "column-1"
    assert columns["data"]["total"] == 1


def test_lazy_catalog_tools_require_parent_ids() -> None:
    assert information_tools.list_schemas()["error"]["code"] == "invalid_arguments"
    assert information_tools.list_tables()["error"]["code"] == "invalid_arguments"
    assert information_tools.list_columns()["error"]["code"] == "invalid_arguments"


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
        lambda _database_id: {
            "schemas_count": 1,
            "schemas": [{"id": "schema-1", "schema_name": "public"}],
        },
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
        lambda _database_id: {
            "schemas_count": 1,
            "schemas": [{"id": "schema-1", "schema_name": "public"}],
        },
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


def test_list_terms_returns_bounded_glossary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        information_tools,
        "fetch_all_terms",
        lambda **_kwargs: [{"id": "term-1", "name": "Revenue"}],
    )
    monkeypatch.setattr(information_tools, "count_terms", lambda **_kwargs: 1)

    result = information_tools.list_terms(search="rev", limit=10)

    assert result["ok"] is True
    assert result["data"]["terms"][0]["name"] == "Revenue"
    assert result["data"]["total"] == 1


def test_get_term_returns_both_attribute_kinds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        information_tools,
        "get_full_term_by_id",
        lambda _term_id: {
            "id": "term-1",
            "name": "Revenue",
            "tables": [{"id": "table-1", "name": "orders"}],
        },
    )
    monkeypatch.setattr(
        information_tools,
        "fetch_column_attributes_by_term_id",
        lambda *_args, **_kwargs: [{"id": "column-attr-1", "name": "Revenue"}],
    )
    monkeypatch.setattr(
        information_tools,
        "fetch_sql_attributes_by_term_id",
        lambda *_args, **_kwargs: [{"id": "sql-attr-1", "name": "Net Revenue"}],
    )

    result = information_tools.get_term(term_id="term-1")

    assert result["data"]["term"]["name"] == "Revenue"
    assert result["data"]["column_attributes"][0]["id"] == "column-attr-1"
    assert result["data"]["sql_attributes"][0]["id"] == "sql-attr-1"


def test_get_column_attribute_returns_term_and_linked_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        information_tools,
        "fetch_column_attribute_exploration_details",
        lambda *_args, **_kwargs: {
            "term": {"id": "term-1", "name": "Revenue"},
            "columns": [{"id": "column-1", "table_name": "orders"}],
            "columns_total": 1,
        },
    )
    monkeypatch.setattr(
        information_tools,
        "get_full_term_by_id",
        lambda _term_id: {"id": "term-1", "name": "Revenue"},
    )
    monkeypatch.setattr(
        information_tools,
        "fetch_column_attributes_by_term_id",
        lambda *_args, **_kwargs: [
            {
                "id": "column-attr-1",
                "name": "Revenue",
                "description": "Gross sales minus refunds.",
            }
        ],
    )

    result = information_tools.get_column_attribute(column_attribute_id="column-attr-1")

    assert result["ok"] is True
    assert result["data"]["column_attribute"]["description"].startswith("Gross")
    assert result["data"]["term"]["id"] == "term-1"
    assert result["data"]["linked_columns"][0]["id"] == "column-1"


def test_get_sql_attribute_returns_formula_and_statement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        information_tools,
        "get_full_sql_attribute_by_id",
        lambda _attr_id: {
            "id": "sql-attr-1",
            "name": "Net Revenue",
            "expression": "gross_sales - refunds",
        },
    )
    monkeypatch.setattr(
        information_tools,
        "fetch_sql_attribute_exploration_details",
        lambda _attr_id: {
            "term": {"id": "term-1", "name": "Revenue"},
            "sql": {"id": "sql-1", "sql": "SELECT gross_sales - refunds"},
        },
    )

    result = information_tools.get_sql_attribute(sql_attribute_id="sql-attr-1")

    assert result["ok"] is True
    assert result["data"]["sql_attribute"]["expression"] == "gross_sales - refunds"
    assert result["data"]["term"]["name"] == "Revenue"
    assert "SELECT" in result["data"]["sql"]["sql"]


def test_semantic_search_is_bounded_and_receives_retriever(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retriever = object()
    captured: dict = {}

    def search(
        actual_retriever: object,
        query: str,
        **kwargs: object,
    ) -> list[dict]:
        captured.update({"retriever": actual_retriever, "query": query, **kwargs})
        return [
            {"id": f"term-{index}", "label": "Term", "text": f"Term {index}"}
            for index in range(12)
        ]

    monkeypatch.setattr(information_tools, "search_semantic_index", search)

    result = information_tools.run_information_tool(
        "search_semantic_layer",
        {"query": "revenue", "labels": ["Term"], "limit": 5},
        semantic_retriever=retriever,
    )

    assert result["ok"] is True
    assert captured["retriever"] is retriever
    assert captured["query"] == "revenue"
    assert captured["label_filter"] == ["Term"]
    assert (
        len(result["data"]["matches"]) == information_tools.MAX_SEMANTIC_SEARCH_RESULTS
    )
    assert result["truncated"] is True
