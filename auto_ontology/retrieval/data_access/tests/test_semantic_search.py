# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from auto_ontology.retrieval.data_access.semantic_search import (
    _build_metadata_where_clause,
    _hits_to_semantic_rows,
)


def test_metadata_where_clause_filters_multiple_table_ids_for_sql_store() -> None:
    where = _build_metadata_where_clause(
        labels=["ColumnAttribute"],
        database_name="db",
        table_ids=["orders", "customers"],
    )

    assert isinstance(where, str)
    assert '"table_id":"orders"' in where
    assert '"table_id":"customers"' in where
    assert " OR " in where
    assert " AND " in where


def test_metadata_where_clause_filters_multiple_table_ids_for_dict_store() -> None:
    where = _build_metadata_where_clause(
        labels=["ColumnAttribute"],
        database_name="db",
        table_ids=["orders", "customers"],
        fmt="dict",
    )

    assert where == {
        "label": "ColumnAttribute",
        "database_name": "db",
        "table_id": ["orders", "customers"],
    }


def test_semantic_rows_preserve_table_scope_metadata() -> None:
    rows = _hits_to_semantic_rows(
        [
            {
                "text": "Order revenue",
                "_distance": 0.1,
                "metadata": {
                    "id": "revenue-col",
                    "label": "ColumnAttribute",
                    "database_name": "db",
                    "table_id": "orders",
                    "table_name": "orders",
                },
            }
        ],
        label_filter={"ColumnAttribute"},
        per_label_k=2,
    )

    assert rows[0]["table_id"] == "orders"
    assert rows[0]["table_name"] == "orders"
