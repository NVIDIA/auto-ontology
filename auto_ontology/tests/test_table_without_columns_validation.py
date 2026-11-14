# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for queries referencing tables without named columns.

A query referencing a table without projecting or filtering on any of its named
columns — e.g. ``SELECT COUNT(*) FROM schema.table``, ``SELECT COUNT(1)``, or
``SELECT 1`` — must retain the source table during extraction and successfully
resolve against catalog schemas instead of being rejected as referencing no
known tables.
"""

from __future__ import annotations

from unittest.mock import MagicMock
import pandas as pd

from auto_ontology.catalog.sql_parse import parse_query_single


def _mock_schema(table_name: str, table_id: str = "tbl-1") -> MagicMock:
    schema = MagicMock()
    schema.columns_df = pd.DataFrame([{"table_name": table_name, "column_name": "id"}])
    schema.table_exists.side_effect = lambda t: t.lower() == table_name.lower()
    schema.get_columns_names.return_value = ["id"]

    def _get_table_node(tbl: str):
        if tbl.lower() != table_name.lower():
            raise ValueError(f"Table {tbl} is not in schema")
        return MagicMock(id=table_id)

    schema.get_table_node.side_effect = _get_table_node
    schema.is_column_in_table.return_value = True
    schema.get_column_node.return_value = MagicMock(id="col-1")
    schema.id_to_node = {table_id: f"mydb.{table_name}"}
    return schema


def test_count_star_resolves_known_table_in_parse_query_single() -> None:
    """COUNT(*) must retain the referenced table and resolve to a valid Query."""
    schemas = {"mydb": _mock_schema("customers", table_id="cust-id")}

    query_obj = parse_query_single(
        sql="SELECT COUNT(*) AS n FROM mydb.customers",
        schemas=schemas,
        dialects=["mysql"],
    )

    assert query_obj is not None
    assert query_obj.get_tables_ids() == ["cust-id"]
    assert query_obj.get_column_ids() == []
