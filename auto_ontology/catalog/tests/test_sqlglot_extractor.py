# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for extract_tables_and_columns in sqlglot_extractor."""

from __future__ import annotations

from unittest.mock import MagicMock
import pandas as pd

from auto_ontology.catalog.parsers.sqlglot_extractor import (
    extract_tables_and_columns,
)
from auto_ontology.catalog.sql_parse import parse_query_single


def test_extract_tables_and_columns_count_star_retains_table() -> None:
    """Queries without named columns (e.g. COUNT(*)) must keep the source table."""
    result = extract_tables_and_columns("SELECT COUNT(*) FROM s.t", ["mysql"], {})
    assert "s.t" in result.tables
    assert result.tables["s.t"].columns == set()


def test_extract_tables_and_columns_select_literal_retains_table() -> None:
    """SELECT 1 FROM table retains the table even without column references."""
    result = extract_tables_and_columns("SELECT 1 FROM my_table", ["postgres"], {})
    assert "my_table" in result.tables
    assert result.tables["my_table"].columns == set()


def test_extract_tables_and_columns_with_column() -> None:
    """Queries referencing columns retain both the table and the columns."""
    result = extract_tables_and_columns("SELECT COUNT(col) FROM s.t", ["mysql"], {})
    assert "s.t" in result.tables
    assert result.tables["s.t"].columns == {"col"}


def test_extract_tables_and_columns_cte_virtual_tables_excluded() -> None:
    """CTE aliases must not be treated as real source tables."""
    result = extract_tables_and_columns(
        "WITH cte AS (SELECT 1 AS x) SELECT * FROM cte", ["postgres"], {}
    )
    assert result.tables == {}


def test_extract_tables_and_columns_cte_with_count_star_from_real_table() -> None:
    """CTE querying a real table with COUNT(*) retains the real table."""
    result = extract_tables_and_columns(
        "WITH cte AS (SELECT COUNT(*) FROM real_tbl) SELECT * FROM cte",
        ["postgres"],
        {},
    )
    assert "real_tbl" in result.tables
    assert "cte" not in result.tables


def test_parse_query_single_succeeds_for_count_star() -> None:
    """parse_query_single returns a Query object for COUNT(*) over a catalogued table."""
    schema = MagicMock()
    schema.columns_df = pd.DataFrame([{"table_name": "customers", "column_name": "id"}])
    schema.table_exists.return_value = True
    schema.get_columns_names.return_value = ["id"]
    schema.get_table_node.return_value = MagicMock(id="node-customers")
    schema.is_column_in_table.return_value = True
    schema.get_column_node.return_value = MagicMock(id="node-id")
    schemas = {"mydb": schema}

    query_obj = parse_query_single(
        sql="SELECT COUNT(*) AS n FROM mydb.customers",
        schemas=schemas,
        dialects=["mysql"],
    )
    assert query_obj is not None
    assert query_obj.get_tables_ids() == ["node-customers"]
    assert query_obj.get_column_ids() == []
