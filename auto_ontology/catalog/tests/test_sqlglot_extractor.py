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


def test_extract_tables_and_columns_bare_columns_per_select() -> None:
    """Without schemas, a bare column belongs to the one table its own SELECT reads."""
    result = extract_tables_and_columns(
        "SELECT a FROM t WHERE b IN (SELECT c FROM u) GROUP BY a HAVING COUNT(d) > 1",
        ["postgres"],
        {},
    )
    assert result.tables["t"].columns == {"a", "b", "d"}
    assert result.tables["u"].columns == {"c"}


def test_extract_tables_and_columns_bare_columns_beside_a_visible_cte() -> None:
    """A CTE a SELECT can see but doesn't read isn't one of its tables."""
    result = extract_tables_and_columns(
        "WITH c AS (SELECT a FROM t), d AS (SELECT b FROM u) SELECT * FROM c CROSS JOIN d",
        ["postgres"],
        {},
    )
    assert result.tables["t"].columns == {"a"}
    assert result.tables["u"].columns == {"b"}


def test_extract_tables_and_columns_subquery_inside_having() -> None:
    """A subquery's own columns aren't taken for the outer HAVING's output names."""
    result = extract_tables_and_columns(
        "SELECT a FROM t GROUP BY a HAVING a IN (SELECT c FROM u)", ["postgres"], {}
    )
    assert result.tables["u"].columns == {"c"}


def test_extract_tables_and_columns_window_order_by_matching_an_output_alias() -> None:
    """An ORDER BY inside a window or WITHIN GROUP isn't a SELECT output clause."""
    window = extract_tables_and_columns(
        "SELECT ROW_NUMBER() OVER (ORDER BY a) AS a FROM t", ["postgres"], {}
    )
    assert window.tables["t"].columns == {"a"}
    within_group = extract_tables_and_columns(
        "SELECT STRING_AGG(b, ',') WITHIN GROUP (ORDER BY a) AS a FROM t",
        ["postgres"],
        {},
    )
    assert within_group.tables["t"].columns == {"a", "b"}


def test_extract_tables_and_columns_aggregate_argument_named_like_an_output() -> None:
    """``SUM(a)`` in HAVING reads the input column a, even beside ``AS a``."""
    result = extract_tables_and_columns(
        "SELECT SUM(b) AS a FROM t HAVING SUM(a) > 1", ["postgres"], {}
    )
    assert result.tables["t"].columns == {"a", "b"}


def test_extract_tables_and_columns_scalar_function_of_an_output_alias() -> None:
    """``ORDER BY ABS(total)`` reads the output alias, not a column of the table."""
    result = extract_tables_and_columns(
        "SELECT SUM(b) AS total, c FROM t GROUP BY c ORDER BY ABS(total)",
        ["postgres"],
        {},
    )
    assert result.tables["t"].columns == {"b", "c"}


def test_extract_tables_and_columns_duplicate_alias_keeps_the_tables() -> None:
    """A query sqlglot can't resolve still returns the tables it names."""
    result = extract_tables_and_columns(
        "SELECT a FROM t AS x JOIN u AS x ON TRUE", ["postgres"], {}
    )
    assert set(result.tables) == {"t", "u"}


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
