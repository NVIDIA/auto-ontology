# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for shared SQL validation helpers."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import sqlglot

from auto_ontology.server import sql_utils


def _connector(database_name: str, dialect: str) -> MagicMock:
    connector = MagicMock()
    connector.database_name = database_name
    connector.dialect = dialect
    return connector


@pytest.mark.parametrize("dialect", sql_utils._DEFAULT_DIALECTS)
def test_every_default_dialect_is_one_sqlglot_accepts(dialect: str) -> None:
    """An unknown dialect name raises before any candidate is tried."""
    assert sqlglot.parse_one("SELECT 1", dialect=dialect or None) is not None


@patch("auto_ontology.server.sql_utils.get_connectors")
def test_falls_back_when_no_connector_matches_the_database(
    mock_connectors: MagicMock,
) -> None:
    mock_connectors.return_value = [_connector("retail", "snowflake")]

    assert sql_utils.get_dialects("warehouse") == sql_utils._DEFAULT_DIALECTS


@patch("auto_ontology.server.sql_utils.get_connectors")
def test_uses_the_matching_connector_dialect(mock_connectors: MagicMock) -> None:
    mock_connectors.return_value = [
        _connector("retail", "snowflake"),
        _connector("warehouse", "mysql"),
    ]

    assert sql_utils.get_dialects("warehouse") == ["mysql"]


@patch("auto_ontology.server.sql_utils.get_connectors")
def test_falls_back_when_there_are_no_connectors(mock_connectors: MagicMock) -> None:
    mock_connectors.return_value = []

    assert sql_utils.get_dialects() == sql_utils._DEFAULT_DIALECTS


def test_sql_naming_no_catalog_table_asks_for_a_full_query() -> None:
    with patch.object(sql_utils, "parse_query_single", return_value=None):
        with pytest.raises(sql_utils.SqlParseError) as raised:
            sql_utils.validate_sql("SUM(amount)", ["postgres"], {})
    message = str(raised.value)
    assert "known to the catalog" in message
    assert "Write a full query" in message


def test_sql_naming_a_column_its_table_lacks_is_rejected() -> None:
    columns = pd.DataFrame(
        [("wwi_order_lines", "quantity")], columns=["table_name", "column_name"]
    )
    schemas = {"dbt_wwi": SimpleNamespace(columns_df=columns)}
    sql = (
        "SELECT SUM(CASE WHEN order_line__quantity >= 20 THEN quantity END) "
        "AS bulk_units FROM dbt_wwi.wwi_order_lines"
    )
    parsed = MagicMock(dialect_used="postgres")
    with patch.object(sql_utils, "parse_query_single", return_value=parsed):
        with pytest.raises(sql_utils.SqlParseError, match="order_line__quantity"):
            sql_utils.validate_sql(sql, ["postgres"], schemas)
        valid = sql.replace("order_line__quantity", "quantity")
        assert sql_utils.validate_sql(valid, ["postgres"], schemas) is parsed
