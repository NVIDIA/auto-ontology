# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for shared SQL validation helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import sqlglot

from gsf.server import sql_utils


def _connector(database_name: str, dialect: str | None) -> MagicMock:
    connector = MagicMock()
    connector.database_name = database_name
    connector.dialect = dialect
    return connector


@pytest.mark.parametrize("dialect", sql_utils._DEFAULT_DIALECTS)
def test_every_default_dialect_is_one_sqlglot_accepts(dialect: str) -> None:
    """An unknown dialect name raises before any candidate is tried."""
    assert sqlglot.parse_one("SELECT 1", dialect=dialect or None) is not None


@patch("gsf.server.sql_utils.get_connectors")
def test_falls_back_when_no_connector_matches_the_database(
    mock_connectors: MagicMock,
) -> None:
    mock_connectors.return_value = [_connector("retail", "snowflake")]

    assert sql_utils.get_dialects("warehouse") == sql_utils._DEFAULT_DIALECTS


@patch("gsf.server.sql_utils.get_connectors")
def test_uses_the_matching_connector_dialect(mock_connectors: MagicMock) -> None:
    mock_connectors.return_value = [
        _connector("retail", "snowflake"),
        _connector("warehouse", "mysql"),
    ]

    assert sql_utils.get_dialects("warehouse") == ["mysql"]


@patch("gsf.server.sql_utils.get_connectors")
def test_unrecognised_connector_dialects_are_dropped(
    mock_connectors: MagicMock,
) -> None:
    mock_connectors.return_value = [_connector("retail", "not-a-dialect")]

    assert sql_utils.get_dialects("retail") == sql_utils._DEFAULT_DIALECTS


@patch("gsf.server.sql_utils.get_connectors")
def test_missing_dialect_does_not_reach_the_parser(
    mock_connectors: MagicMock,
) -> None:
    mock_connectors.return_value = [_connector("retail", None)]

    assert sql_utils.get_dialects("retail") == sql_utils._DEFAULT_DIALECTS
