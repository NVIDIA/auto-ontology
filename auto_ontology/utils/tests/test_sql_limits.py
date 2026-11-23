# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for dialect-aware row limits."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from auto_ontology.connectors.registry import CONNECTOR_REGISTRY
from auto_ontology.utils.sql_limits import with_row_limit

_QUERY = 'SELECT DISTINCT "c" FROM "s"."t" WHERE "c" IS NOT NULL ORDER BY "c"'


@pytest.mark.parametrize(
    "dialect",
    ["postgres", "mysql", "sqlite", "duckdb", "snowflake", "clickhouse", "trino"],
)
def test_limit_dialects_keep_the_sql_byte_for_byte(dialect: str) -> None:
    # No round trip through sqlglot, so the SQL these engines run is unchanged.
    assert with_row_limit(_QUERY, 20, dialect) == f"{_QUERY} LIMIT 20"


def test_every_registered_connector_keeps_limit() -> None:
    # Guards the claim above for the real connectors, not a hand-kept list: a
    # connector whose dialect does not spell LIMIT should be a deliberate change.
    for connector_cls in set(CONNECTOR_REGISTRY.values()):
        dialect = connector_cls.dialect.fget(MagicMock())
        if not isinstance(dialect, str):
            continue
        assert with_row_limit("SELECT 1", 5, dialect) == "SELECT 1 LIMIT 5", (
            connector_cls.__name__
        )


def test_oracle_uses_fetch_first() -> None:
    assert with_row_limit(_QUERY, 20, "oracle").endswith("FETCH FIRST 20 ROWS ONLY")


def test_oracle_keeps_unaliased_join_probe_shape() -> None:
    # Oracle rejects ``AS`` before a table alias, so the rewrite must not add one.
    sql = with_row_limit(
        'SELECT 1 AS hit FROM "s"."a" s JOIN "s"."b" t ON s."x" = t."y"', 1, "oracle"
    )
    assert sql == (
        'SELECT 1 AS hit FROM "s"."a" s JOIN "s"."b" t ON s."x" = t."y" '
        "FETCH FIRST 1 ROWS ONLY"
    )


def test_tsql_uses_top() -> None:
    assert (
        with_row_limit("SELECT [c] FROM [t]", 5, "tsql") == "SELECT TOP 5 [c] FROM [t]"
    )


@pytest.mark.parametrize("dialect", [None, "", "not-a-dialect", MagicMock()])
def test_missing_or_unknown_dialect_falls_back_to_limit(dialect: object) -> None:
    assert with_row_limit("SELECT 1", 5, dialect) == "SELECT 1 LIMIT 5"


def test_unparseable_sql_raises_for_a_rewriting_dialect() -> None:
    with pytest.raises(ValueError):
        with_row_limit("SELECT FROM WHERE (", 5, "oracle")
