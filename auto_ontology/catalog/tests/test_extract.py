# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed catalog extraction coverage tests."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from auto_ontology.catalog.extract import (
    IncompleteCatalogExtractionError,
    _relation_keys,
    create_dataframe,
)


def _connector(tables: list[dict], columns: list[dict]) -> SimpleNamespace:
    return SimpleNamespace(
        get_tables=lambda: pd.DataFrame(tables),
        get_columns=lambda: pd.DataFrame(columns),
        get_views=lambda: pd.DataFrame(),
        get_queries=lambda: pd.DataFrame(),
        get_pks=lambda: pd.DataFrame(),
        get_fks=lambda: pd.DataFrame(),
    )


def test_empty_catalog_is_valid() -> None:
    tables, columns, *_rest = create_dataframe(_connector([], []))

    assert tables.empty
    assert columns.empty


def test_nullable_relation_identifiers_are_normalized_without_boolean_coercion() -> (
    None
):
    frame = pd.DataFrame(
        [
            {"table_schema": pd.NA, "table_name": "orders"},
            {"table_schema": "sales", "table_name": pd.NA},
        ]
    )

    assert _relation_keys(frame) == {("", "orders")}


def test_every_listed_relation_with_columns_is_valid() -> None:
    connector = _connector(
        [
            {"table_schema": "sales", "table_name": "orders"},
            {"table_schema": "sales", "table_name": "customers"},
        ],
        [
            {
                "table_schema": "sales",
                "table_name": "orders",
                "column_name": "id",
            },
            {
                "table_schema": "sales",
                "table_name": "customers",
                "column_name": "id",
            },
        ],
    )

    tables, columns, *_rest = create_dataframe(connector)

    assert len(tables) == 2
    assert len(columns) == 2


def test_listed_relation_without_columns_fails_before_catalog_write() -> None:
    connector = _connector(
        [
            {"table_schema": "sales", "table_name": "orders"},
            {"table_schema": "private", "table_name": "customers"},
        ],
        [
            {
                "table_schema": "sales",
                "table_name": "orders",
                "column_name": "id",
            }
        ],
    )

    with pytest.raises(IncompleteCatalogExtractionError) as exc_info:
        create_dataframe(connector)

    message = str(exc_info.value)
    assert "2 relation(s)" in message
    assert "1 had no described columns" in message
    assert "private.customers" in message
