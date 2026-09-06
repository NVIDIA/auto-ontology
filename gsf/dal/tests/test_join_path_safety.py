# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed authoritative join-path formatting tests."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from gsf.dal import attributes


def _path() -> list[dict[str, str]]:
    return [
        {"id": "source", "kind": "column"},
        {"id": "attribute", "kind": "attribute"},
        {"id": "target", "kind": "column"},
    ]


def _context(table: str) -> dict[str, str]:
    return {
        "database_name": "cards_db",
        "schema_name": "main",
        "table_name": table,
    }


@pytest.mark.parametrize("target_column", ["variations", "number"])
def test_same_table_cross_column_path_is_rejected(target_column: str) -> None:
    with (
        patch("gsf.dal.attributes._bfs_path", return_value=_path()),
        patch(
            "gsf.dal.attributes._name_columns",
            return_value={"source": "borderColor", "target": target_column},
        ),
        patch(
            "gsf.dal.attributes.fetch_col_table_contexts",
            return_value={"source": _context("cards"), "target": _context("cards")},
        ),
    ):
        assert attributes.find_join_path("source", "target") == []


def test_cross_table_path_is_preserved() -> None:
    with (
        patch("gsf.dal.attributes._bfs_path", return_value=_path()),
        patch(
            "gsf.dal.attributes._name_columns",
            return_value={"source": "customer_id", "target": "id"},
        ),
        patch(
            "gsf.dal.attributes.fetch_col_table_contexts",
            return_value={
                "source": _context("orders"),
                "target": _context("customers"),
            },
        ),
    ):
        assert attributes.find_join_path("source", "target") == [
            {
                "source_database": "cards_db",
                "source_schema": "main",
                "source_table": "orders",
                "source_column": "customer_id",
                "target_database": "cards_db",
                "target_schema": "main",
                "target_table": "customers",
                "target_column": "id",
            }
        ]
