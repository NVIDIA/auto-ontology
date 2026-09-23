# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from typing import cast

import pytest
from auto_ontology.connectors.base import SQLDatabase

from auto_ontology.retrieval.text_to_sql.connector_routing import (
    resolve_connector_from_tables,
    resolve_target_database_name,
)


def _connector(database_name: str) -> SQLDatabase:
    return cast(SQLDatabase, SimpleNamespace(database_name=database_name))


def test_routes_all_tables_to_their_database() -> None:
    first = _connector("first")
    second = _connector("second")

    resolved = resolve_connector_from_tables(
        [
            {"name": "orders", "database_name": "second"},
            {"name": "customers", "database_name": "second"},
        ],
        [first, second],
    )

    assert resolved is second


def test_rejects_missing_database_with_multiple_connectors() -> None:
    connectors = [_connector("first"), _connector("second")]

    with pytest.raises(ValueError, match="do not identify a database"):
        resolve_connector_from_tables([{"name": "orders"}], connectors)


def test_allows_missing_database_with_one_connector() -> None:
    connector = _connector("only")

    assert resolve_connector_from_tables([{"name": "orders"}], [connector]) is connector


def test_resolves_target_database_to_canonical_connector_name() -> None:
    connectors = [_connector("wideworldimporters")]

    assert (
        resolve_target_database_name("WIDEWORLDIMPORTERS", connectors)
        == "wideworldimporters"
    )


def test_rejects_unknown_target_database() -> None:
    connectors = [_connector("wideworldimporters")]

    with pytest.raises(ValueError, match="does not match a configured database"):
        resolve_target_database_name(
            "42d62c22-b1c4-4ed7-8173-6aec9e2bd432",
            connectors,
        )
