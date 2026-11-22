# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from typing import cast

import pytest
from auto_ontology.connectors.base import SQLDatabase

from auto_ontology.retrieval.text_to_sql.connector_routing import (
    connector_named,
    databases_for_sql,
    resolve_connector_for_sql,
    resolve_connector_from_tables,
    resolve_target_database_name,
)


def _connector(database_name: str, dialect: str = "sqlite") -> SQLDatabase:
    return cast(
        SQLDatabase, SimpleNamespace(database_name=database_name, dialect=dialect)
    )


def _table(database_name: str, schema: str, name: str) -> dict:
    return {"database_name": database_name, "schema_name": schema, "name": name}


def test_sql_decides_when_the_shown_tables_span_databases() -> None:
    census, works = _connector("census"), _connector("adventureworks")
    shown = [
        _table("census", "acs", "state_2014"),
        _table("adventureworks", "main", "salesorderheader"),
    ]

    assert resolve_connector_for_sql(shown, [census, works], None) is None
    sql = "SELECT SUM(totaldue) FROM main.salesorderheader"
    assert resolve_connector_for_sql(shown, [census, works], sql) is works
    joined = "SELECT 1 FROM acs.state_2014 JOIN main.salesorderheader ON TRUE"
    assert resolve_connector_for_sql(shown, [census, works], joined) is None
    assert resolve_connector_for_sql(shown, [census, works], "SELECT 1 FROM x") is None


def test_sql_routing_ignores_cte_names() -> None:
    first, second = _connector("first"), _connector("second")
    shown = [_table("first", "main", "totals"), _table("second", "main", "orders")]
    sql = "WITH totals AS (SELECT 1 AS n) SELECT * FROM totals JOIN main.orders ON TRUE"

    assert resolve_connector_for_sql(shown, [first, second], sql) is second


def test_sql_routing_parses_bigquery_names() -> None:
    austin, works = _connector("austin", "bigquery"), _connector("adventureworks")
    shown = [
        _table("adventureworks", "main", "orders"),
        _table("austin", "austin_311", "311_service_requests"),
    ]

    for sql in (
        "SELECT COUNT(*) FROM `bigquery-public-data.austin_311.311_service_requests`",
        "SELECT COUNT(*) FROM austin_311.311_service_requests",
    ):
        assert resolve_connector_for_sql(shown, [austin, works], sql) is austin


def test_a_table_name_in_several_databases_is_not_guessed() -> None:
    shop, works = _connector("shop"), _connector("adventureworks")
    shown = [
        _table("adventureworks", "main", "orders"),
        _table("shop", "main", "orders"),
    ]
    sql = "SELECT COUNT(*) FROM main.orders"

    assert resolve_connector_for_sql(shown, [shop, works], sql) is None
    assert databases_for_sql(shown, [shop, works], sql) == ["adventureworks", "shop"]


def test_connector_named_ignores_case() -> None:
    shop, works = _connector("shop"), _connector("AdventureWorks")

    assert connector_named("adventureworks", [shop, works]) is works
    assert connector_named("missing", [shop, works]) is None


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


def test_a_shard_name_routes_to_the_database_with_its_wildcard_table() -> None:
    tables = [_table("ga360", "ga", "ga_sessions_*"), _table("shop", "main", "orders")]
    connectors = [_connector("ga360", "bigquery"), _connector("shop")]

    assert databases_for_sql(
        tables,
        connectors,
        "SELECT COUNT(*) FROM `proj.ga.ga_sessions_2017*` WHERE _TABLE_SUFFIX > '07'",
    ) == ["ga360"]
