# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from auto_ontology.connectors.registry import (
    has_connector_for_database,
    parse_connection_strings,
)
from auto_ontology.dal import connections


def test_parse_connection_strings_accepts_commas_and_newlines() -> None:
    raw = """
        postgresql://localhost/first
        clickhouse://localhost/second,

        snowflake://account?database=third
    """

    assert parse_connection_strings(raw) == [
        "postgresql://localhost/first",
        "clickhouse://localhost/second",
        "snowflake://account?database=third",
    ]


@pytest.mark.parametrize(
    ("url", "name"),
    [
        ("postgres://user@host/default?dbname=Sales%20Data", "Sales Data"),
        ("mysql://user:pass@host/Sales%20Data", "Sales Data"),
        ("snowflake://user:pass@account/Sales%20Data?warehouse=wh", "Sales Data"),
        (
            "databricks://token:pass@host/Sales%20Data?http_path=/sql/warehouse",
            "Sales Data",
        ),
        ("duckdb:///path/to/report.archive.duckdb", "report"),
        ("duckdb:///path/to/temp.duckdb", "temp_db"),
        ("sqlite:///path/to/report.archive.sqlite", "report.archive"),
        ("clickhouse://host/Sales%20Data", "Sales Data"),
        ("heavydb://user:pass@host/Sales%20Data", "Sales Data"),
        ("kyuubi://user:pass@host?auth=token&catalog=Sales%20Data", "Sales Data"),
        ("trino://user@host?catalog=Sales%20Data", "Sales Data"),
        ("bigquery://project/Sales%20Data", "Sales Data"),
        ("bigquery://project", "project"),
    ],
)
def test_configured_connector_uses_its_catalog_name(
    monkeypatch: pytest.MonkeyPatch, url: str, name: str
) -> None:
    monkeypatch.setattr(connections, "list_connections", lambda: [])
    # One malformed URL must not stop the following connection from matching.
    monkeypatch.setenv("CONNECTION_STRINGS", f"postgresql://host?invalid=secret,{url}")

    assert has_connector_for_database(name)
    assert not has_connector_for_database(name.upper())


def test_configured_connector_reads_saved_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        connections,
        "list_connections",
        lambda: [
            {
                "type": "postgres",
                "host": "unreachable",
                "database": "Sales Data",
                "user": "user",
                "password": "password",
            }
        ],
    )
    monkeypatch.setenv("CONNECTION_STRINGS", "postgresql://host/env_only")

    assert has_connector_for_database("Sales Data")
    assert not has_connector_for_database("env_only")


def test_configured_postgres_connector_honors_libpq_database_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(connections, "list_connections", lambda: [])
    monkeypatch.setenv("CONNECTION_STRINGS", "postgresql://analyst@host")
    monkeypatch.delenv("PGSERVICE", raising=False)
    monkeypatch.setenv("PGDATABASE", "reporting")

    assert has_connector_for_database("reporting")
    assert not has_connector_for_database("analyst")

    monkeypatch.delenv("PGDATABASE")
    assert has_connector_for_database("analyst")
