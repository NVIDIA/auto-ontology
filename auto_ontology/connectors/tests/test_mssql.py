# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import pandas as pd
import pymssql
import pytest
from pytest import MonkeyPatch

from auto_ontology.connectors.base import StatementTimeout
from auto_ontology.connectors.connection_string_factory import build_connection_string
from auto_ontology.connectors.mssql import MSSQLDatabase, _parse_connection_string
from auto_ontology.connectors.registry import CONNECTOR_REGISTRY


def test_mssql_connector_is_registered() -> None:
    assert "mssql" in CONNECTOR_REGISTRY
    assert CONNECTOR_REGISTRY["mssql"].__name__ == "MSSQLDatabase"


def test_sqlserver_alias_is_registered() -> None:
    assert CONNECTOR_REGISTRY["sqlserver"] is CONNECTOR_REGISTRY["mssql"]


def test_build_mssql_connection_string_encodes_credentials() -> None:
    connection_string = build_connection_string(
        {
            "type": "mssql",
            "host": "db.example.com",
            "user": "analyst@domain",
            "password": "p@ss/word",
            "database": "sales data",
        }
    )

    assert connection_string == (
        "mssql://analyst%40domain:p%40ss%2Fword@db.example.com:1433/sales%20data"
    )


def test_parse_mssql_connection_string_decodes_components() -> None:
    assert _parse_connection_string(
        "mssql+pymssql://analyst%40domain:p%40ss%2Fword@db.example.com:2046/sales%20data"
    ) == {
        "server": "db.example.com",
        "port": 2046,
        "user": "analyst@domain",
        "password": "p@ss/word",
        "database": "sales data",
    }


def test_parse_mssql_connection_string_requires_host_and_database() -> None:
    with pytest.raises(ValueError, match="missing a host"):
        _parse_connection_string("mssql:///sales")
    with pytest.raises(ValueError, match="missing a database"):
        _parse_connection_string("mssql://analyst:secret@db.example.com")


def test_registry_creates_read_only_mssql_connector(monkeypatch: MonkeyPatch) -> None:
    connections: list[FakeConnection] = []
    connect_kwargs: list[dict[str, Any]] = []

    def connect(**kwargs: Any) -> FakeConnection:
        connect_kwargs.append(kwargs)
        connection = FakeConnection()
        connections.append(connection)
        return connection

    monkeypatch.setattr("auto_ontology.connectors.mssql.pymssql.connect", connect)
    database = MSSQLDatabase("mssql://analyst:secret@db.example.com/sales")

    assert database.database_name == "sales"
    assert database.dialect == "tsql"
    assert connections[0].cursor_instance.executed == ("SELECT DB_NAME()", None)
    assert connect_kwargs[0]["read_only"] is True
    assert connect_kwargs[0]["autocommit"] is True
    assert connections[0].closed


def test_execute_returns_dataframe_binds_parameters_and_closes(
    monkeypatch: MonkeyPatch,
) -> None:
    connections: list[FakeConnection] = []
    connect_kwargs: list[dict[str, Any]] = []

    def connect(**kwargs: Any) -> FakeConnection:
        connect_kwargs.append(kwargs)
        connection = FakeConnection()
        connections.append(connection)
        return connection

    monkeypatch.setattr("auto_ontology.connectors.mssql.pymssql.connect", connect)
    database = MSSQLDatabase("mssql://analyst:secret@db.example.com/sales")
    result = database.execute("SELECT %s AS value", ["hello"], timeout_s=1.2)

    assert isinstance(result, pd.DataFrame)
    assert result.to_dict(orient="records") == [{"value": "hello"}]
    assert connections[1].cursor_instance.executed == (
        "SELECT %s AS value",
        ("hello",),
    )
    assert connect_kwargs[1]["timeout"] == 2
    assert connect_kwargs[1]["read_only"] is True
    assert connections[1].cursor_instance.closed
    assert connections[1].closed


def test_metadata_methods_return_expected_empty_shapes(
    monkeypatch: MonkeyPatch,
) -> None:
    statements: list[str] = []

    def connect(**_kwargs: Any) -> FakeConnection:
        return FakeConnection(statements)

    monkeypatch.setattr("auto_ontology.connectors.mssql.pymssql.connect", connect)
    database = MSSQLDatabase("mssql://analyst:secret@db.example.com/sales")
    statements.clear()

    assert list(database.get_queries().columns) == ["end_time", "query_text"]
    for method in (
        database.get_tables,
        database.get_columns,
        database.get_views,
        database.get_pks,
        database.get_fks,
    ):
        assert isinstance(method(), pd.DataFrame)

    assert "INFORMATION_SCHEMA.TABLES" in statements[0]
    assert "'view'" in statements[0]
    assert "'base table'" in statements[0]
    assert "IS_NULLABLE" in statements[1]
    assert "AS BIT" in statements[1]
    assert "INFORMATION_SCHEMA.VIEWS" in statements[2]
    assert "PRIMARY KEY" in statements[3]
    assert "UNIQUE_CONSTRAINT_NAME" in statements[4]


def test_execute_maps_driver_query_timeout(monkeypatch: MonkeyPatch) -> None:
    calls = 0

    def connect(**_kwargs: Any) -> FakeConnection:
        nonlocal calls
        calls += 1
        if calls == 1:
            return FakeConnection()
        return FakeConnection(cursor=TimeoutCursor())

    monkeypatch.setattr("auto_ontology.connectors.mssql.pymssql.connect", connect)
    database = MSSQLDatabase("mssql://analyst:secret@db.example.com/sales")

    with pytest.raises(StatementTimeout):
        database.execute("SELECT 1", timeout_s=1)


class FakeCursor:
    description = [("value",)]
    closed = False

    def __init__(self, statements: list[str] | None = None) -> None:
        self.statements = statements
        self.executed: tuple[str, tuple[Any, ...] | None] | None = None

    def execute(
        self, statement: str, parameters: tuple[Any, ...] | None = None
    ) -> None:
        self.executed = (statement, parameters)
        if self.statements is not None:
            self.statements.append(statement)

    def fetchone(self) -> tuple[str]:
        return ("sales",)

    def fetchall(self) -> list[dict[str, str]]:
        return [{"value": "hello"}]

    def close(self) -> None:
        self.closed = True


class FakeConnection:
    def __init__(
        self,
        statements: list[str] | None = None,
        cursor: FakeCursor | None = None,
    ) -> None:
        self.cursor_instance = cursor or FakeCursor(statements)
        self.closed = False

    def cursor(self, *, as_dict: bool = False) -> FakeCursor:
        return self.cursor_instance

    def close(self) -> None:
        self.closed = True


class TimeoutCursor(FakeCursor):
    def execute(
        self, statement: str, parameters: tuple[Any, ...] | None = None
    ) -> None:
        raise pymssql.OperationalError("query timeout")
