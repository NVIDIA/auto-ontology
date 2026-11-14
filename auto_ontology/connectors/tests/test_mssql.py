# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
from typing import Any

import pandas as pd
import pymssql
import pytest
from pytest import MonkeyPatch
from pathlib import Path

from auto_ontology.connectors.base import StatementTimeout
from auto_ontology.connectors.connection_string_factory import build_connection_string
from auto_ontology.connectors.mssql import (
    MSSQLDatabase,
    _MSSQL_OPERATION_LOCK,
    _parse_connection_string,
)
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
    assert connect_kwargs[0]["encryption"] == "require"
    assert "timeout" not in connect_kwargs[0]
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
    assert "timeout" not in connect_kwargs[1]
    assert connect_kwargs[1]["read_only"] is True
    assert connections[1].query_timeouts == [2, 0]
    assert connections[1].cursor_instance.as_dict is False
    assert connections[1].cursor_instance.closed
    assert connections[1].closed


@pytest.mark.parametrize(
    ("description", "row", "expected_columns"),
    [
        ([("",)], [(9,)], [""]),
        ([("id",), ("id",)], [(1, 2)], ["id", "id"]),
    ],
    ids=["unnamed-expression", "duplicate-column-name"],
)
def test_execute_preserves_positional_result_values(
    monkeypatch: MonkeyPatch,
    description: list[tuple[str]],
    row: list[tuple[int, ...]],
    expected_columns: list[str],
) -> None:
    def connect(**_kwargs: Any) -> FakeConnection:
        return FakeConnection(cursor=FakeCursor(description=description, rows=row))

    monkeypatch.setattr("auto_ontology.connectors.mssql.pymssql.connect", connect)
    database = MSSQLDatabase("mssql://analyst:secret@db.example.com/sales")

    result = database.execute("SELECT 1, 2")

    assert result.columns.tolist() == expected_columns
    assert result.iloc[0].tolist() == list(row[0])


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
    assert "sys.foreign_key_columns" in statements[4]
    assert "parent_column_id" in statements[4]
    assert "referenced_column_id" in statements[4]
    assert "INFORMATION_SCHEMA.KEY_COLUMN_USAGE" not in statements[4]


def test_get_fks_discovers_reference_to_standalone_unique_index(
    monkeypatch: MonkeyPatch,
) -> None:
    statements: list[str] = []
    fk_row = (
        "dbo",
        "orders",
        "customer_code",
        "dbo",
        "customers",
        "external_code",
    )

    def connect(**_kwargs: Any) -> FakeConnection:
        cursor = FakeCursor(
            statements,
            description=[
                ("table_schema",),
                ("table_name",),
                ("column_name",),
                ("referenced_schema",),
                ("referenced_table",),
                ("referenced_column",),
            ],
            rows=[fk_row],
        )
        return FakeConnection(statements, cursor=cursor)

    monkeypatch.setattr("auto_ontology.connectors.mssql.pymssql.connect", connect)
    database = MSSQLDatabase("mssql://analyst:secret@db.example.com/sales")
    statements.clear()

    result = database.get_fks()

    assert result.to_dict(orient="records") == [
        {
            "table_schema": "dbo",
            "table_name": "orders",
            "column_name": "customer_code",
            "referenced_schema": "dbo",
            "referenced_table": "customers",
            "referenced_column": "external_code",
        }
    ]
    sql = statements[0]
    assert "sys.foreign_key_columns" in sql
    assert "fk.referenced_column_id = target_column.column_id" in sql


def test_freetds_config_requires_trusted_hostname_validation() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    config = (repo_root / "freetds.conf").read_text(encoding="utf-8").lower()

    assert "ca file = system" in config
    assert "check certificate hostname = yes" in config
    assert "certificate hostname = host setting" in config
    dockerfile = (repo_root / "Dockerfile").read_text(encoding="utf-8")
    assert "FREETDS=/etc/freetds/freetds.conf" in dockerfile


def test_execute_maps_driver_query_timeout(monkeypatch: MonkeyPatch) -> None:
    calls = 0
    connections: list[FakeConnection] = []

    def connect(**_kwargs: Any) -> FakeConnection:
        nonlocal calls
        calls += 1
        if calls == 1:
            connection = FakeConnection()
        else:
            connection = FakeConnection(cursor=TimeoutCursor())
        connections.append(connection)
        return connection

    monkeypatch.setattr("auto_ontology.connectors.mssql.pymssql.connect", connect)
    database = MSSQLDatabase("mssql://analyst:secret@db.example.com/sales")

    with pytest.raises(StatementTimeout):
        database.execute("SELECT 1", timeout_s=1)

    assert connections[-1].query_timeouts == [1, 0]


def test_uncapped_operation_resets_timeout_after_previous_capped_call(
    monkeypatch: MonkeyPatch,
) -> None:
    connections: list[FakeConnection] = []

    def connect(**_kwargs: Any) -> FakeConnection:
        connection = FakeConnection()
        connections.append(connection)
        return connection

    monkeypatch.setattr("auto_ontology.connectors.mssql.pymssql.connect", connect)
    database = MSSQLDatabase("mssql://analyst:secret@db.example.com/sales")

    database.execute("SELECT 1", timeout_s=2)
    database.execute("SELECT 2")

    assert connections[1].query_timeouts == [2, 0]
    assert connections[2].query_timeouts == [0, 0]


def test_mssql_operations_share_a_process_wide_timeout_lock(
    monkeypatch: MonkeyPatch,
) -> None:
    connect_started = threading.Event()
    connections: list[FakeConnection] = []

    def connect(**_kwargs: Any) -> FakeConnection:
        connect_started.set()
        connection = FakeConnection()
        connections.append(connection)
        return connection

    monkeypatch.setattr("auto_ontology.connectors.mssql.pymssql.connect", connect)
    database = MSSQLDatabase("mssql://analyst:secret@db.example.com/sales")
    connect_started.clear()
    operation_started = threading.Event()

    def execute() -> None:
        operation_started.set()
        database.execute("SELECT 1")

    with _MSSQL_OPERATION_LOCK:
        worker = threading.Thread(target=execute)
        worker.start()
        assert operation_started.wait(timeout=1)
        assert not connect_started.wait(timeout=0.05)

    assert connect_started.wait(timeout=1)
    worker.join(timeout=1)
    assert not worker.is_alive()


class FakeCursor:
    def __init__(
        self,
        statements: list[str] | None = None,
        *,
        description: list[tuple[str]] | None = None,
        rows: list[tuple[Any, ...]] | None = None,
    ) -> None:
        self.statements = statements
        self.description = description if description is not None else [("value",)]
        self.rows = rows if rows is not None else [("hello",)]
        self.as_dict = False
        self.closed = False
        self.executed: tuple[str, tuple[Any, ...] | None] | None = None

    def execute(
        self, statement: str, parameters: tuple[Any, ...] | None = None
    ) -> None:
        self.executed = (statement, parameters)
        if self.statements is not None:
            self.statements.append(statement)

    def fetchone(self) -> tuple[str]:
        return ("sales",)

    def fetchall(self) -> list[tuple[Any, ...]]:
        return self.rows

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
        self.query_timeouts: list[int] = []
        self._conn = FakeMSSQLConnection(self.query_timeouts)

    def cursor(self, *, as_dict: bool = False) -> FakeCursor:
        self.cursor_instance.as_dict = as_dict
        return self.cursor_instance

    def close(self) -> None:
        self.closed = True


class FakeMSSQLConnection:
    def __init__(self, query_timeouts: list[int]) -> None:
        self._query_timeouts = query_timeouts

    @property
    def query_timeout(self) -> int:
        return self._query_timeouts[-1] if self._query_timeouts else 0

    @query_timeout.setter
    def query_timeout(self, value: int) -> None:
        self._query_timeouts.append(value)


class TimeoutCursor(FakeCursor):
    def execute(
        self, statement: str, parameters: tuple[Any, ...] | None = None
    ) -> None:
        raise pymssql.OperationalError("query timeout")
