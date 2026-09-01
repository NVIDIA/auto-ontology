import os
from typing import Any

import pandas as pd
import pytest
from pytest import MonkeyPatch

from gsf.connectors.connection_string_factory import build_connection_string
from gsf.connectors.mysql import MySQLDatabase, _parse_connection_string
from gsf.connectors.registry import create_connector


def _connection_string(password: str = "secret") -> str:
    return f"mysql://user:{password}@localhost:3306/beaver"


def test_parse_connection_string() -> None:
    kwargs = _parse_connection_string(
        "mysql://user%40domain:secret%2Fvalue@example.com:3307/beaver%20db"
    )

    assert kwargs == {
        "host": "example.com",
        "port": 3307,
        "user": "user@domain",
        "password": "secret/value",
        "database": "beaver db",
    }


def test_build_connection_string() -> None:
    connection_string = build_connection_string(
        {
            "type": "mysql",
            "host": "localhost",
            "port": "3307",
            "user": "user@domain",
            "password": "secret/value",
            "database": "beaver db",
        }
    )

    assert (
        connection_string
        == "mysql://user%40domain:secret%2Fvalue@localhost:3307/beaver%20db"
    )


def test_registry_creates_mysql_connector(monkeypatch: MonkeyPatch) -> None:
    connections: list[Connection] = []

    def connect(**_kwargs: Any) -> "Connection":
        connection = Connection()
        connections.append(connection)
        return connection

    monkeypatch.setattr("gsf.connectors.mysql.mysql.connector.connect", connect)

    database = create_connector(_connection_string())

    assert isinstance(database, MySQLDatabase)
    assert database.dialect == "mysql"
    assert database.database_name == "beaver"
    assert connections[0].pinged
    assert connections[0].closed


def test_execute_returns_dataframe_and_closes_connection(
    monkeypatch: MonkeyPatch,
) -> None:
    connections: list[Connection] = []

    def connect(**kwargs: Any) -> "Connection":
        connection = Connection(connect_kwargs=kwargs)
        connections.append(connection)
        return connection

    monkeypatch.setattr("gsf.connectors.mysql.mysql.connector.connect", connect)
    database = MySQLDatabase(_connection_string())

    result = database.execute("SELECT %s AS value", [1])

    assert result.to_dict(orient="records") == [{"value": 1}]
    assert connections[1].connect_kwargs["database"] == "beaver"
    assert connections[1].cursor_instance.executed == ("SELECT %s AS value", [1])
    assert connections[1].closed


def _live_connection() -> dict[str, Any]:
    # Opt in explicitly so pytest never dials a real server just because MYSQL_*
    # happens to be exported in the shell.
    if os.environ.get("GSF_LIVE_MYSQL") != "1":
        pytest.skip("Set GSF_LIVE_MYSQL=1 to run the live MySQL test")

    environment_fields = {
        "host": "MYSQL_HOST",
        "user": "MYSQL_USER",
        "password": "MYSQL_PASSWORD",
        "database": "MYSQL_DATABASE",
    }
    missing = [
        environment_name
        for environment_name in environment_fields.values()
        if not os.environ.get(environment_name)
    ]
    if missing:
        pytest.skip("Live MySQL credentials are not configured: " + ", ".join(missing))

    connection: dict[str, Any] = {
        "type": "mysql",
        **{
            field: os.environ[environment_name]
            for field, environment_name in environment_fields.items()
        },
    }
    port = os.environ.get("MYSQL_PORT")
    if port:
        connection["port"] = port
    return connection


def test_live_mysql_connector_end_to_end() -> None:
    connection = _live_connection()
    database = create_connector(build_connection_string(connection))

    try:
        assert isinstance(database, MySQLDatabase)
        assert database.dialect == "mysql"
        assert database.database_name == connection["database"]

        database.ping()
        assert database.execute("SELECT 1 AS value").iloc[0]["value"] == 1
        assert database.execute("SELECT %s AS value", ["x"]).iloc[0]["value"] == "x"

        metadata_frames = [
            database.get_tables(),
            database.get_columns(),
            database.get_views(),
            database.get_pks(),
            database.get_fks(),
            database.get_queries(),
        ]
        assert all(isinstance(frame, pd.DataFrame) for frame in metadata_frames)

        tables = database.get_tables()
        assert not tables.empty
        assert set(tables.columns) == {"table_schema", "table_name", "table_type"}
        # MySQL has no schema namespace, so the catalog reports every table's
        # TABLE_SCHEMA as the connected database. This is what makes
        # ``qualify_table`` collapse the duplicate rather than emit db.db.table.
        assert set(tables["table_schema"]) == {database.database_name}
        assert set(tables["table_type"]) <= {"base table", "view"}

        columns = database.get_columns()
        assert not columns.empty
        assert set(columns["table_name"]) <= set(tables["table_name"])

        # A database-qualified reference must run; a schema-qualified one built
        # by naively joining all three catalog parts must not.
        table_name = str(tables.iloc[0]["table_name"])
        qualified = f"`{database.database_name}`.`{table_name}`"
        assert (
            database.execute(f"SELECT COUNT(*) AS n FROM {qualified}").iloc[0]["n"] >= 0
        )
    finally:
        database.close()


class Cursor:
    description = [("value",)]
    executed: tuple[str, list[int] | None] | None = None
    closed = False

    def execute(self, statement: str, parameters: list[int] | None = None) -> None:
        self.executed = (statement, parameters)

    def fetchall(self) -> list[dict[str, int]]:
        return [{"value": 1}]

    def close(self) -> None:
        self.closed = True


class Connection:
    def __init__(self, connect_kwargs: dict[str, Any] | None = None) -> None:
        self.connect_kwargs = connect_kwargs or {}
        self.cursor_instance = Cursor()
        self.pinged = False
        self.closed = False

    def cursor(self, *, dictionary: bool = False) -> Cursor:
        assert dictionary
        return self.cursor_instance

    def ping(self, *, reconnect: bool = False) -> None:
        assert not reconnect
        self.pinged = True

    def close(self) -> None:
        self.closed = True
