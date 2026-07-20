import os
from typing import Any
from urllib.parse import quote

import pandas as pd
import pytest
from pytest import MonkeyPatch

from gsf.connectors.connection_string_factory import build_connection_string
from gsf.connectors.databricks import DatabricksDatabase, _parse_connection_string
from gsf.connectors.registry import create_connector


def _connection_string(token: str = "secret") -> str:
    return (
        f"databricks://token:{quote(token, safe='')}@example.databricks.com/main"
        "?http_path=%2Fsql%2F1.0%2Fwarehouses%2Fwarehouse-id"
    )


def test_parse_connection_string() -> None:
    kwargs, catalog = _parse_connection_string(_connection_string("secret/with@chars"))

    assert kwargs == {
        "server_hostname": "example.databricks.com",
        "http_path": "/sql/1.0/warehouses/warehouse-id",
        "access_token": "secret/with@chars",
        "catalog": "main",
    }
    assert catalog == "main"


def test_build_connection_string() -> None:
    connection_string = build_connection_string(
        {
            "type": "databricks",
            "host": "https://example.databricks.com",
            "http_path": "/sql/1.0/warehouses/warehouse-id",
            "password": "token/with@reserved",
            "database": "main",
            "schemas": ["analytics"],
        }
    )

    assert connection_string == (
        "databricks://token:token%2Fwith%40reserved@example.databricks.com/main"
        "?http_path=%2Fsql%2F1.0%2Fwarehouses%2Fwarehouse-id"
    )


def test_schema_selection_filters_introspection() -> None:
    database = DatabricksDatabase(_connection_string(), schemas=["analytics"])
    tables = pd.DataFrame(
        {
            "table_schema": ["default", "Analytics", "analytics"],
            "table_name": ["customers", "orders", "line_items"],
        }
    )

    filtered = database._filter_by_schema(tables)

    assert filtered["table_name"].tolist() == ["orders", "line_items"]


def test_registry_creates_databricks_connector_with_schema_selection() -> None:
    database = create_connector(_connection_string(), schemas=["analytics"])

    assert isinstance(database, DatabricksDatabase)
    assert database._schema_filter == {"analytics"}


def test_execute_returns_dataframe_and_closes_connection(
    monkeypatch: MonkeyPatch,
) -> None:
    class Cursor:
        description = [("VALUE",)]
        executed: tuple[str, list[int] | None] | None = None

        def __enter__(self) -> "Cursor":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def execute(self, statement: str, parameters: list[int] | None = None) -> None:
            self.executed = (statement, parameters)

        def fetchall(self) -> list[tuple[int]]:
            return [(1,)]

    class Connection:
        closed = False
        cursor_instance = Cursor()

        def cursor(self) -> Cursor:
            return self.cursor_instance

        def close(self) -> None:
            self.closed = True

    connection = Connection()
    monkeypatch.setattr(
        "gsf.connectors.databricks.sql.connect",
        lambda **_kwargs: connection,
    )
    database = DatabricksDatabase(_connection_string())

    result = database.execute("SELECT ?", [1])

    assert result.to_dict(orient="records") == [{"value": 1}]
    assert connection.cursor_instance.executed == ("SELECT ?", [1])
    assert connection.closed


def _live_connection() -> dict[str, Any]:
    environment_fields = {
        "host": "DATABRICKS_SERVER_HOSTNAME",
        "http_path": "DATABRICKS_HTTP_PATH",
        "password": "DATABRICKS_TOKEN",
        "database": "DATABRICKS_CATALOG",
    }
    missing = [
        environment_name
        for environment_name in environment_fields.values()
        if not os.environ.get(environment_name)
    ]
    if missing:
        pytest.skip(
            "Live Databricks credentials are not configured: " + ", ".join(missing)
        )

    connection: dict[str, Any] = {
        "type": "databricks",
        **{
            field: os.environ[environment_name]
            for field, environment_name in environment_fields.items()
        },
    }
    schema = os.environ.get("DATABRICKS_SCHEMA")
    if schema:
        connection["schemas"] = [schema]
    return connection


def test_live_databricks_connector_end_to_end() -> None:
    connection = _live_connection()
    connection_string = build_connection_string(connection)
    database = DatabricksDatabase(
        connection_string,
        schemas=connection.get("schemas"),
    )

    try:
        database.ping()
        assert database.execute("SELECT 1 AS value").iloc[0]["value"] == 1
        assert isinstance(database.get_schemas(), list)

        metadata_frames = [
            database.get_tables(),
            database.get_columns(),
            database.get_views(),
            database.get_pks(),
            database.get_fks(),
            database.get_queries(),
        ]
        assert all(isinstance(frame, pd.DataFrame) for frame in metadata_frames)
    finally:
        database.close()
