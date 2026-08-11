import os
from contextlib import contextmanager
from typing import Any
from urllib.parse import quote

import pandas as pd
import pytest
from pytest import MonkeyPatch

from nemo_retriever.tabular_data.ingestion.model.reserved_words import TableTypes

from gsf.connectors.connection_string_factory import build_connection_string
from gsf.connectors.databricks import (
    AUTH_SSO_FEDERATION,
    AUTH_STORED_TOKEN,
    DatabricksDatabase,
    _parse_connection_string,
)

from gsf.connectors.registry import create_connector


def _connection_string(token: str = "secret") -> str:
    return (
        f"databricks://token:{quote(token, safe='')}@example.databricks.com/main"
        "?http_path=%2Fsql%2F1.0%2Fwarehouses%2Fwarehouse-id"
    )


def test_parse_connection_string() -> None:
    kwargs, catalog, auth_mode = _parse_connection_string(
        _connection_string("secret/with@chars")
    )

    assert kwargs == {
        "server_hostname": "example.databricks.com",
        "http_path": "/sql/1.0/warehouses/warehouse-id",
        "access_token": "secret/with@chars",
        "catalog": "main",
        "enable_telemetry": False,
    }
    assert catalog == "main"
    assert auth_mode == AUTH_STORED_TOKEN


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


def test_parse_connection_string_marks_sso_federation() -> None:
    _kwargs, _catalog, auth_mode = _parse_connection_string(
        _connection_string() + "&auth=sso"
    )

    assert auth_mode == AUTH_SSO_FEDERATION


def test_build_connection_string_flags_exchanged_token() -> None:
    """An exchanged token is marked so the connector can log which one it used."""
    federated = build_connection_string(
        {
            "type": "databricks",
            "host": "example.databricks.com",
            "http_path": "/sql/1.0/warehouses/w",
            "password": "stored-pat",
            "database": "main",
            "access_token_override": "exchanged",
        }
    )
    stored = build_connection_string(
        {
            "type": "databricks",
            "host": "example.databricks.com",
            "http_path": "/sql/1.0/warehouses/w",
            "password": "stored-pat",
            "database": "main",
        }
    )

    assert federated.endswith("&auth=sso")
    assert "auth=sso" not in stored
    assert _parse_connection_string(federated)[2] == AUTH_SSO_FEDERATION
    assert _parse_connection_string(stored)[2] == AUTH_STORED_TOKEN


def test_auth_mode_property_reports_credential() -> None:
    assert DatabricksDatabase(_connection_string()).auth_mode == AUTH_STORED_TOKEN
    assert (
        DatabricksDatabase(_connection_string() + "&auth=sso").auth_mode
        == AUTH_SSO_FEDERATION
    )


def test_schema_condition_uses_equals_for_one_schema_and_in_for_many() -> None:
    """One schema is the common case and gets the simplest, most prunable form."""
    one = DatabricksDatabase(_connection_string(), schemas=["nbu_dmt_explorer"])
    assert one._schema_condition() == "table_schema = 'nbu_dmt_explorer'"

    many = DatabricksDatabase(_connection_string(), schemas=["b_two", "a_one"])
    assert many._schema_condition() == "table_schema IN ('a_one', 'b_two')"


def test_schema_condition_drops_information_schema_exclusion_when_filtered() -> None:
    """An allowlist can never name information_schema, so excluding it too is dead
    weight — and a second predicate only gives the planner more to reason about."""
    filtered = DatabricksDatabase(_connection_string(), schemas=["nbu_dmt_explorer"])
    assert "information_schema" not in filtered._schema_condition()

    # Catalog-wide scans still exclude Databricks' own metadata schema.
    assert (
        DatabricksDatabase(_connection_string())._schema_condition()
        == "table_schema != 'information_schema'"
    )


def test_schema_match_never_wraps_the_column_in_a_function() -> None:
    """lower() would make the comparison non-sargable and defeat metadata pruning."""
    for schemas in (["nbu_dmt_explorer"], ["b_two", "a_one"], None):
        db = DatabricksDatabase(_connection_string(), schemas=schemas)
        assert "lower(" not in db._schema_condition()
        assert "lower(" not in db._schema_predicate()


def test_schema_predicate_is_and_prefixed_for_constraint_queries() -> None:
    """get_pks/get_fks already have a WHERE, so the match arrives as an AND clause."""
    db = DatabricksDatabase(_connection_string(), schemas=["nbu_dmt_explorer"])
    assert db._schema_predicate("keys.table_schema") == (
        "AND keys.table_schema = 'nbu_dmt_explorer'"
    )
    assert DatabricksDatabase(_connection_string())._schema_predicate() == ""


def test_schema_condition_escapes_quotes() -> None:
    db = DatabricksDatabase(_connection_string(), schemas=["a'b"])
    assert db._schema_condition() == "table_schema = 'a''b'"


def test_single_schema_lists_tables_with_show_tables() -> None:
    """SHOW TABLES is scoped to one schema; information_schema spans the catalog.

    SHOW TABLES lists views alongside tables and carries no type column, so a
    second SHOW VIEWS supplies the names to classify — otherwise a view would be
    ingested as a base table.
    """
    db = DatabricksDatabase(_connection_string(), schemas=["nbu_dmt_explorer"])
    issued: list[str] = []

    def fake_execute(sql_text: str, *args: Any, **kwargs: Any) -> pd.DataFrame:
        normalised = " ".join(sql_text.split())
        issued.append(normalised)
        if normalised.startswith("SHOW VIEWS"):
            return pd.DataFrame(
                {
                    "namespace": ["nbu_dmt_explorer"],
                    "viewname": ["customers"],
                    "istemporary": [False],
                }
            )
        return pd.DataFrame(
            {
                "database": ["nbu_dmt_explorer", "nbu_dmt_explorer"],
                "tablename": ["orders", "customers"],
                "istemporary": [False, False],
            }
        )

    db.execute = fake_execute  # type: ignore[method-assign]
    tables = db.get_tables()

    assert issued == [
        "SHOW TABLES IN `main`.`nbu_dmt_explorer`",
        "SHOW VIEWS IN `main`.`nbu_dmt_explorer`",
    ]
    assert not any("information_schema" in q for q in issued)
    assert list(tables["table_name"]) == ["orders", "customers"]
    assert list(tables["table_schema"]) == ["nbu_dmt_explorer"] * 2
    # `customers` is in the SHOW VIEWS result, so only `orders` is a base table.
    assert list(tables["table_type"]) == [TableTypes.BASE_TABLE, TableTypes.VIEW]


def test_single_schema_describes_each_table_on_one_connection() -> None:
    """All statements share ONE connection: opening one costs ~0.9s and can stall for
    minutes, so a connect per table would dominate the run."""
    db = DatabricksDatabase(_connection_string(), schemas=["nbu_dmt_explorer"])
    issued: list[str] = []
    connects = 0

    @contextmanager
    def fake_connect(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal connects
        connects += 1
        yield object()

    def fake_run(_conn: Any, sql_text: str, *args: Any, **kwargs: Any) -> pd.DataFrame:
        issued.append(" ".join(sql_text.split()))
        if sql_text.startswith("SHOW TABLES"):
            return pd.DataFrame(
                {"database": ["nbu_dmt_explorer"], "tablename": ["orders"]}
            )
        return pd.DataFrame(
            {
                "col_name": ["order_id", "placed_at", "", "# Partition Information"],
                "data_type": ["bigint", "timestamp", "", ""],
                "comment": ["the order id", None, "", ""],
            }
        )

    db._connect = fake_connect  # type: ignore[method-assign]
    db._run = fake_run  # type: ignore[method-assign]
    columns = db.get_columns()

    assert connects == 1
    assert issued[0].startswith("SHOW TABLES IN")
    assert issued[1] == "DESCRIBE TABLE EXTENDED `main`.`nbu_dmt_explorer`.`orders`"
    assert not any("information_schema" in q for q in issued)

    # Trailing metadata sections are not columns.
    assert list(columns["column_name"]) == ["order_id", "placed_at"]
    assert list(columns["ordinal_position"]) == [1, 2]
    # DESCRIBE carries the type, which SHOW COLUMNS omits entirely.
    assert list(columns["data_type"]) == ["bigint", "timestamp"]
    # ...and the comment, which normalize_columns types as the description.
    assert list(columns["description"])[0] == "the order id"
    # Nullability is reported by neither, so it is left for normalize_columns to fill.
    assert "is_nullable" not in columns.columns


def test_multiple_schemas_still_use_information_schema() -> None:
    """SHOW is per-schema, so more than one falls back to the catalog-wide query."""
    db = DatabricksDatabase(_connection_string(), schemas=["a_one", "b_two"])
    issued: list[str] = []

    def fake_execute(sql_text: str, *args: Any, **kwargs: Any) -> pd.DataFrame:
        issued.append(" ".join(sql_text.split()))
        return pd.DataFrame(columns=["table_schema", "table_name"])

    db.execute = fake_execute  # type: ignore[method-assign]
    db.get_tables()
    db.get_columns()

    assert all("information_schema" in q for q in issued)
    assert all("ORDER BY" in q for q in issued)
    assert not any("LIMIT" in q for q in issued)


def test_show_columns_skips_a_table_it_cannot_read() -> None:
    """A dropped table or a missing grant must not abandon the rest of the schema."""
    from databricks.sql.exc import Error as DatabricksError

    db = DatabricksDatabase(_connection_string(), schemas=["s"])

    @contextmanager
    def fake_connect(*_args: Any, **_kwargs: Any) -> Any:
        yield object()

    def fake_run(_conn: Any, sql_text: str, *args: Any, **kwargs: Any) -> pd.DataFrame:
        if sql_text.startswith("SHOW TABLES"):
            return pd.DataFrame({"database": ["s", "s"], "tablename": ["ok", "denied"]})
        if "`denied`" in sql_text:
            raise DatabricksError("permission denied")
        return pd.DataFrame({"col_name": ["id"], "data_type": ["bigint"]})

    db._connect = fake_connect  # type: ignore[method-assign]
    db._run = fake_run  # type: ignore[method-assign]
    columns = db.get_columns()

    assert list(columns["table_name"]) == ["ok"]


def _describe_rows(constraints: list[tuple[str, str]]) -> pd.DataFrame:
    """DESCRIBE TABLE EXTENDED output: columns, then metadata sections."""
    rows = [
        ("order_id", "bigint", "the order id"),
        ("customer_id", "bigint", None),
        ("", "", ""),
        ("# Detailed Table Information", "", ""),
        ("Catalog", "main", ""),
    ]
    if constraints:
        rows.append(("# Constraints", "", ""))
        rows.extend((name, definition, "") for name, definition in constraints)
    return pd.DataFrame(rows, columns=["col_name", "data_type", "comment"])


def _describing_db(constraints: list[tuple[str, str]]) -> tuple[Any, list[str]]:
    db = DatabricksDatabase(_connection_string(), schemas=["s"])
    issued: list[str] = []

    @contextmanager
    def fake_connect(*_args: Any, **_kwargs: Any) -> Any:
        yield object()

    def fake_run(_conn: Any, sql_text: str, *args: Any, **kwargs: Any) -> pd.DataFrame:
        issued.append(" ".join(sql_text.split()))
        if sql_text.startswith("SHOW TABLES"):
            return pd.DataFrame({"database": ["s"], "tablename": ["orders"]})
        return _describe_rows(constraints)

    db._connect = fake_connect  # type: ignore[method-assign]
    db._run = fake_run  # type: ignore[method-assign]
    return db, issued


def test_primary_key_is_read_from_the_describe_constraints_section() -> None:
    """Replaces the information_schema join that was the slowest statement left."""
    db, issued = _describing_db([("pk_orders", "PRIMARY KEY (`order_id`)")])

    pks = db.get_pks()

    assert not any("information_schema" in q for q in issued)
    assert pks.to_dict("records") == [
        {
            "table_schema": "s",
            "table_name": "orders",
            "column_name": "order_id",
            "ordinal_position": 1,
        }
    ]


def test_foreign_key_is_read_from_the_describe_constraints_section() -> None:
    db, _issued = _describing_db(
        [
            (
                "fk_customer",
                "FOREIGN KEY (`customer_id`) REFERENCES `main`.`s`.`customers` (`id`)",
            )
        ]
    )

    fks = db.get_fks()

    assert fks.to_dict("records") == [
        {
            "table_schema": "s",
            "table_name": "orders",
            "column_name": "customer_id",
            "referenced_schema": "s",
            "referenced_table": "customers",
            "referenced_column": "id",
        }
    ]


def test_composite_keys_pair_columns_positionally() -> None:
    db, _issued = _describing_db(
        [
            ("pk_x", "PRIMARY KEY (`a`, `b`)"),
            ("fk_y", "FOREIGN KEY (`a`, `b`) REFERENCES `c`.`s`.`t` (`x`, `y`)"),
        ]
    )

    assert list(db.get_pks()["ordinal_position"]) == [1, 2]
    fks = db.get_fks()
    assert list(fks["column_name"]) == ["a", "b"]
    assert list(fks["referenced_column"]) == ["x", "y"]


def test_a_table_without_constraints_yields_empty_key_frames() -> None:
    """Unity Catalog constraints are informational and often simply absent."""
    db, _issued = _describing_db([])

    assert db.get_pks().empty
    assert db.get_fks().empty
    # The contract's columns still exist so downstream normalization is unaffected.
    assert list(db.get_pks().columns) == DatabricksDatabase._PK_FIELDS


def test_describe_pass_runs_once_for_columns_pks_and_fks() -> None:
    """The extract operator asks for all three separately; describing every table
    three times would triple the cost."""
    db, issued = _describing_db([("pk_orders", "PRIMARY KEY (`order_id`)")])

    db.get_columns()
    db.get_pks()
    db.get_fks()

    # One SHOW TABLES + one DESCRIBE, not three of each.
    assert issued == [
        "SHOW TABLES IN `main`.`s`",
        "DESCRIBE TABLE EXTENDED `main`.`s`.`orders`",
    ]


def test_get_tables_resets_the_describe_cache() -> None:
    """Each ingest starts with get_tables, so that is where staleness is cleared."""
    db, _issued = _describing_db([])
    db.get_columns()
    assert db._describe_cache is not None

    db._single_schema = lambda: None  # type: ignore[method-assign]
    db.execute = lambda *a, **k: pd.DataFrame(  # type: ignore[method-assign]
        columns=["table_schema", "table_name"]
    )
    db.get_tables()

    assert db._describe_cache is None


def test_get_queries_returns_empty_without_querying() -> None:
    """system.query.history is a whole-workspace scan and only enriches ingestion."""
    db = DatabricksDatabase(_connection_string(), schemas=["s"])

    def fail(*_args: Any, **_kwargs: Any) -> pd.DataFrame:
        raise AssertionError("get_queries must not run a statement")

    db.execute = fail  # type: ignore[method-assign]

    assert db.get_queries().empty
    assert list(db.get_queries().columns) == ["end_time", "query_text"]


def test_single_schema_lists_views_with_show_views() -> None:
    """information_schema.views is a catalog-wide read; SHOW VIEWS is scoped."""
    db = DatabricksDatabase(_connection_string(), schemas=["nbu_dmt_explorer"])
    issued: list[str] = []

    def fake_execute(sql_text: str, *args: Any, **kwargs: Any) -> pd.DataFrame:
        issued.append(" ".join(sql_text.split()))
        return pd.DataFrame(
            {
                "namespace": ["nbu_dmt_explorer"],
                "viewname": ["v_orders"],
                "istemporary": [False],
            }
        )

    db.execute = fake_execute  # type: ignore[method-assign]
    views = db.get_views()

    assert issued == ["SHOW VIEWS IN `main`.`nbu_dmt_explorer`"]
    assert not any("information_schema" in q for q in issued)
    assert list(views["table_name"]) == ["v_orders"]
    assert list(views["table_schema"]) == ["nbu_dmt_explorer"]
    # SHOW VIEWS reports no definition; nothing consumes it today.
    assert views["view_definition"].isna().all()


def test_multiple_schemas_still_read_views_from_information_schema() -> None:
    db = DatabricksDatabase(_connection_string(), schemas=["a_one", "b_two"])
    issued: list[str] = []

    def fake_execute(sql_text: str, *args: Any, **kwargs: Any) -> pd.DataFrame:
        issued.append(" ".join(sql_text.split()))
        return pd.DataFrame(columns=["table_schema", "table_name", "view_definition"])

    db.execute = fake_execute  # type: ignore[method-assign]
    db.get_views()

    assert "information_schema.views" in issued[0]


def test_reuse_connection_opens_once_for_every_statement(
    monkeypatch: MonkeyPatch,
) -> None:
    """Connecting is the slowest and least reliable step, so a run pays for it once."""
    opened = []

    class FakeConnection:
        def __init__(self) -> None:
            self.closed = False

        def cursor(self) -> Any:
            raise AssertionError("not exercised here")

        def close(self) -> None:
            self.closed = True

    def fake_connect(**_kwargs: Any) -> FakeConnection:
        connection = FakeConnection()
        opened.append(connection)
        return connection

    monkeypatch.setattr("gsf.connectors.databricks.sql.connect", fake_connect)
    db = DatabricksDatabase(_connection_string())

    with db.reuse_connection():
        with db._connect() as first:
            pass
        with db._connect() as second:
            pass
        assert first is second
        assert not first.closed  # statements must not close the shared connection

    assert len(opened) == 1
    assert opened[0].closed  # ...and the block closes it on exit


def test_reuse_connection_nests_without_reopening(monkeypatch: MonkeyPatch) -> None:
    opened = []

    class FakeConnection:
        def close(self) -> None:
            return None

    monkeypatch.setattr(
        "gsf.connectors.databricks.sql.connect",
        lambda **_kwargs: opened.append(FakeConnection()) or opened[-1],
    )
    db = DatabricksDatabase(_connection_string())

    with db.reuse_connection():
        with db.reuse_connection():
            pass
        # The inner block must not have closed the outer block's connection.
        with db._connect() as connection:
            assert connection is opened[0]

    assert len(opened) == 1


def test_a_statement_timeout_still_gets_its_own_session(
    monkeypatch: MonkeyPatch,
) -> None:
    """The cap is fixed per session, so a capped statement cannot share the pool."""
    kwargs_seen: list[dict[str, Any]] = []

    class FakeConnection:
        def close(self) -> None:
            return None

    def fake_connect(**kwargs: Any) -> FakeConnection:
        kwargs_seen.append(kwargs)
        return FakeConnection()

    monkeypatch.setattr("gsf.connectors.databricks.sql.connect", fake_connect)
    db = DatabricksDatabase(_connection_string())

    with db.reuse_connection():
        with db._connect(timeout_s=30):
            pass

    # One for the shared connection, one dedicated to the capped statement.
    assert len(kwargs_seen) == 2
    assert kwargs_seen[1]["session_configuration"] == {"statement_timeout": 30}
