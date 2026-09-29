# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ClickHouse connector tests.

Everything here is server-free: the connector's HTTP client is replaced with an
``httpx.MockTransport`` that records the statement and replays a canned
JSONCompact body, so no ClickHouse has to be reachable.
"""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx
import pandas as pd
import pytest

from auto_ontology.catalog.constants import TableTypes
from auto_ontology.connectors.clickhouse import (
    ClickHouseDatabase,
    ClickHouseError,
    _is_datetime,
    _is_nullable,
    _parse_connection_string,
    _quoted_literal,
)
from auto_ontology.connectors.connection_string_factory import build_connection_string
from auto_ontology.connectors.registry import CONNECTOR_REGISTRY, _redact

HOST = "ch.example.com"


def _connection(**overrides: object) -> dict[str, object]:
    connection: dict[str, object] = {
        "type": "clickhouse",
        "host": HOST,
        "user": "default",
        "database": "analytics",
    }
    connection.update(overrides)
    return connection


# ----------------------------------------------------------------------
# Fake transport
# ----------------------------------------------------------------------


class _FakeServer:
    """Replays queued responses and records the statements it was sent."""

    def __init__(self) -> None:
        self.statements: list[str] = []
        self.params: list[dict[str, list[str]]] = []
        self.responses: list[httpx.Response] = []

    def queue_rows(self, meta: list[tuple[str, str]], data: list[list[Any]]) -> None:
        body = {
            "meta": [{"name": name, "type": kind} for name, kind in meta],
            "data": data,
        }
        self.responses.append(httpx.Response(200, text=json.dumps(body)))

    def queue_empty(self) -> None:
        """A statement that returns no result set at all, the way DDL does."""
        self.responses.append(httpx.Response(200, text=""))

    def queue_error(self, status: int = 400, text: str = "") -> None:
        self.responses.append(httpx.Response(status, text=text))

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.statements.append(request.content.decode("utf-8"))
        self.params.append(parse_qs(request.url.query.decode("utf-8")))
        if not self.responses:
            return httpx.Response(200, text=json.dumps({"meta": [], "data": []}))
        return self.responses.pop(0)


def _database(**overrides: object) -> tuple[ClickHouseDatabase, _FakeServer]:
    db = ClickHouseDatabase(build_connection_string(_connection(**overrides)))
    server = _FakeServer()
    # Installed directly rather than through ``_ensure_client``: that method
    # only builds a client when there is none, so seeding one is enough to make
    # every request go through the mock transport.
    db._client = httpx.Client(
        base_url="http://test", transport=httpx.MockTransport(server.handle)
    )
    return db, server


# ----------------------------------------------------------------------
# Connection strings
# ----------------------------------------------------------------------


def test_factory_builds_a_url_the_connector_parses() -> None:
    url = build_connection_string(_connection(password="s3cr3t", port="9000"))
    assert url == "clickhouse://default:s3cr3t@ch.example.com:9000/analytics"
    settings = _parse_connection_string(url)
    assert settings["username"] == "default"
    assert settings["password"] == "s3cr3t"
    assert settings["database"] == "analytics"
    assert settings["port"] == 9000


def test_factory_defaults_the_http_port() -> None:
    assert build_connection_string(_connection()).endswith(":8123/analytics")


def test_factory_omits_an_empty_password() -> None:
    """A bare colon would be an empty password, not an absent one."""
    assert build_connection_string(_connection(password="")) == (
        "clickhouse://default@ch.example.com:8123/analytics"
    )


def test_factory_omits_credentials_entirely_when_there_is_no_user() -> None:
    assert build_connection_string(_connection(user="")) == (
        "clickhouse://ch.example.com:8123/analytics"
    )


def test_factory_encodes_a_password_containing_url_punctuation() -> None:
    url = build_connection_string(_connection(password="p@ss/w:rd"))
    assert _parse_connection_string(url)["password"] == "p@ss/w:rd"


def test_factory_keeps_a_scheme_pasted_into_the_host() -> None:
    """A copied ClickHouse Cloud endpoint arrives as ``https://...``."""
    url = build_connection_string(
        _connection(host="https://abc.clickhouse.cloud", port="8443")
    )
    assert "https://abc" not in url.split("://", 1)[1].split("?")[0]
    assert parse_qs(urlparse(url).query)["http_scheme"] == ["https"]
    assert _parse_connection_string(url)["host"] == "abc.clickhouse.cloud"


@pytest.mark.parametrize(
    ("port", "expected"),
    [(8123, "http"), (9000, "http"), (443, "https"), (8443, "https")],
)
def test_transport_is_inferred_from_the_port(port: int, expected: str) -> None:
    url = f"clickhouse://default@{HOST}:{port}/analytics"
    assert _parse_connection_string(url)["http_scheme"] == expected


def test_a_password_alone_does_not_imply_tls() -> None:
    """Authenticated cleartext on 8123 is the common self-managed shape.

    Inferring HTTPS from a credential the way the Trino connector does would
    make every stock ClickHouse connection fail to handshake.
    """
    url = f"clickhouse://default:secret@{HOST}:8123/analytics"
    assert _parse_connection_string(url)["http_scheme"] == "http"


def test_explicit_http_scheme_overrides_the_port() -> None:
    url = f"clickhouse://default@{HOST}:8443/analytics?http_scheme=http"
    assert _parse_connection_string(url)["http_scheme"] == "http"


def test_rejects_an_unsupported_http_scheme() -> None:
    with pytest.raises(ValueError, match="http_scheme"):
        _parse_connection_string(f"clickhouse://{HOST}:8123/analytics?http_scheme=tcp")


def test_rejects_a_url_without_a_database() -> None:
    with pytest.raises(ValueError, match="requires a database"):
        _parse_connection_string(f"clickhouse://default@{HOST}:8123/")


def test_rejects_a_url_without_a_host() -> None:
    with pytest.raises(ValueError, match="missing host"):
        _parse_connection_string("clickhouse:///analytics")


def test_rejects_another_engines_url() -> None:
    with pytest.raises(ValueError, match="Not a ClickHouse URL"):
        _parse_connection_string("postgresql://user@host:5432/db")


@pytest.mark.parametrize("database", ["system", "information_schema"])
def test_rejects_binding_a_system_database(database: str) -> None:
    """Cataloguing the engine's own bookkeeping is never what was meant."""
    with pytest.raises(ValueError, match="bookkeeping"):
        _parse_connection_string(f"clickhouse://{HOST}:8123/{database}")


def test_registry_maps_the_clickhouse_scheme() -> None:
    assert CONNECTOR_REGISTRY["clickhouse"] is ClickHouseDatabase


def test_registry_redacts_the_password_before_logging() -> None:
    url = build_connection_string(_connection(password="s3cr3t"))
    assert "s3cr3t" not in _redact(url)


# ----------------------------------------------------------------------
# Identity
# ----------------------------------------------------------------------


def test_dialect_and_database_name() -> None:
    db, _ = _database()
    assert db.dialect == "clickhouse"
    assert db.database_name == "analytics"


def test_qualify_emits_a_two_part_name() -> None:
    """ClickHouse has no third level, so the base class default is correct."""
    db, _ = _database()
    assert db.qualify("analytics", "orders") == '"analytics"."orders"'


# ----------------------------------------------------------------------
# Execution
# ----------------------------------------------------------------------


def test_execute_decodes_a_jsoncompact_body() -> None:
    db, server = _database()
    server.queue_rows([("id", "UInt64"), ("name", "String")], [[1, "a"], [2, "b"]])

    frame = db.execute("SELECT id, name FROM orders")

    assert list(frame.columns) == ["id", "name"]
    assert frame["id"].tolist() == [1, 2]
    assert server.statements == ["SELECT id, name FROM orders"]


def test_execute_binds_the_database_and_response_format() -> None:
    db, server = _database()
    server.queue_rows([("x", "UInt8")], [[1]])

    db.execute("SELECT 1 AS x")

    params = server.params[0]
    assert params["database"] == ["analytics"]
    assert params["default_format"] == ["JSONCompact"]


def test_execute_returns_an_empty_frame_for_a_statement_with_no_result() -> None:
    db, server = _database()
    server.queue_empty()
    assert db.execute("CREATE TABLE t (a UInt8) ENGINE = Memory").empty


def test_execute_parses_datetime_columns() -> None:
    """JSONCompact renders every date as a string; profiling needs timestamps."""
    db, server = _database()
    server.queue_rows(
        [("seen", "DateTime"), ("day", "Nullable(Date)")],
        [["2026-01-02 03:04:05", "2026-01-02"]],
    )

    frame = db.execute("SELECT seen, day FROM events")

    assert pd.api.types.is_datetime64_any_dtype(frame["seen"])
    assert pd.api.types.is_datetime64_any_dtype(frame["day"])


def test_execute_passes_a_statement_timeout_to_the_server() -> None:
    db, server = _database()
    server.queue_rows([("x", "UInt8")], [[1]])

    db.execute("SELECT 1 AS x", timeout_s=5)

    assert server.params[0]["max_execution_time"] == ["5"]


def test_supports_statement_timeout_is_advertised() -> None:
    """Profiling only passes a cap to connectors that claim to honour one."""
    assert ClickHouseDatabase.supports_statement_timeout is True


def test_execute_rejects_positional_parameters() -> None:
    """Silently inlining them would be a SQL-injection shaped guess at types."""
    db, _ = _database()
    with pytest.raises(ValueError, match="positional parameter"):
        db.execute("SELECT * FROM orders WHERE id = ?", [1])


def test_execute_surfaces_the_servers_own_diagnosis() -> None:
    db, server = _database()
    server.queue_error(404, "Code: 60. DB::Exception: Table analytics.nope not found.")

    with pytest.raises(ClickHouseError, match="Table analytics.nope not found"):
        db.execute("SELECT 1 FROM nope")


def test_execute_reports_an_unreachable_server_as_a_connection_error() -> None:
    """``db_errors`` classifies this by message, so the wording matters."""
    from auto_ontology.connectors.db_errors import is_infrastructure_error

    db, _ = _database()

    def _refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("[Errno 61] Connection refused")

    db._client = httpx.Client(
        base_url="http://test", transport=httpx.MockTransport(_refuse)
    )

    with pytest.raises(ClickHouseError) as caught:
        db.execute("SELECT 1")
    assert is_infrastructure_error(caught.value)


def test_execute_rejects_a_non_json_body() -> None:
    db, server = _database()
    server.responses.append(httpx.Response(200, text="<html>proxy error</html>"))

    with pytest.raises(ClickHouseError, match="non-JSON"):
        db.execute("SELECT 1")


# ----------------------------------------------------------------------
# Introspection
# ----------------------------------------------------------------------


def test_get_tables_classifies_views_by_engine() -> None:
    db, server = _database()
    server.queue_rows(
        [
            ("table_schema", "String"),
            ("table_name", "String"),
            ("table_type", "String"),
        ],
        [
            ["analytics", "orders", TableTypes.BASE_TABLE],
            ["analytics", "order_summary", TableTypes.MATERIALIZED_VIEW],
        ],
    )

    frame = db.get_tables()

    assert list(frame.columns) == ["table_schema", "table_name", "table_type"]
    statement = server.statements[0]
    assert "system.tables" in statement
    assert "'MaterializedView'" in statement
    assert "NOT is_temporary" in statement
    assert "database = 'analytics'" in statement


def test_get_tables_returns_a_shaped_frame_when_the_database_is_empty() -> None:
    db, server = _database()
    server.queue_rows([], [])
    assert list(db.get_tables().columns) == ["table_schema", "table_name", "table_type"]


def test_get_columns_derives_nullability_from_the_type() -> None:
    db, server = _database()
    server.queue_rows(
        [
            ("table_schema", "String"),
            ("table_name", "String"),
            ("column_name", "String"),
            ("data_type", "String"),
            ("ordinal_position", "UInt64"),
        ],
        [
            ["analytics", "orders", "id", "UInt64", 1],
            ["analytics", "orders", "note", "Nullable(String)", 2],
            ["analytics", "orders", "tag", "LowCardinality(Nullable(String))", 3],
            ["analytics", "orders", "tags", "Array(Nullable(String))", 4],
        ],
    )

    frame = db.get_columns()

    assert list(frame.columns) == [
        "table_schema",
        "table_name",
        "column_name",
        "data_type",
        "is_nullable",
        "ordinal_position",
    ]
    assert dict(zip(frame.column_name, frame.is_nullable)) == {
        "id": False,
        "note": True,
        "tag": True,
        # An array that may hold nulls is not itself a nullable column.
        "tags": False,
    }


def test_get_columns_nullability_is_a_bool_dtype() -> None:
    """An object column of Python bools would still be a latent string column."""
    db, server = _database()
    server.queue_rows(
        [
            ("table_schema", "String"),
            ("table_name", "String"),
            ("column_name", "String"),
            ("data_type", "String"),
            ("ordinal_position", "UInt64"),
        ],
        [["analytics", "orders", "id", "UInt64", 1]],
    )
    assert db.get_columns().is_nullable.dtype == bool


@pytest.mark.parametrize(
    ("data_type", "expected"),
    [
        ("UInt64", False),
        ("Nullable(String)", True),
        ("LowCardinality(Nullable(String))", True),
        ("LowCardinality(String)", False),
        ("Array(Nullable(Int32))", False),
        ("Map(String, Nullable(String))", False),
    ],
)
def test_is_nullable_reads_the_outermost_wrapper(
    data_type: str, expected: bool
) -> None:
    assert _is_nullable(data_type) is expected


def test_get_views_prefers_as_select_and_falls_back() -> None:
    """Servers before 23.x have no ``as_select``; the whole query is retried."""
    db, server = _database()
    server.queue_error(400, "Code: 47. DB::Exception: Missing columns: 'as_select'")
    server.queue_rows(
        [
            ("table_schema", "String"),
            ("table_name", "String"),
            ("view_definition", "String"),
        ],
        [["analytics", "v", "CREATE VIEW analytics.v AS SELECT 1"]],
    )

    frame = db.get_views()

    assert frame["view_definition"].tolist() == ["CREATE VIEW analytics.v AS SELECT 1"]
    assert "as_select AS view_definition" in server.statements[0]
    assert "create_table_query AS view_definition" in server.statements[1]


def test_get_views_is_empty_when_the_database_has_none() -> None:
    db, server = _database()
    server.queue_rows([], [])
    assert list(db.get_views().columns) == [
        "table_schema",
        "table_name",
        "view_definition",
    ]


def test_get_pks_numbers_columns_within_the_key() -> None:
    db, server = _database()
    server.queue_rows(
        [
            ("table_schema", "String"),
            ("table_name", "String"),
            ("column_name", "String"),
            ("ordinal_position", "UInt64"),
        ],
        [
            ["analytics", "orders", "day", 1],
            ["analytics", "orders", "id", 2],
        ],
    )

    frame = db.get_pks()

    assert frame["ordinal_position"].tolist() == [1, 2]
    assert "is_in_primary_key" in server.statements[0]


def test_get_fks_is_always_empty() -> None:
    """ClickHouse declares no foreign keys."""
    db, server = _database()
    assert db.get_fks().empty
    assert server.statements == []


def test_get_queries_filters_to_finished_queries_touching_this_database() -> None:
    db, server = _database()
    server.queue_rows(
        [("end_time", "DateTime"), ("query_text", "String")],
        [["2026-01-02 03:04:05", "SELECT 1"]],
    )

    frame = db.get_queries(hours=6)

    assert list(frame.columns) == ["end_time", "query_text"]
    statement = server.statements[0]
    assert "system.query_log" in statement
    assert "type = 'QueryFinish'" in statement
    assert "INTERVAL 6 HOUR" in statement
    assert "has(databases, 'analytics')" in statement


def test_get_queries_is_empty_when_the_log_is_unreadable() -> None:
    """``system.query_log`` is optional and often not grantable."""
    db, server = _database()
    server.queue_error(403, "Code: 497. DB::Exception: Not enough privileges")

    assert list(db.get_queries().columns) == ["end_time", "query_text"]


# ----------------------------------------------------------------------
# Lifecycle
# ----------------------------------------------------------------------


def test_ping_sends_the_bound_database_with_a_trivial_query() -> None:
    """The server resolves the ``database`` parameter before running anything."""
    db, server = _database()
    server.queue_rows([("1", "UInt8")], [[1]])

    db.ping()

    assert server.statements == ["SELECT 1"]
    assert server.params[0]["database"] == ["analytics"]


def test_ping_fails_when_the_server_does_not_know_the_database() -> None:
    db, server = _database()
    server.queue_error(404, "Code: 81. DB::Exception: Database nope does not exist.")

    with pytest.raises(ClickHouseError, match="does not exist"):
        db.ping()


def test_ping_passes_for_a_database_the_user_cannot_see_listed() -> None:
    """``system.databases`` is grant-filtered; a lookup there is a false negative.

    Observed against a real cluster: a read-only user querying ``perfbot``
    gets a working connection, but ``perfbot`` is absent from both
    ``system.databases`` and ``SHOW DATABASES``.
    """
    db, server = _database(database="perfbot")
    server.queue_rows([("1", "UInt8")], [[1]])

    db.ping()

    assert "system.databases" not in server.statements[0]


def test_close_releases_the_client_and_is_idempotent() -> None:
    db, _ = _database()
    db.close()
    assert db._client is None
    db.close()


# ----------------------------------------------------------------------
# Quoting
# ----------------------------------------------------------------------


def test_quoted_literal_escapes_backslashes_before_quotes() -> None:
    """Backslash is an escape inside a ClickHouse literal, unlike ANSI SQL."""
    assert _quoted_literal("a'b") == "'a\\'b'"
    assert _quoted_literal("a\\b") == "'a\\\\b'"


# ----------------------------------------------------------------------
# Regressions
# ----------------------------------------------------------------------


def test_pasted_cloud_endpoint_does_not_duplicate_the_port() -> None:
    """A copied endpoint carries its port; the Port field is optional.

    Emitting both produced ``host:8443:8123``, which is not merely the wrong
    port -- ``urlparse(...).port`` raises on it, so the connector crashed before
    it could report anything useful.
    """
    url = build_connection_string(
        _connection(host="https://abc.clickhouse.cloud:8443", port="")
    )
    assert urlparse(url).port == 8443
    settings = _parse_connection_string(url)
    assert settings["host"] == "abc.clickhouse.cloud"
    assert settings["port"] == 8443
    assert settings["http_scheme"] == "https"


def test_explicit_port_beats_the_one_pasted_into_the_host() -> None:
    url = build_connection_string(
        _connection(host="https://abc.clickhouse.cloud:8443", port="9440")
    )
    assert _parse_connection_string(url)["port"] == 9440


def test_a_password_survives_a_blank_user() -> None:
    """The form marks User optional but still offers Password.

    Dropping the password when no user is given downgrades the connection to
    unauthenticated, which surfaces as an auth failure the form cannot explain.
    """
    url = build_connection_string(_connection(user="", password="s3cr3t"))
    assert _parse_connection_string(url)["password"] == "s3cr3t"


def test_blank_user_with_a_password_authenticates_as_the_default_user() -> None:
    """Built without the mock client: the assertion is about the real one."""
    db = ClickHouseDatabase(
        build_connection_string(_connection(user="", password="s3cr3t"))
    )
    try:
        assert isinstance(db._ensure_client().auth, httpx.BasicAuth)
    finally:
        db.close()


@pytest.mark.parametrize(
    "error",
    [
        httpx.ConnectError("All connection attempts failed"),
        httpx.ConnectTimeout(""),
        httpx.ReadTimeout(""),
    ],
)
def test_transport_failures_classify_as_infrastructure(error: Exception) -> None:
    """``db_errors`` matches on message text, and these stringify to nothing.

    Misclassified, an unreachable server reads as bad SQL and the text-to-SQL
    graph burns retries rewriting a query that was never the problem.
    """
    from auto_ontology.connectors.db_errors import is_infrastructure_error

    db, _ = _database()
    db._client = httpx.Client(
        base_url="http://test",
        transport=httpx.MockTransport(lambda request: (_ for _ in ()).throw(error)),
    )

    with pytest.raises(ClickHouseError) as caught:
        db.execute("SELECT 1")
    assert is_infrastructure_error(caught.value)


def test_get_views_does_not_swallow_a_connectivity_failure() -> None:
    """Only an absent column justifies the fallback.

    Retrying on any error means a server that goes away mid-pass fails both
    attempts and reports "no views" for a database full of them.
    """
    db, server = _database()
    server.queue_error(502, "Bad Gateway")
    server.queue_error(502, "Bad Gateway")

    with pytest.raises(ClickHouseError):
        db.get_views()


def test_get_views_still_falls_back_on_an_absent_column() -> None:
    db, server = _database()
    server.queue_error(400, "Code: 47. DB::Exception: Missing columns: 'as_select'")
    server.queue_rows(
        [
            ("table_schema", "String"),
            ("table_name", "String"),
            ("view_definition", "String"),
        ],
        [["analytics", "v", "CREATE VIEW analytics.v AS SELECT 1"]],
    )

    assert len(db.get_views()) == 1


def test_get_pks_orders_by_the_key_not_the_table() -> None:
    """``system.columns`` carries no key ordinal.

    Ordering its rows by ``position`` reports ``ORDER BY (day, id)`` on a table
    declared ``(id, day)`` as ``(id, day)`` -- reversed, and reversed silently.
    The order lives in ``system.tables.primary_key``.
    """
    db, server = _database()
    server.queue_rows([], [])

    db.get_pks()

    statement = server.statements[0]
    assert "system.tables" in statement
    assert "primary_key" in statement
    assert "splitByChar" in statement
    # The real columns are joined back in so an expression key contributes only
    # its columns, rather than comma-torn fragments of a function call.
    assert "is_in_primary_key" in statement
    assert "ORDER BY keys.key_position" in " ".join(statement.split())


def test_introspection_is_not_row_capped() -> None:
    """A wide enough schema would otherwise truncate ``get_columns`` silently."""
    db, server = _database()
    server.queue_rows([], [])
    db.get_columns()
    assert "max_result_rows" not in server.params[0]


def test_user_queries_are_still_row_capped() -> None:
    db, server = _database()
    server.queue_rows([("x", "UInt8")], [[1]])
    db.execute("SELECT 1 AS x")
    assert server.params[0]["max_result_rows"] == ["50000"]


@pytest.mark.parametrize(
    ("data_type", "expected"),
    [
        ("DateTime", True),
        ("Date32", True),
        ("Nullable(Date)", True),
        ("LowCardinality(Nullable(DateTime))", True),
        ("DateTime64(3, 'UTC')", True),
        # A substring test read this as a date and coerced every value to NaT,
        # leaving an all-null timestamp column rather than failing loudly.
        ("Enum8('Date' = 1, 'Time' = 2)", False),
        ("Array(DateTime)", False),
        ("String", False),
    ],
)
def test_is_datetime_anchors_on_the_unwrapped_type(
    data_type: str, expected: bool
) -> None:
    assert _is_datetime(data_type) is expected


def test_an_enum_naming_date_is_not_coerced_to_null() -> None:
    db, server = _database()
    server.queue_rows([("kind", "Enum8('Date' = 1, 'Time' = 2)")], [["Date"], ["Time"]])

    frame = db.execute("SELECT kind FROM t")

    assert frame["kind"].tolist() == ["Date", "Time"]
