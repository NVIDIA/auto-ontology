# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Apache Pinot connector tests.

Server-free: the connector's HTTP client is replaced with an
``httpx.MockTransport`` that routes broker and controller requests to canned
JSON, so no Pinot cluster has to be reachable.
"""

from __future__ import annotations

import base64
import json
from typing import Any, Callable

import httpx
import pandas as pd
import pytest

from auto_ontology.catalog.constants import TableTypes
from auto_ontology.connectors.base import StatementTimeout
from auto_ontology.connectors.connection_string_factory import build_connection_string
from auto_ontology.connectors.db_errors import is_infrastructure_error
from auto_ontology.connectors.pinot import (
    PinotDatabase,
    PinotError,
    _parse_connection_string,
)
from auto_ontology.connectors.registry import CONNECTOR_REGISTRY, _redact

BROKER = "http://broker.example.com:8099"
CONTROLLER = "http://broker.example.com:9000"


def _connection(**overrides: object) -> dict[str, object]:
    connection: dict[str, object] = {
        "type": "pinot",
        "host": "broker.example.com",
        "database": "default",
    }
    connection.update(overrides)
    return connection


# ----------------------------------------------------------------------
# Fake cluster
# ----------------------------------------------------------------------


def _result(columns: list[tuple[str, str]], rows: list[list[Any]]) -> dict[str, Any]:
    return {
        "resultTable": {
            "dataSchema": {
                "columnNames": [name for name, _ in columns],
                "columnDataTypes": [kind for _, kind in columns],
            },
            "rows": rows,
        },
        "exceptions": [],
    }


class _FakeCluster:
    """A broker and a controller behind one mock transport."""

    def __init__(self) -> None:
        self.queries: list[dict[str, Any]] = []
        self.requests: list[httpx.Request] = []
        self.broker_responses: list[httpx.Response] = []
        self.tables: list[str] = []
        self.schemas: dict[str, dict[str, Any]] = {}
        self.controller_status = 200

    def queue(self, body: dict[str, Any], status: int = 200) -> None:
        self.broker_responses.append(httpx.Response(status, json=body))

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        url = str(request.url)
        if url == f"{BROKER}/health":
            return httpx.Response(200, text="OK")
        if url.startswith(f"{BROKER}/query/sql"):
            self.queries.append(json.loads(request.content))
            if self.broker_responses:
                return self.broker_responses.pop(0)
            return httpx.Response(200, json=_result([("1", "INT")], [[1]]))
        if url.startswith(CONTROLLER):
            if self.controller_status != 200:
                return httpx.Response(self.controller_status, text="boom")
            path = request.url.path
            if path == "/tables":
                return httpx.Response(200, json={"tables": self.tables})
            if path.startswith("/tables/") and path.endswith("/schema"):
                table = path[len("/tables/") : -len("/schema")]
                if table in self.schemas:
                    return httpx.Response(200, json=self.schemas[table])
                return httpx.Response(404, json={"error": "not found"})
        return httpx.Response(599, text=f"unexpected request {url}")


def _database(
    url: str | None = None, **overrides: object
) -> tuple[PinotDatabase, _FakeCluster]:
    db = PinotDatabase(url or build_connection_string(_connection(**overrides)))
    cluster = _FakeCluster()
    # Seeded directly: ``_ensure_client`` only builds a client when there is
    # none. Headers and auth are copied so the tests see what would be sent.
    real = db._ensure_client()
    db._client = httpx.Client(
        headers=real.headers,
        auth=real.auth,
        transport=httpx.MockTransport(cluster.handle),
    )
    real.close()
    return db, cluster


# ----------------------------------------------------------------------
# Connection strings
# ----------------------------------------------------------------------


def test_registered_under_its_scheme() -> None:
    assert CONNECTOR_REGISTRY["pinot"] is PinotDatabase


def test_factory_defaults_the_broker_port() -> None:
    assert build_connection_string(_connection()) == (
        "pinot://broker.example.com:8099/default"
    )


def test_factory_sends_user_and_password_as_basic_auth() -> None:
    url = build_connection_string(_connection(user="admin", password="s3/cr@t"))
    settings = _parse_connection_string(url)
    assert settings["username"] == "admin"
    assert settings["password"] == "s3/cr@t"
    assert settings["token"] is None


def test_factory_sends_a_password_without_a_user_as_a_token() -> None:
    url = build_connection_string(_connection(password="tok&en"))
    assert "@" not in url
    settings = _parse_connection_string(url)
    assert settings["token"] == "tok&en"
    assert settings["username"] is None


def test_factory_keeps_a_pasted_scheme_and_the_controller() -> None:
    url = build_connection_string(
        _connection(
            host="https://broker.example.com",
            port="443",
            controller="https://controller.example.com",
        )
    )
    settings = _parse_connection_string(url)
    assert settings["broker_url"] == "https://broker.example.com:443"
    # A full URL without a port means its scheme's port, as behind an ingress.
    assert settings["controller_url"] == "https://controller.example.com:443"


def test_factory_takes_a_port_pasted_into_the_host() -> None:
    url = build_connection_string(_connection(host="https://broker.example.com:443"))
    assert url == "pinot://broker.example.com:443/default?http_scheme=https"
    assert _parse_connection_string(url)["broker_url"] == (
        "https://broker.example.com:443"
    )


def test_port_field_wins_over_a_pasted_port() -> None:
    url = build_connection_string(_connection(host="broker:1234", port="8099"))
    assert url == "pinot://broker:8099/default"


def test_raw_plus_in_a_token_is_kept() -> None:
    """``parse_qs`` would read a base64 ``+`` as a space."""
    settings = _parse_connection_string("pinot://broker/default?token=ab+c%2Fd%25e")
    assert settings["token"] == "ab+c/d%e"


def test_factory_token_round_trips_exactly() -> None:
    token = "a+b/c=%2B"
    url = build_connection_string(_connection(password=token))
    assert _parse_connection_string(url)["token"] == token


def test_controller_defaults_to_the_broker_host() -> None:
    settings = _parse_connection_string("pinot://broker:8099/default")
    assert settings["controller_url"] == "http://broker:9000"


def test_bare_controller_host_takes_the_broker_scheme() -> None:
    settings = _parse_connection_string(
        "pinot://broker:8443/default?controller=ctl:9443"
    )
    assert settings["controller_url"] == "https://ctl:9443"


def test_database_defaults_to_default() -> None:
    assert _parse_connection_string("pinot://broker")["database"] == "default"


@pytest.mark.parametrize(
    "url",
    [
        "pinot:///default",
        "mysql://broker/default",
        "pinot://broker/default?http_scheme=ftp",
        "pinot://u:p@broker/default?token=t",
        "pinot://broker/default?controller=http://ctl:9000/api",
    ],
)
def test_rejects_invalid_urls(url: str) -> None:
    with pytest.raises(ValueError):
        _parse_connection_string(url)


def test_parser_errors_never_echo_the_url() -> None:
    """The registry logs exception text verbatim, so it must hold no secret."""
    for url in ("pinot:///default?token=SECRET", "pinotx://u:SECRET@broker/default"):
        with pytest.raises(ValueError) as error:
            _parse_connection_string(url)
        assert "SECRET" not in str(error.value)


def test_encoded_parameter_names_are_not_accepted() -> None:
    """``%74oken`` would dodge the log redaction that looks for ``token=``."""
    settings = _parse_connection_string("pinot://broker/default?%74oken=SECRET")
    assert settings["token"] is None


def test_ipv6_hosts_keep_their_brackets() -> None:
    settings = _parse_connection_string("pinot://[::1]:8099/default")
    assert settings["broker_url"] == "http://[::1]:8099"
    assert settings["controller_url"] == "http://[::1]:9000"
    httpx.URL(settings["broker_url"])  # parses


@pytest.mark.parametrize("name", ["token", "%74oken", "TOKEN", "%54oken"])
def test_redact_masks_encoded_token_names(name: str) -> None:
    redacted = _redact(f"pinot://broker/default?{name}=SECRET&http_scheme=ftp")
    assert "SECRET" not in redacted
    assert "http_scheme=ftp" in redacted


def test_failed_connector_log_never_holds_the_token(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A bad URL is logged in full by the registry; the token must not be."""
    from auto_ontology.connectors.registry import create_connector

    with pytest.raises(ValueError):
        create_connector("pinot://broker/default?%74oken=SECRET&http_scheme=ftp")
    assert caplog.records
    assert "SECRET" not in caplog.text


def test_redact_masks_the_token() -> None:
    redacted = _redact("pinot://broker:8099/default?token=SECRET&http_scheme=https")
    assert "SECRET" not in redacted
    assert "http_scheme=https" in redacted


# ----------------------------------------------------------------------
# Execution
# ----------------------------------------------------------------------


def test_execute_decodes_the_result_table() -> None:
    db, cluster = _database()
    cluster.queue(
        _result(
            [("city", "STRING"), ("n", "LONG"), ("ts", "TIMESTAMP")],
            [["SF", 3, "2026-01-02 03:04:05.0"], ["NY", 1, "2026-01-03 00:00:00.0"]],
        )
    )
    frame = db.execute('SELECT city, n, ts FROM "trips"')
    assert list(frame.columns) == ["city", "n", "ts"]
    assert frame["n"].tolist() == [3, 1]
    assert pd.api.types.is_datetime64_any_dtype(frame["ts"])
    assert cluster.queries[-1]["sql"] == 'SELECT city, n, ts FROM "trips"'


def test_multistage_engine_is_on_by_default() -> None:
    db, cluster = _database()
    db.execute("SELECT 1")
    assert cluster.queries[-1]["queryOptions"] == "useMultistageEngine=true"


def test_multistage_can_be_turned_off() -> None:
    db, cluster = _database("pinot://broker.example.com:8099/default?multistage=false")
    db.execute("SELECT 1")
    # Explicit, since a broker may be configured to default to multi-stage.
    assert cluster.queries[-1]["queryOptions"] == "useMultistageEngine=false"


def test_timeout_travels_as_a_query_option() -> None:
    db, cluster = _database()
    db.execute("SELECT 1", timeout_s=2.5)
    assert cluster.queries[-1]["queryOptions"] == (
        "useMultistageEngine=true;timeoutMs=2500"
    )


def test_empty_result_is_an_empty_frame() -> None:
    db, cluster = _database()
    cluster.queue({"exceptions": [], "numDocsScanned": 0})
    assert db.execute("SELECT 1").empty


def test_parameters_are_refused() -> None:
    db, _ = _database()
    with pytest.raises(ValueError):
        db.execute("SELECT ?", [1])


def test_exceptions_alongside_rows_fail_the_statement() -> None:
    """A broker that lost a server still answers 200 with the rows it has."""
    db, cluster = _database()
    body = _result([("n", "LONG")], [[1]])
    body["exceptions"] = [{"errorCode": 200, "message": "QueryExecutionError:\nat x"}]
    cluster.queue(body)
    with pytest.raises(PinotError, match=r"\[200\] QueryExecutionError:$"):
        db.execute("SELECT n FROM t")


def test_partial_result_fails_the_statement() -> None:
    db, cluster = _database()
    body = _result([("n", "LONG")], [[1]])
    body["partialResult"] = True
    cluster.queue(body)
    with pytest.raises(PinotError, match="partial result"):
        db.execute("SELECT n FROM a JOIN b ON a.id = b.id")


@pytest.mark.parametrize("code", [240, 250, 400])
def test_timeout_codes_are_statement_timeouts(code: int) -> None:
    db, cluster = _database()
    cluster.queue(
        {"exceptions": [{"errorCode": code, "message": "timed out"}]}, status=408
    )
    with pytest.raises(StatementTimeout):
        db.execute("SELECT 1", timeout_s=5)


def test_timeout_code_without_a_cap_is_an_ordinary_error() -> None:
    db, cluster = _database()
    cluster.queue({"exceptions": [{"errorCode": 250, "message": "timed out"}]})
    with pytest.raises(PinotError):
        db.execute("SELECT 1")


def test_client_read_timeout_with_a_cap_is_a_statement_timeout() -> None:
    db, _ = _database()

    def stall(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("stalled", request=request)

    db._client = httpx.Client(transport=httpx.MockTransport(stall))
    with pytest.raises(StatementTimeout):
        db.execute("SELECT 1", timeout_s=1)


def test_uncapped_read_timeout_is_a_timeout_not_an_outage() -> None:
    """A slow query must not be reported as an unreachable cluster."""
    db, _ = _database()

    def stall(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("stalled", request=request)

    db._client = httpx.Client(transport=httpx.MockTransport(stall))
    with pytest.raises(StatementTimeout) as error:
        db.execute("SELECT 1")
    assert not is_infrastructure_error(str(error.value), "SELECT 1")


def test_single_stage_warns_on_a_possible_implicit_limit(
    caplog: pytest.LogCaptureFixture,
) -> None:
    db, cluster = _database("pinot://broker.example.com:8099/default?multistage=false")
    for sql in ("SELECT city FROM trips", "SELECT city FROM trips LIMIT 10"):
        cluster.queue(_result([("city", "STRING")], [[str(i)] for i in range(10)]))
        db.execute(sql)
    warnings = [r for r in caplog.records if "implicit LIMIT" in r.getMessage()]
    assert len(warnings) == 1
    assert warnings[0].getMessage().endswith("Statement: SELECT city FROM trips")


def test_unreachable_broker_reads_as_infrastructure() -> None:
    db, _ = _database()

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("All connection attempts failed", request=request)

    db._client = httpx.Client(transport=httpx.MockTransport(refuse))
    with pytest.raises(PinotError) as error:
        db.execute("SELECT 1")
    assert is_infrastructure_error(error.value, "SELECT 1")


def test_unavailable_servers_read_as_infrastructure() -> None:
    db, cluster = _database()
    cluster.queue({"exceptions": [{"errorCode": 427, "message": "1 servers"}]})
    with pytest.raises(PinotError) as error:
        db.execute("SELECT 1")
    assert is_infrastructure_error(error.value, "SELECT 1")


def test_sql_errors_do_not_read_as_infrastructure() -> None:
    db, cluster = _database()
    cluster.queue(
        {"exceptions": [{"errorCode": 710, "message": "Unknown column: nope"}]},
        status=400,
    )
    with pytest.raises(PinotError) as error:
        db.execute("SELECT nope FROM t")
    assert not is_infrastructure_error(error.value, "SELECT nope FROM t")


def test_non_json_error_response_is_reported() -> None:
    db, cluster = _database()
    cluster.broker_responses.append(httpx.Response(502, text="Bad Gateway"))
    with pytest.raises(PinotError, match="HTTP 502"):
        db.execute("SELECT 1")


# ----------------------------------------------------------------------
# Authentication and database scoping
# ----------------------------------------------------------------------


def _sent(cluster: _FakeCluster, header: str) -> set[str | None]:
    return {request.headers.get(header) for request in cluster.requests}


def test_basic_auth_reaches_broker_and_controller() -> None:
    db, cluster = _database(user="admin", password="pw")
    db.ping()
    expected = "Basic " + base64.b64encode(b"admin:pw").decode()
    assert _sent(cluster, "authorization") == {expected}
    assert {r.url.port for r in cluster.requests} == {8099, 9000}


def test_token_is_sent_as_bearer() -> None:
    db, cluster = _database(password="TOKEN")
    db.ping()
    assert _sent(cluster, "authorization") == {"Bearer TOKEN"}


def test_default_database_sends_no_header() -> None:
    db, cluster = _database()
    db.ping()
    assert _sent(cluster, "database") == {None}


def test_named_database_is_sent_as_a_header() -> None:
    db, cluster = _database(database="analytics")
    db.ping()
    assert _sent(cluster, "database") == {"analytics"}


def test_single_stage_ping_counts_a_table_instead_of_select_1() -> None:
    """Pinot 1.4's single-stage engine rejects a table-free ``SELECT 1``."""
    db, cluster = _database("pinot://broker.example.com:8099/default?multistage=false")
    cluster.tables = ["trips"]
    db.ping()
    assert cluster.queries == [
        {
            "sql": 'SELECT COUNT(*) FROM "trips"',
            "queryOptions": "useMultistageEngine=false",
        }
    ]


def test_single_stage_ping_without_tables_reads_broker_health() -> None:
    db, cluster = _database("pinot://broker.example.com:8099/default?multistage=false")
    db.ping()
    assert cluster.queries == []
    assert cluster.requests[-1].url.path == "/health"


def test_ping_fails_when_the_controller_is_down() -> None:
    db, cluster = _database()
    cluster.controller_status = 503
    with pytest.raises(PinotError, match="controller"):
        db.ping()


# ----------------------------------------------------------------------
# Introspection
# ----------------------------------------------------------------------


def _trips_schema(null_handling: bool = False) -> dict[str, Any]:
    return {
        "schemaName": "trips",
        "enableColumnBasedNullHandling": null_handling,
        "primaryKeyColumns": ["trip_id", "city"],
        "dimensionFieldSpecs": [
            {"name": "trip_id", "dataType": "STRING", "notNull": True},
            {"name": "city", "dataType": "STRING"},
            {"name": "tags", "dataType": "STRING", "singleValueField": False},
        ],
        "metricFieldSpecs": [{"name": "fare", "dataType": "DOUBLE"}],
        "dateTimeFieldSpecs": [
            {
                "name": "ts",
                "dataType": "TIMESTAMP",
                "format": "1:MILLISECONDS:TIMESTAMP",
                "granularity": "1:MILLISECONDS",
            }
        ],
    }


def test_get_tables_lists_logical_tables() -> None:
    db, cluster = _database()
    cluster.tables = ["trips", "airports"]
    cluster.schemas = {"trips": _trips_schema(), "airports": _trips_schema()}
    frame = db.get_tables()
    assert frame.to_dict("records") == [
        {
            "table_schema": "default",
            "table_name": "airports",
            "table_type": str(TableTypes.BASE_TABLE),
        },
        {
            "table_schema": "default",
            "table_name": "trips",
            "table_type": str(TableTypes.BASE_TABLE),
        },
    ]


def test_named_database_prefix_is_stripped() -> None:
    db, cluster = _database(database="analytics")
    cluster.tables = ["analytics.trips"]
    cluster.schemas["trips"] = _trips_schema()
    assert db.get_tables()["table_name"].tolist() == ["trips"]


def test_get_columns_reads_every_field_spec_in_order() -> None:
    db, cluster = _database()
    cluster.tables = ["trips"]
    cluster.schemas["trips"] = _trips_schema()
    frame = db.get_columns()
    assert frame["column_name"].tolist() == ["trip_id", "city", "tags", "fare", "ts"]
    assert frame["data_type"].tolist() == [
        "STRING",
        "STRING",
        "STRING_ARRAY",
        "DOUBLE",
        "TIMESTAMP",
    ]
    assert frame["ordinal_position"].tolist() == [1, 2, 3, 4, 5]
    assert set(frame["table_schema"]) == {"default"}


def test_legacy_time_field_spec_is_a_column() -> None:
    db, cluster = _database()
    cluster.tables = ["legacy"]
    cluster.schemas["legacy"] = {
        "schemaName": "legacy",
        "dimensionFieldSpecs": [{"name": "city", "dataType": "STRING"}],
        "metricFieldSpecs": [{"name": "fare", "dataType": "DOUBLE"}],
        "timeFieldSpec": {
            "incomingGranularitySpec": {
                "name": "epoch_s",
                "dataType": "LONG",
                "timeType": "SECONDS",
            },
            "outgoingGranularitySpec": {
                "name": "days",
                "dataType": "INT",
                "timeType": "DAYS",
            },
        },
        "dateTimeFieldSpecs": [{"name": "ts", "dataType": "TIMESTAMP"}],
    }
    frame = db.get_columns()
    assert frame["column_name"].tolist() == ["city", "fare", "days", "ts"]
    assert frame["data_type"].tolist() == ["STRING", "DOUBLE", "INT", "TIMESTAMP"]
    assert frame["ordinal_position"].tolist() == [1, 2, 3, 4]


def test_legacy_time_field_falls_back_to_incoming_spec() -> None:
    db, cluster = _database()
    cluster.tables = ["legacy"]
    cluster.schemas["legacy"] = {
        "schemaName": "legacy",
        "timeFieldSpec": {
            "incomingGranularitySpec": {"name": "epoch_s", "dataType": "LONG"}
        },
    }
    assert db.get_columns()[["column_name", "data_type"]].values.tolist() == [
        ["epoch_s", "LONG"]
    ]


def test_nullability_is_unknown_without_null_handling() -> None:
    db, cluster = _database()
    cluster.tables = ["trips"]
    cluster.schemas["trips"] = _trips_schema(null_handling=False)
    assert db.get_columns()["is_nullable"].isna().all()


def test_nullability_follows_not_null_with_null_handling() -> None:
    db, cluster = _database()
    cluster.tables = ["trips"]
    cluster.schemas["trips"] = _trips_schema(null_handling=True)
    assert db.get_columns()["is_nullable"].tolist() == [False, True, True, True, True]


def test_table_without_a_schema_is_dropped_from_tables_and_columns() -> None:
    """Listing a table with no columns would fail ingestion's coverage check."""
    db, cluster = _database()
    cluster.tables = ["trips", "orphan"]
    cluster.schemas["trips"] = _trips_schema()
    assert db.get_tables()["table_name"].tolist() == ["trips"]
    assert set(db.get_columns()["table_name"]) == {"trips"}


def test_no_describable_table_raises_rather_than_ingesting_nothing() -> None:
    db, cluster = _database()
    cluster.tables = ["orphan"]
    with pytest.raises(PinotError, match="schema for none"):
        db.get_tables()


def test_introspection_lists_and_describes_once() -> None:
    db, cluster = _database()
    cluster.tables = ["trips"]
    cluster.schemas["trips"] = _trips_schema()
    db.get_tables()
    db.get_columns()
    db.get_pks()
    paths = [request.url.path for request in cluster.requests]
    assert paths == ["/tables", "/tables/trips/schema"]


def test_get_pks_reports_upsert_keys_in_order() -> None:
    db, cluster = _database()
    cluster.tables = ["trips"]
    cluster.schemas["trips"] = _trips_schema()
    assert db.get_pks().to_dict("records") == [
        {
            "table_schema": "default",
            "table_name": "trips",
            "column_name": "trip_id",
            "ordinal_position": 1,
        },
        {
            "table_schema": "default",
            "table_name": "trips",
            "column_name": "city",
            "ordinal_position": 2,
        },
    ]


@pytest.mark.parametrize(
    ("method", "columns"),
    [
        (
            PinotDatabase.get_fks,
            [
                "table_schema",
                "table_name",
                "column_name",
                "referenced_schema",
                "referenced_table",
                "referenced_column",
            ],
        ),
        (
            PinotDatabase.get_views,
            ["table_schema", "table_name", "view_definition"],
        ),
        (PinotDatabase.get_queries, ["end_time", "query_text"]),
    ],
)
def test_unsupported_metadata_is_correctly_shaped_and_empty(
    method: Callable[[PinotDatabase], pd.DataFrame], columns: list[str]
) -> None:
    db, cluster = _database()
    frame = method(db)
    assert frame.empty
    assert list(frame.columns) == columns
    assert cluster.requests == []


def test_qualify_uses_a_named_database_as_schema() -> None:
    db, _ = _database(database="analytics")
    assert db.qualify("analytics", "trips") == '"analytics"."trips"'


def test_qualify_leaves_default_bare_for_pre_database_clusters() -> None:
    db, _ = _database()
    assert db.qualify("default", "trips") == '"trips"'


def test_empty_cluster_returns_shaped_frames() -> None:
    db, _ = _database()
    assert list(db.get_tables().columns) == [
        "table_schema",
        "table_name",
        "table_type",
    ]
    assert db.get_columns().empty
    assert db.get_pks().empty
