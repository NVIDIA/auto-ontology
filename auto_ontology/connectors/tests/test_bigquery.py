# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BigQuery connector tests.

Everything here is network-free: the connector is exercised against a fake
``bigquery.Client``, so no project or credentials are needed.
"""

from __future__ import annotations

import base64
import concurrent.futures
import datetime as dt
import decimal
import json
from pathlib import Path
from typing import Any, Callable
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

import pandas as pd
import pytest
from google.cloud.bigquery import SchemaField
from google.cloud.bigquery.table import ColumnReference, ForeignKey, TableReference

from auto_ontology.catalog.constants import TableTypes
from auto_ontology.connectors.base import StatementTimeout
from auto_ontology.connectors.bigquery import (
    BytesBilledLimitExceeded,
    DEFAULT_MAXIMUM_BYTES_BILLED,
    TABLE_SUFFIX_COLUMN,
    BigQueryDatabase,
    _decode_credentials,
    _parse_connection_string,
    shard_families,
)
from auto_ontology.connectors.connection_string_factory import build_connection_string
from auto_ontology.connectors.registry import (
    CONNECTOR_REGISTRY,
    _redact,
    create_connector,
)

PROJECT = "my-project"
KEY = {
    "type": "service_account",
    "project_id": PROJECT,
    "client_email": "reader@my-project.iam.gserviceaccount.com",
    "private_key": "-----BEGIN PRIVATE KEY-----\nabc+/=\n-----END PRIVATE KEY-----\n",
    "token_uri": "https://oauth2.googleapis.com/token",
}


def _connection(**overrides: object) -> dict[str, object]:
    connection: dict[str, object] = {
        "type": "bigquery",
        "database": PROJECT,
        "credentials_json": json.dumps(KEY),
    }
    connection.update(overrides)
    return connection


def _database(
    schemas: list[str] | None = None, **overrides: object
) -> BigQueryDatabase:
    return BigQueryDatabase(
        build_connection_string(_connection(**overrides)), schemas=schemas
    )


# ----------------------------------------------------------------------
# Fake client
# ----------------------------------------------------------------------


class _FakeRow:
    def __init__(self, values: list[Any]) -> None:
        self._values = values

    def values(self) -> list[Any]:
        return self._values


class _FakeRows:
    def __init__(self, columns: list[str], rows: list[list[Any]]) -> None:
        self.schema = [SchemaField(name, "STRING") for name in columns]
        self._rows = rows

    def __iter__(self) -> Any:
        return iter(_FakeRow(row) for row in self._rows)


class _FakeJob:
    def __init__(self, outcome: _FakeRows | BaseException) -> None:
        self.job_id = "job-1"
        self._outcome = outcome
        self.result_timeout: float | None = None
        self.cancelled = False

    def result(self, timeout: float | None = None) -> _FakeRows:
        self.result_timeout = timeout
        if isinstance(self._outcome, BaseException):
            raise self._outcome
        return self._outcome

    def cancel(self, timeout: float | None = None) -> None:
        self.cancelled = True
        self.cancel_timeout = timeout


class _FakeDataset:
    def __init__(self, dataset_id: str) -> None:
        self.dataset_id = dataset_id


class _FakeClient:
    def __init__(
        self,
        answer: Callable[[str], _FakeRows | BaseException] | None = None,
        datasets: list[str] | None = None,
        tables: dict[str, Any] | None = None,
    ) -> None:
        self._answer = answer or (lambda sql: _FakeRows([], []))
        self._datasets = datasets or []
        self._tables = tables or {}
        self.statements: list[str] = []
        self.configs: list[Any] = []
        self.jobs: list[_FakeJob] = []
        self.closed = False

    def query(
        self, sql: str, job_config: Any = None, timeout: float | None = None
    ) -> _FakeJob:
        self.statements.append(sql)
        self.configs.append(job_config)
        job = _FakeJob(self._answer(sql))
        self.jobs.append(job)
        return job

    def list_datasets(self, project: str | None = None) -> list[_FakeDataset]:
        return [_FakeDataset(name) for name in self._datasets]

    def get_table(self, table_id: str, timeout: float | None = None) -> Any:
        return self._tables[table_id]

    def close(self) -> None:
        self.closed = True


def _wire(database: BigQueryDatabase, client: _FakeClient) -> _FakeClient:
    database._client = client  # noqa: SLF001 - test double for the driver
    return client


# ----------------------------------------------------------------------
# Connection string
# ----------------------------------------------------------------------


def test_bigquery_is_registered() -> None:
    assert CONNECTOR_REGISTRY["bigquery"] is BigQueryDatabase


def test_the_key_round_trips_through_the_url() -> None:
    """The PEM inside carries ``+`` and ``/``, which a raw query string mangles."""
    settings = _parse_connection_string(build_connection_string(_connection()))

    assert settings["project"] == PROJECT
    assert settings["credentials"] == KEY


def test_the_url_carries_location_and_dataset() -> None:
    url = build_connection_string(_connection(location="EU", schema="sales"))
    query = parse_qs(urlparse(url).query)

    assert urlparse(url).netloc == PROJECT
    assert query["location"] == ["EU"]
    assert query["schema"] == ["sales"]


def test_a_ui_connection_requires_a_key() -> None:
    """Without one the connector would run as the pod's own identity."""
    with pytest.raises(ValueError, match="credentials_json"):
        build_connection_string(_connection(credentials_json=""))


def test_a_ui_connection_requires_a_project() -> None:
    with pytest.raises(ValueError, match="database"):
        build_connection_string(_connection(database=""))


def test_a_key_that_is_not_json_is_rejected() -> None:
    with pytest.raises(ValueError, match="JSON"):
        build_connection_string(_connection(credentials_json="not-a-key"))


def test_raw_json_credentials_are_accepted_in_a_hand_written_url() -> None:
    assert _decode_credentials(json.dumps(KEY)) == KEY


def test_a_key_without_a_token_endpoint_gets_googles() -> None:
    key = {k: v for k, v in KEY.items() if k != "token_uri"}
    assert _decode_credentials(json.dumps(key)) == KEY


def test_credentials_must_be_a_service_account_key() -> None:
    encoded = base64.urlsafe_b64encode(b'{"type": "authorized_user"}').decode()
    with pytest.raises(ValueError, match="service-account"):
        _decode_credentials(encoded)


@pytest.mark.parametrize(
    "token_uri",
    [
        "https://oauth2.googleapis.com/token",
        "https://accounts.google.com/o/oauth2/token",
    ],
)
def test_a_key_may_name_googles_token_endpoint(token_uri: str) -> None:
    key = {**KEY, "token_uri": token_uri}
    assert _decode_credentials(json.dumps(key))["token_uri"] == token_uri


@pytest.mark.parametrize(
    "token_uri",
    [
        "http://169.254.169.254/computeMetadata/v1/token",
        "https://oauth2.googleapis.com.evil.example/token",
        "http://oauth2.googleapis.com/token",
    ],
)
def test_a_key_naming_another_token_endpoint_is_refused(token_uri: str) -> None:
    key = {**KEY, "token_uri": token_uri}
    with pytest.raises(ValueError, match="token endpoint"):
        _decode_credentials(json.dumps(key))


def test_the_path_names_the_database() -> None:
    settings = _parse_connection_string("bigquery://billing/austin")

    assert settings["project"] == "billing"
    assert settings["database"] == "austin"


def test_without_a_path_the_project_names_the_database() -> None:
    assert _parse_connection_string("bigquery://p")["database"] == "p"


def test_named_datasets_keep_their_own_project() -> None:
    settings = _parse_connection_string(
        "bigquery://billing/austin"
        "?dataset=bigquery-public-data.austin_311&dataset=local_ds"
    )

    assert settings["datasets"] == [
        ("bigquery-public-data", "austin_311"),
        ("billing", "local_ds"),
    ]


def test_one_dataset_name_cannot_belong_to_two_projects() -> None:
    with pytest.raises(ValueError, match="both"):
        _parse_connection_string("bigquery://b/db?dataset=p1.sales&dataset=p2.sales")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1000", 1000),
        ("20GiB", 20 * 1024**3),
        ("500MB", 500 * 10**6),
        ("1.5KiB", 1536),
        ("unlimited", None),
        ("none", None),
        ("0", None),
    ],
)
def test_the_byte_cap_can_be_overridden(raw: str, expected: int | None) -> None:
    url = f"bigquery://p?maximum_bytes_billed={raw}"
    assert _parse_connection_string(url)["maximum_bytes_billed"] == expected


def test_the_byte_cap_defaults_and_rejects_nonsense() -> None:
    assert (
        _parse_connection_string("bigquery://p")["maximum_bytes_billed"]
        == DEFAULT_MAXIMUM_BYTES_BILLED
    )
    with pytest.raises(ValueError, match="byte size"):
        _parse_connection_string("bigquery://p?maximum_bytes_billed=lots")


@pytest.mark.parametrize("raw", ["0.1B", "0.0001KB", "0.0000001MB", "0.5"])
def test_a_cap_below_one_byte_is_rejected_rather_than_lifted(raw: str) -> None:
    with pytest.raises(ValueError, match="at least one byte"):
        _parse_connection_string(f"bigquery://p?maximum_bytes_billed={raw}")


@pytest.mark.parametrize(
    ("raw", "sent"),
    [
        (None, str(DEFAULT_MAXIMUM_BYTES_BILLED)),
        ("20GiB", str(20 * 1024**3)),
        ("0", None),
        ("none", None),
        ("unlimited", None),
    ],
)
def test_the_cap_reaches_the_api_only_when_finite(
    raw: str | None, sent: str | None
) -> None:
    url = build_connection_string(_connection())
    if raw is not None:
        url += f"&maximum_bytes_billed={raw}"
    database = BigQueryDatabase(url)
    client = _wire(database, _FakeClient())

    database.execute("SELECT 1")

    query = client.configs[0].to_api_repr()["query"]
    assert query.get("maximumBytesBilled") == sent


def test_a_url_without_a_project_is_rejected() -> None:
    with pytest.raises(ValueError, match="project"):
        _parse_connection_string("bigquery://")


def test_the_key_is_masked_in_logs() -> None:
    url = build_connection_string(_connection(location="US"))
    redacted = _redact(url)

    assert "credentials=***" in redacted
    assert "location=US" in redacted
    assert url.split("credentials=")[1].split("&")[0] not in redacted


def test_create_connector_passes_the_dataset_allowlist() -> None:
    database = create_connector(
        build_connection_string(_connection()), schemas=["sales", "ops"]
    )

    assert isinstance(database, BigQueryDatabase)
    assert database._datasets() == ["ops", "sales"]  # noqa: SLF001


def test_constructing_the_connector_makes_no_network_call() -> None:
    with patch("google.cloud.bigquery.Client") as client_cls:
        _database()
    client_cls.assert_not_called()


# ----------------------------------------------------------------------
# Credentials
# ----------------------------------------------------------------------


def test_an_explicit_key_is_used_rather_than_default_credentials() -> None:
    with (
        patch(
            "google.oauth2.service_account.Credentials.from_service_account_info"
        ) as from_info,
        patch("google.auth.default") as auth_default,
        patch("google.cloud.bigquery.Client") as client_cls,
    ):
        _database(location="US").client

    from_info.assert_called_once()
    assert from_info.call_args.args[0] == KEY
    auth_default.assert_not_called()
    assert client_cls.call_args.kwargs["credentials"] is from_info.return_value
    assert client_cls.call_args.kwargs["project"] == PROJECT
    assert client_cls.call_args.kwargs["location"] == "US"


def test_a_key_file_is_loaded_from_disk(tmp_path: Path) -> None:
    key_file = tmp_path / "sa.json"
    key_file.write_text(json.dumps(KEY))
    with (
        patch(
            "google.oauth2.service_account.Credentials.from_service_account_info"
        ) as from_info,
        patch("google.cloud.bigquery.Client") as client_cls,
    ):
        BigQueryDatabase(f"bigquery://p?credentials_file={key_file}").client

    assert from_info.call_args.args[0] == KEY
    assert client_cls.call_args.kwargs["credentials"] is from_info.return_value


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ({**KEY, "token_uri": "https://attacker.example/token"}, "token endpoint"),
        ({**KEY, "type": "authorized_user"}, "service-account"),
        ("not json", "valid JSON key"),
    ],
)
def test_a_key_file_gets_the_inline_key_checks(
    tmp_path: Path, contents: object, message: str
) -> None:
    key_file = tmp_path / "sa.json"
    key_file.write_text(contents if isinstance(contents, str) else json.dumps(contents))
    database = BigQueryDatabase(f"bigquery://p?credentials_file={key_file}")

    with pytest.raises(ValueError, match=message):
        database.client


def test_a_missing_key_file_is_reported_without_falling_back(tmp_path: Path) -> None:
    database = BigQueryDatabase(f"bigquery://p?credentials_file={tmp_path}/no.json")

    with (
        patch("google.cloud.bigquery.Client") as client_cls,
        pytest.raises(ValueError, match="cannot be read"),
    ):
        database.client

    client_cls.assert_not_called()


def test_no_credentials_defers_to_application_default() -> None:
    with patch("google.cloud.bigquery.Client") as client_cls:
        BigQueryDatabase("bigquery://p").client

    assert client_cls.call_args.kwargs["credentials"] is None


# ----------------------------------------------------------------------
# Identity
# ----------------------------------------------------------------------


def test_dialect_and_database_name() -> None:
    database = _database()

    assert database.dialect == "bigquery"
    assert database.database_name == PROJECT


def test_qualify_prepends_the_project() -> None:
    assert _database().qualify("sales", "orders") == "`my-project`.`sales`.`orders`"


def test_qualify_requires_a_dataset() -> None:
    with pytest.raises(ValueError, match="dataset"):
        _database().qualify(None, "orders")


# ----------------------------------------------------------------------
# Execution
# ----------------------------------------------------------------------


def test_execute_returns_a_frame() -> None:
    database = _database()
    _wire(
        database, _FakeClient(lambda sql: _FakeRows(["a", "b"], [[1, "x"], [2, "y"]]))
    )

    frame = database.execute("SELECT a, b FROM t")

    assert frame.to_dict("records") == [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]


def test_a_statement_without_rows_returns_an_empty_frame() -> None:
    database = _database()
    _wire(database, _FakeClient())

    assert database.execute("CREATE TEMP TABLE t (a INT64)").empty


def test_every_job_is_capped_and_labelled() -> None:
    database = _database()
    client = _wire(database, _FakeClient())

    database.execute("SELECT 1")
    config = client.configs[0]

    assert config.maximum_bytes_billed == DEFAULT_MAXIMUM_BYTES_BILLED
    assert config.labels == {"source": "auto-ontology"}
    assert config.default_dataset is None
    assert config.job_timeout_ms is None


def test_a_single_dataset_becomes_the_default_dataset() -> None:
    database = _database(schema="sales")
    client = _wire(database, _FakeClient())

    database.execute("SELECT * FROM orders")

    assert str(client.configs[0].default_dataset) == "my-project.sales"


def test_parameters_are_bound_with_explicit_types() -> None:
    database = _database()
    client = _wire(database, _FakeClient())

    database.execute(
        "SELECT ?",
        [
            True,
            42,
            1.5,
            decimal.Decimal("1.25"),
            dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc),
            dt.datetime(2026, 1, 1),
            dt.date(2026, 1, 1),
            None,
            "text",
        ],
    )
    params = client.configs[0].query_parameters

    assert [p.type_ for p in params] == [
        "BOOL",
        "INT64",
        "FLOAT64",
        "BIGNUMERIC",
        "TIMESTAMP",
        "DATETIME",
        "DATE",
        "STRING",
        "STRING",
    ]
    assert params[0].value is True
    assert params[7].value is None


def test_a_timeout_is_sent_to_the_server_and_waited_on_with_grace() -> None:
    database = _database()
    client = _wire(database, _FakeClient())

    database.execute("SELECT 1", timeout_s=30)

    assert int(client.configs[0].job_timeout_ms) == pytest.approx(30_000, abs=500)
    assert client.jobs[0].result_timeout == pytest.approx(40.0, abs=0.5)


def test_a_client_side_timeout_cancels_the_job() -> None:
    database = _database()
    client = _wire(database, _FakeClient(lambda sql: concurrent.futures.TimeoutError()))

    with pytest.raises(StatementTimeout):
        database.execute("SELECT 1", timeout_s=1)

    assert client.jobs[0].cancelled
    assert client.jobs[0].cancel_timeout is not None


def test_a_server_side_timeout_is_a_statement_timeout() -> None:
    from google.api_core.exceptions import BadRequest

    error = BadRequest(
        "Job timed out after 1s", errors=[{"reason": "timeout", "message": "x"}]
    )
    database = _database()
    _wire(database, _FakeClient(lambda sql: error))

    with pytest.raises(StatementTimeout):
        database.execute("SELECT 1", timeout_s=1)


def test_a_job_stopped_at_its_timeout_is_a_statement_timeout() -> None:
    """What BigQuery actually raises when ``job_timeout_ms`` fires."""
    from google.api_core.exceptions import Cancelled

    error = Cancelled(
        "Job execution was cancelled: Job timed out after 1 sec",
        errors=[
            {
                "message": "Job execution was cancelled: Job timed out after 1 sec",
                "domain": "global",
                "reason": "stopped",
            }
        ],
    )
    database = _database()
    _wire(database, _FakeClient(lambda sql: error))

    with pytest.raises(StatementTimeout):
        database.execute("SELECT 1", timeout_s=1)


@pytest.mark.parametrize(
    "error",
    [
        {"reason": "stopped", "message": "User requested cancellation"},
        {"reason": "invalidQuery", "message": "Connection timed out reading x"},
    ],
)
def test_a_timeout_is_told_by_its_reason_not_any_timed_out_text(
    error: dict[str, str],
) -> None:
    from google.api_core.exceptions import BadRequest

    database = _database()
    _wire(
        database, _FakeClient(lambda sql: BadRequest(error["message"], errors=[error]))
    )

    with pytest.raises(BadRequest):
        database.execute("SELECT 1", timeout_s=1)


def test_other_errors_propagate_unchanged() -> None:
    from google.api_core.exceptions import BadRequest

    error = BadRequest("Unrecognized name: nope", errors=[{"reason": "invalidQuery"}])
    database = _database()
    _wire(database, _FakeClient(lambda sql: error))

    with pytest.raises(BadRequest):
        database.execute("SELECT nope", timeout_s=1)


def test_a_query_over_the_byte_cap_says_how_much_it_would_scan() -> None:
    from google.api_core.exceptions import InternalServerError

    text = (
        "Query exceeded limit for bytes billed: 21474836480. "
        "189457760256 or higher required."
    )
    error = InternalServerError(
        text, errors=[{"reason": "bytesBilledLimitExceeded", "message": text}]
    )
    database = _database()
    _wire(database, _FakeClient(lambda sql: error))

    with pytest.raises(BytesBilledLimitExceeded) as caught:
        database.execute("SELECT * FROM big.events")

    message = str(caught.value)
    assert "would scan 176.4 GiB, over the 20.0 GiB cap" in message
    assert "exceeded limit for bytes billed" in message.lower()
    assert "_TABLE_SUFFIX" in message
    assert caught.value.errors[0]["reason"] == "bytesBilledLimitExceeded"
    assert caught.value.__cause__ is error


@pytest.mark.parametrize(
    ("size", "text"),
    [(512, "512.0 B"), (11 * 1024**2, "11.0 MiB"), (5 * 1024**4, "5.0 TiB")],
)
def test_byte_sizes_are_shown_in_a_readable_unit(size: int, text: str) -> None:
    assert BytesBilledLimitExceeded(size, None).args[0].count(text) == 1


# ----------------------------------------------------------------------
# Schema introspection
# ----------------------------------------------------------------------


def test_get_schemas_lists_datasets() -> None:
    database = _database()
    _wire(database, _FakeClient(datasets=["sales", "ops"]))

    assert database.get_schemas() == ["ops", "sales"]


def test_introspection_runs_once_per_dataset_in_scope() -> None:
    database = _database(schemas=["sales", "ops"])
    client = _wire(database, _FakeClient(datasets=["sales", "ops", "secret"]))
    database.get_tables()
    client.statements.clear()

    database.get_columns()

    assert len(client.statements) == 2
    assert all("COLUMN_FIELD_PATHS" in sql for sql in client.statements)
    assert "`my-project`.`ops`.INFORMATION_SCHEMA.COLUMNS" in client.statements[0]
    assert "`my-project`.`sales`.INFORMATION_SCHEMA.COLUMNS" in client.statements[1]


def test_without_an_allowlist_every_listed_dataset_is_introspected() -> None:
    database = _database()
    client = _wire(database, _FakeClient(datasets=["a", "b", "c"]))

    database.get_views()

    assert len(client.statements) == 3


def test_get_columns_keeps_top_level_fields_and_reports_booleans() -> None:
    database = _database(schema="sales")
    client = _wire(database, _FakeClient())

    database.get_columns()
    sql = client.statements[0]

    assert "p.field_path = c.column_name" in sql
    assert "c.is_nullable = 'YES' AS is_nullable" in sql


def test_get_tables_maps_types_and_decodes_descriptions() -> None:
    rows = [
        ["sales", "orders", "BASE TABLE", '"Customer orders"'],
        ["sales", "v_orders", "VIEW", None],
        ["sales", "mv_orders", "MATERIALIZED VIEW", None],
        ["sales", "ext", "EXTERNAL", None],
    ]
    database = _database(schema="sales")
    _wire(
        database,
        _FakeClient(
            lambda sql: _FakeRows(
                ["table_schema", "table_name", "table_type", "description"], rows
            )
        ),
    )

    frame = database.get_tables()

    assert frame["table_type"].tolist() == [
        TableTypes.BASE_TABLE,
        TableTypes.VIEW,
        TableTypes.MATERIALIZED_VIEW,
        TableTypes.BASE_TABLE,
    ]
    assert frame["description"].tolist()[0] == "Customer orders"
    assert frame["description"].tolist()[1] is None


def test_empty_introspection_keeps_the_contract_columns() -> None:
    database = _database(schema="sales")
    _wire(database, _FakeClient())

    assert list(database.get_tables().columns) == [
        "table_schema",
        "table_name",
        "table_type",
        "description",
    ]
    assert list(database.get_pks().columns) == [
        "table_schema",
        "table_name",
        "column_name",
        "ordinal_position",
    ]


def test_get_pks_reads_primary_key_constraints() -> None:
    database = _database(schema="sales")
    client = _wire(database, _FakeClient())

    database.get_pks()
    sql = client.statements[0]

    assert "constraint_type = 'PRIMARY KEY'" in sql
    assert "c.table_name = k.table_name" in sql


def _table_with_foreign_keys(*keys: ForeignKey) -> Any:
    class _Constraints:
        foreign_keys = list(keys)

    class _Table:
        table_constraints = _Constraints()

    return _Table()


def test_get_fks_pairs_composite_key_columns_exactly() -> None:
    """A two-column key yields two pairs, not the four a name-only join would."""
    key = ForeignKey(
        name="fk_line_order",
        referenced_table=TableReference.from_string("my-project.sales.orders"),
        column_references=[
            ColumnReference("order_id", "id"),
            ColumnReference("order_region", "region"),
        ],
    )
    cross_dataset = ForeignKey(
        name="fk_line_product",
        referenced_table=TableReference.from_string("my-project.catalog.products"),
        column_references=[ColumnReference("product_id", "id")],
    )
    database = _database(schema="sales")
    _wire(
        database,
        _FakeClient(
            lambda sql: (
                _FakeRows(["table_schema", "table_name"], [["sales", "order_lines"]])
                if "TABLE_CONSTRAINTS" in sql
                else _FakeRows(
                    ["table_schema", "table_name", "table_type"],
                    [["sales", "order_lines", "BASE TABLE"]],
                )
            ),
            tables={
                "my-project.sales.order_lines": _table_with_foreign_keys(
                    key, cross_dataset
                )
            },
        ),
    )

    frame = database.get_fks()

    assert frame.to_dict("records") == [
        {
            "table_schema": "sales",
            "table_name": "order_lines",
            "column_name": "order_id",
            "referenced_schema": "sales",
            "referenced_table": "orders",
            "referenced_column": "id",
        },
        {
            "table_schema": "sales",
            "table_name": "order_lines",
            "column_name": "order_region",
            "referenced_schema": "sales",
            "referenced_table": "orders",
            "referenced_column": "region",
        },
        {
            "table_schema": "sales",
            "table_name": "order_lines",
            "column_name": "product_id",
            "referenced_schema": "catalog",
            "referenced_table": "products",
            "referenced_column": "id",
        },
    ]


def test_a_key_into_another_project_is_left_out() -> None:
    """``other-project.sales.events_20260101`` is not this connection's ``sales``."""
    foreign = ForeignKey(
        name="fk_foreign",
        referenced_table=TableReference.from_string(
            "other-project.sales.events_20260101"
        ),
        column_references=[ColumnReference("event_id", "id")],
    )
    local = ForeignKey(
        name="fk_local",
        referenced_table=TableReference.from_string("my-project.sales.orders"),
        column_references=[ColumnReference("order_id", "id")],
    )
    database = _database(schema="sales")
    _wire(
        database,
        _FakeClient(
            lambda sql: (
                _FakeRows(["table_schema", "table_name"], [["sales", "order_lines"]])
                if "TABLE_CONSTRAINTS" in sql
                else _FakeRows(
                    ["table_schema", "table_name", "table_type"],
                    [["sales", "order_lines", "BASE TABLE"]],
                )
            ),
            tables={
                "my-project.sales.order_lines": _table_with_foreign_keys(foreign, local)
            },
        ),
    )

    frame = database.get_fks()

    assert frame[["column_name", "referenced_table"]].to_dict("records") == [
        {"column_name": "order_id", "referenced_table": "orders"}
    ]


def test_get_fks_without_constraints_is_empty_and_shaped() -> None:
    database = _database(schema="sales")
    _wire(database, _FakeClient())

    frame = database.get_fks()

    assert frame.empty
    assert "referenced_column" in frame.columns


# ----------------------------------------------------------------------
# Named datasets in other projects
# ----------------------------------------------------------------------

PUBLIC_URL = (
    "bigquery://billing/austin"
    "?dataset=bigquery-public-data.austin_311"
    "&dataset=bigquery-public-data.austin_crime"
)


def test_connections_billing_one_project_are_routed_by_their_path() -> None:
    austin = BigQueryDatabase(PUBLIC_URL)
    bbc = BigQueryDatabase(
        "bigquery://billing/bbc?dataset=bigquery-public-data.bbc_news"
    )

    assert austin.database_name == "austin"
    assert bbc.database_name == "bbc"


def test_jobs_are_billed_to_the_host_project() -> None:
    with patch("google.cloud.bigquery.Client") as client_cls:
        BigQueryDatabase(PUBLIC_URL).client

    assert client_cls.call_args.kwargs["project"] == "billing"


def test_named_datasets_are_the_schemas_and_each_is_checked() -> None:
    database = BigQueryDatabase(PUBLIC_URL)
    client = _wire(database, _FakeClient())
    fetched: list[str] = []
    client.get_dataset = fetched.append  # type: ignore[attr-defined]

    assert database.get_schemas() == ["austin_311", "austin_crime"]
    assert fetched == [
        "bigquery-public-data.austin_311",
        "bigquery-public-data.austin_crime",
    ]


def test_the_allowlist_narrows_named_datasets() -> None:
    database = BigQueryDatabase(PUBLIC_URL, schemas=["austin_crime"])

    assert database._datasets() == ["austin_crime"]  # noqa: SLF001


def test_dataset_ids_that_differ_only_in_case_stay_distinct() -> None:
    """BigQuery dataset IDs are case-sensitive; Sales and sales can coexist."""
    url = "bigquery://billing?dataset=one.Sales&dataset=two.sales"

    picked = BigQueryDatabase(url, schemas=["Sales"])

    assert picked._datasets() == ["Sales"]  # noqa: SLF001
    assert picked.qualify("Sales", "t") == "`one`.`Sales`.`t`"
    assert picked.qualify("sales", "t") == "`two`.`sales`.`t`"
    assert picked.qualify("SALES", "t") == "`billing`.`SALES`.`t`"


def test_sql_on_a_billing_dataset_is_not_sent_to_its_case_twin() -> None:
    url = "bigquery://billing?dataset=billing.Sales&dataset=other.sales"
    database = BigQueryDatabase(url)
    client = _wire(database, _FakeClient())

    database.execute("SELECT COUNT(*) FROM Sales.orders")
    database.execute("SELECT COUNT(*) FROM sales.orders")

    assert "other" not in client.statements[0]
    assert "other.sales.orders" in client.statements[1]


def test_a_dataset_named_in_another_case_resolves_when_unambiguous() -> None:
    database = BigQueryDatabase(PUBLIC_URL, schemas=["AUSTIN_CRIME"])

    assert database._datasets() == ["austin_crime"]  # noqa: SLF001
    assert database.qualify("Austin_311", "requests").startswith(
        "`bigquery-public-data`."
    )


def test_shard_families_of_case_twin_datasets_stay_apart() -> None:
    rows = [
        ["Sales", "events_20260101", "BASE TABLE", None],
        ["Sales", "events_20260102", "BASE TABLE", None],
        ["sales", "events_20260101", "BASE TABLE", None],
    ]
    database = BigQueryDatabase("bigquery://billing?dataset=Sales&dataset=sales")
    _wire(
        database,
        _FakeClient(
            lambda sql: _FakeRows(
                ["table_schema", "table_name", "table_type", "description"], rows
            )
        ),
    )

    families = database._shard_families()  # noqa: SLF001

    assert list(families["Sales"]) == ["events_*"]
    assert families["sales"] == {}


def test_introspection_reads_each_dataset_in_its_own_project() -> None:
    database = BigQueryDatabase(PUBLIC_URL)
    client = _wire(database, _FakeClient())

    database.get_views()

    assert (
        "`bigquery-public-data`.`austin_311`.INFORMATION_SCHEMA.VIEWS"
        in client.statements[0]
    )


def test_qualify_uses_the_datasets_project() -> None:
    assert (
        BigQueryDatabase(PUBLIC_URL).qualify("austin_311", "requests")
        == "`bigquery-public-data`.`austin_311`.`requests`"
    )


def test_two_part_names_get_their_datasets_project() -> None:
    database = BigQueryDatabase(PUBLIC_URL)
    client = _wire(database, _FakeClient())

    database.execute("SELECT COUNT(*) FROM austin_311.requests")

    assert "`bigquery-public-data`.austin_311.requests" in client.statements[0]


def test_sql_naming_its_projects_is_sent_unchanged() -> None:
    sql = "SELECT 1 FROM `bigquery-public-data.austin_311.requests`"
    database = BigQueryDatabase(PUBLIC_URL)
    client = _wire(database, _FakeClient())

    database.execute(sql)

    assert client.statements == [sql]


def test_unknown_datasets_are_left_for_bigquery_to_resolve() -> None:
    sql = "SELECT 1 FROM other_ds.t"
    database = BigQueryDatabase(PUBLIC_URL)
    client = _wire(database, _FakeClient())

    database.execute(sql)

    assert client.statements == [sql]


def test_a_single_named_dataset_is_the_default_dataset() -> None:
    database = BigQueryDatabase(
        "bigquery://billing/bbc?dataset=bigquery-public-data.bbc_news"
    )
    client = _wire(database, _FakeClient())

    database.execute("SELECT * FROM fulltext")

    assert str(client.configs[0].default_dataset) == "bigquery-public-data.bbc_news"


# ----------------------------------------------------------------------
# Query history
# ----------------------------------------------------------------------


def test_query_history_needs_a_location() -> None:
    database = _database()
    client = _wire(database, _FakeClient())

    frame = database.get_queries()

    assert list(frame.columns) == ["end_time", "query_text"]
    assert client.statements == []


def test_query_history_reads_the_regional_jobs_view() -> None:
    database = _database(location="US")
    client = _wire(
        database,
        _FakeClient(
            lambda sql: _FakeRows(
                ["end_time", "query_text"],
                [[pd.Timestamp("2026-01-01", tz="UTC"), "SELECT 1"]],
            )
        ),
    )

    frame = database.get_queries(hours=6)
    sql = client.statements[0]

    assert "`my-project`.`region-US`.INFORMATION_SCHEMA.JOBS_BY_PROJECT" in sql
    assert "AND end_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 6 HOUR)" in sql
    assert (
        "creation_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 12 HOUR)" in sql
    )
    assert "label.value = 'auto-ontology'" in sql
    assert frame["query_text"].tolist() == ["SELECT 1"]


def test_query_history_without_permission_is_empty() -> None:
    from google.api_core.exceptions import Forbidden

    database = _database(location="US")
    _wire(database, _FakeClient(lambda sql: Forbidden("Access Denied")))

    frame = database.get_queries()

    assert frame.empty
    assert list(frame.columns) == ["end_time", "query_text"]


# ----------------------------------------------------------------------
# Lifecycle
# ----------------------------------------------------------------------


def test_ping_runs_a_job() -> None:
    database = _database()
    client = _wire(database, _FakeClient())

    database.ping()

    assert client.statements == ["SELECT 1"]


def test_close_releases_the_client() -> None:
    database = _database()
    client = _wire(database, _FakeClient())

    database.close()

    assert client.closed
    assert database._client is None  # noqa: SLF001


# ----------------------------------------------------------------------
# Date-sharded tables
# ----------------------------------------------------------------------

_SHARDED_TABLES = [
    ["ga", "ga_sessions_20170801", "BASE TABLE", '"Latest day"'],
    ["ga", "ga_sessions_20170731", "BASE TABLE", None],
    ["ga", "ga_sessions_20160801", "BASE TABLE", None],
    ["ga", "visitors", "BASE TABLE", None],
    ["ga", "v_sessions_20170801", "VIEW", None],
]


def _sharded_client() -> _FakeClient:
    def answer(sql: str) -> _FakeRows:
        if "REGEXP_CONTAINS" in sql:
            return _FakeRows(
                ["table_name", "column_name", "data_type"],
                [
                    ["ga_sessions_20170801", "visitId", "STRING"],
                    ["ga_sessions_20170801", "date", "STRING"],
                    ["ga_sessions_20170731", "visitId", "STRING"],
                    ["ga_sessions_20160801", "visitId", "INT64"],
                ],
            )
        if "INFORMATION_SCHEMA.COLUMNS" in sql:
            return _FakeRows(
                [
                    "table_schema",
                    "table_name",
                    "column_name",
                    "data_type",
                    "is_nullable",
                    "ordinal_position",
                    "description",
                ],
                [
                    ["ga", name, column, "STRING", "YES", position, None]
                    for name in ("ga_sessions_20170801", "ga_sessions_20160801")
                    for position, column in enumerate(["visitId", "date"], start=1)
                ]
                + [["ga", "visitors", "id", "INT64", "NO", 1, None]],
            )
        if "KEY_COLUMN_USAGE" in sql:
            return _FakeRows(
                ["table_schema", "table_name", "column_name", "ordinal_position"],
                [
                    ["ga", "ga_sessions_20170801", "visitId", 1],
                    ["ga", "ga_sessions_20160801", "visitId", 1],
                ],
            )
        if "INFORMATION_SCHEMA.TABLES" in sql:
            return _FakeRows(
                ["table_schema", "table_name", "table_type", "description"],
                _SHARDED_TABLES,
            )
        return _FakeRows([], [])

    return _FakeClient(answer)


def test_a_shard_with_a_column_the_newest_lacks_stays_a_table_of_its_own() -> None:
    """The wildcard reads with the newest shard's columns, so that one is lost."""
    families = shard_families(
        ["trees_1995", "trees_2005", "trees_2015"],
        {
            "trees_1995": {"recordid": "INT64"},
            "trees_2005": {"tree_id": "INT64"},
            "trees_2015": {"tree_id": "INT64", "block_id": "INT64"},
        },
    )

    assert [(f.shards, f.also_matches) for f in families] == [
        (("trees_2005", "trees_2015"), ("trees_1995",))
    ]


def test_a_shard_with_a_changed_type_stays_a_table_of_its_own() -> None:
    families = shard_families(
        ["s_2018", "s_2019", "s_2020"],
        {
            "s_2018": {"a": "INT64"},
            "s_2019": {"a": "STRING"},
            "s_2020": {"a": "STRING"},
        },
    )

    assert [f.shards for f in families] == [("s_2019", "s_2020")]


def test_a_family_left_with_one_shard_is_not_collapsed() -> None:
    columns = {"s_2019": {"a": "INT64"}, "s_2020": {"b": "INT64"}}

    assert shard_families(["s_2019", "s_2020"], columns) == []


def test_shard_families_need_two_shards_and_a_date_suffix() -> None:
    families = shard_families(
        ["events_20210102", "events_20210101", "gsod1929", "icoads_core_1662_2000"]
    )

    assert [(f.wildcard, f.shards) for f in families] == [
        ("events_*", ("events_20210101", "events_20210102"))
    ]


def test_a_wildcard_names_the_other_tables_it_matches() -> None:
    (family,) = shard_families(["ghcnd_2015", "ghcnd_2016", "ghcnd_countries"])

    assert family.also_matches == ("ghcnd_countries",)
    assert "also matches ghcnd_countries" in family.note()


def test_no_wildcard_is_formed_over_a_prefix_a_view_matches() -> None:
    """BigQuery rejects such a wildcard even with a _TABLE_SUFFIX filter."""
    names = ["events_20260101", "events_20260102"]

    assert shard_families(names, views=["events_summary"]) == []
    assert [f.wildcard for f in shard_families(names, views=["summary"])] == [
        "events_*"
    ]


@pytest.mark.parametrize("list_tables_first", [True, False])
def test_shards_beside_a_matching_view_stay_separate(list_tables_first: bool) -> None:
    rows = [
        ["ga", "events_20260101", "BASE TABLE", None],
        ["ga", "events_20260102", "BASE TABLE", None],
        ["ga", "events_summary", "VIEW", None],
    ]
    database = _database(schema="ga")
    _wire(
        database,
        _FakeClient(
            lambda sql: (
                _FakeRows(
                    ["table_schema", "table_name", "table_type", "description"], rows
                )
                if "INFORMATION_SCHEMA.TABLES" in sql
                else _FakeRows([], [])
            )
        ),
    )
    if list_tables_first:
        assert database.get_tables()["table_name"].tolist() == [
            "events_20260101",
            "events_20260102",
            "events_summary",
        ]

    assert database._shard_families() == {"ga": {}}  # noqa: SLF001


def test_date_shards_collapse_into_one_wildcard_table() -> None:
    database = _database(schema="ga")
    _wire(database, _sharded_client())

    frame = database.get_tables()

    assert frame["table_name"].tolist() == [
        "ga_sessions_*",
        "visitors",
        "v_sessions_20170801",
    ]
    description = frame["description"].tolist()[0]
    assert "3 date-sharded tables ga_sessions_20160801 ... ga_sessions_20170801" in (
        description
    )
    assert description.endswith("Latest day")


def test_a_wildcard_table_has_the_latest_shard_columns_and_table_suffix() -> None:
    database = _database(schema="ga")
    _wire(database, _sharded_client())
    database.get_tables()

    frame = database.get_columns()
    wildcard = frame[frame["table_name"] == "ga_sessions_*"]

    assert wildcard["column_name"].tolist() == ["visitId", "date", TABLE_SUFFIX_COLUMN]
    assert wildcard["ordinal_position"].tolist()[-1] == 3
    assert set(frame["table_name"]) == {"ga_sessions_*", "visitors"}


def test_keys_of_the_latest_shard_belong_to_the_wildcard() -> None:
    database = _database(schema="ga")
    _wire(database, _sharded_client())

    frame = database.get_pks()

    assert frame[["table_name", "column_name"]].values.tolist() == [
        ["ga_sessions_*", "visitId"]
    ]


def test_qualifying_a_wildcard_reads_its_latest_shard() -> None:
    """Profiling a wildcard would otherwise scan every shard, LIMIT or not."""
    database = _database(schema="ga")
    _wire(database, _sharded_client())

    assert database.qualify("ga", "ga_sessions_*").endswith(
        "`ga`.`ga_sessions_20170801`"
    )
    assert database.qualify("ga", "visitors").endswith("`ga`.`visitors`")


def test_a_shard_that_lands_later_is_qualified_to_after_an_hour(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [10_000.0]
    monkeypatch.setattr(
        "auto_ontology.connectors.bigquery.time.monotonic", lambda: clock[0]
    )
    rows = [
        ["ga", "ga_sessions_20170731", "BASE TABLE", None],
        ["ga", "ga_sessions_20170801", "BASE TABLE", None],
    ]
    database = _database(schema="ga")
    _wire(
        database,
        _FakeClient(
            lambda sql: _FakeRows(
                ["table_schema", "table_name", "table_type", "description"], rows
            )
        ),
    )
    assert database.qualify("ga", "ga_sessions_*").endswith("ga_sessions_20170801`")

    rows.append(["ga", "ga_sessions_20170802", "BASE TABLE", None])
    clock[0] += 60
    assert database.qualify("ga", "ga_sessions_*").endswith("ga_sessions_20170801`")

    clock[0] += 3600
    assert database.qualify("ga", "ga_sessions_*").endswith("ga_sessions_20170802`")


def test_a_struct_that_gained_fields_still_reads_through_the_wildcard() -> None:
    """GA360's ``hits`` gained nested fields over the years; older rows read NULL."""
    families = shard_families(
        ["ga_20160801", "ga_20170801"],
        {
            "ga_20160801": {"Hits": "ARRAY<STRUCT<hitNumber INT64>>"},
            "ga_20170801": {"hits": "ARRAY<STRUCT<hitNumber INT64, isExit BOOL>>"},
        },
    )

    assert [f.shards for f in families] == [("ga_20160801", "ga_20170801")]


# ----------------------------------------------------------------------
# First-rows reads
# ----------------------------------------------------------------------


class _StoredTable:
    def __init__(self, table_type: str = "TABLE") -> None:
        self.table_type = table_type
        self.schema = [SchemaField("id", "INT64"), SchemaField("name", "STRING")]


class _ListingClient(_FakeClient):
    def __init__(self, table_type: str = "TABLE") -> None:
        super().__init__(lambda sql: _FakeRows(["id"], [[9]]))
        self._stored = _StoredTable(table_type)
        self.listed: list[tuple[list[str], int]] = []

    def get_table(self, table_id: str, timeout: float | None = None) -> Any:
        assert table_id == "bigquery-public-data.ga.visitors"
        return self._stored

    def list_rows(
        self, table: Any, selected_fields: Any, max_results: int, timeout: Any
    ) -> list[_FakeRow]:
        self.listed.append(([f.name for f in selected_fields], max_results))
        return [_FakeRow([1, "a"][: len(selected_fields)])]


def _listing(table_type: str = "TABLE") -> tuple[BigQueryDatabase, _ListingClient]:
    database = _database(schema="ga")
    return database, _wire(database, _ListingClient(table_type))


def test_reading_first_rows_lists_them_instead_of_querying() -> None:
    """As a query it would scan, and bill, every row of the table."""
    database, client = _listing()

    frame = database.execute(
        "SELECT * FROM `bigquery-public-data`.`ga`.`visitors` LIMIT 1000"
    )

    assert client.statements == []
    assert client.listed == [(["id", "name"], 1000)]
    assert frame.values.tolist() == [[1, "a"]]


def test_first_rows_of_named_columns_are_listed_too() -> None:
    database, client = _listing()

    frame = database.execute(
        "SELECT `id` FROM `bigquery-public-data.ga.visitors` LIMIT 5"
    )

    assert client.listed == [(["id"], 5)]
    assert frame.columns.tolist() == ["id"]


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT id FROM `bigquery-public-data.ga.visitors` WHERE id > 1 LIMIT 5",
        "SELECT DISTINCT id FROM `bigquery-public-data.ga.visitors` LIMIT 5",
        "SELECT id FROM `bigquery-public-data.ga.visitors` ORDER BY id LIMIT 5",
        "SELECT COUNT(*) FROM `bigquery-public-data.ga.visitors` LIMIT 5",
        "SELECT id FROM `bigquery-public-data.ga.visitors`",
        "SELECT * EXCEPT (id) FROM `bigquery-public-data.ga.visitors` LIMIT 5",
        "SELECT * REPLACE (1 AS id) FROM `bigquery-public-data.ga.visitors` LIMIT 5",
        "SELECT id, * FROM `bigquery-public-data.ga.visitors` LIMIT 5",
        "SELECT *, id FROM `bigquery-public-data.ga.visitors` LIMIT 5",
        "SELECT id FROM `bigquery-public-data.ga.visitors` FOR SYSTEM_TIME AS OF "
        "TIMESTAMP '2026-01-01 00:00:00+00' LIMIT 5",
        "SELECT id FROM `bigquery-public-data.ga.visitors` "
        "TABLESAMPLE SYSTEM (10 PERCENT) LIMIT 5",
    ],
)
def test_anything_more_than_first_rows_runs_as_a_query(sql: str) -> None:
    database, client = _listing()

    database.execute(sql)

    assert client.listed == []
    assert len(client.statements) == 1


def test_a_slow_listing_spends_the_budget_the_query_would_have_had(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Falling back to a query must not start the timeout over."""
    clock = [100.0]
    monkeypatch.setattr(
        "auto_ontology.connectors.bigquery.time.monotonic", lambda: clock[0]
    )
    database, client = _listing()

    def slow_failure(*args: Any, **kwargs: Any) -> list[_FakeRow]:
        clock[0] += 31
        raise RuntimeError("listing failed")

    client.list_rows = slow_failure  # type: ignore[method-assign]

    with pytest.raises(StatementTimeout):
        database.execute(
            "SELECT * FROM `bigquery-public-data.ga.visitors` LIMIT 5", timeout_s=30
        )

    assert client.statements == []


def test_a_view_runs_as_a_query() -> None:
    database, client = _listing("VIEW")

    database.execute("SELECT * FROM `bigquery-public-data.ga.visitors` LIMIT 5")

    assert client.listed == []
    assert len(client.statements) == 1
