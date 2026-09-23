# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import json
import logging
from contextlib import contextmanager
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import pandas as pd
import pytest
from TCLIService.ttypes import TOperationState

from auto_ontology.catalog.extract import IncompleteCatalogExtractionError
from auto_ontology.connectors.connection_string_factory import build_connection_string
from auto_ontology.connectors.kyuubi import (
    KyuubiDatabase,
    KyuubiQueryTimeout,
    _parse_connection_string,
)
from auto_ontology.connectors.registry import CONNECTOR_REGISTRY

HOST = "hive.pdx-aws.data.nvidia.com"
SSA_URL = "https://svc.ssa.nvidia.com"


def _connection(**overrides: object) -> dict[str, object]:
    connection = {
        "type": "kyuubi",
        "host": HOST,
        "user": "nvssa-prd-abc",
        "password": "secret",
        "database": "nvdp",
        "ssa_url": SSA_URL,
    }
    connection.update(overrides)
    return connection


def _database(**overrides: object) -> KyuubiDatabase:
    return KyuubiDatabase(build_connection_string(_connection(**overrides)))


# ----------------------------------------------------------------------
# Connection string
# ----------------------------------------------------------------------


def test_kyuubi_is_registered() -> None:
    assert CONNECTOR_REGISTRY["kyuubi"] is KyuubiDatabase


def test_default_port_is_applied() -> None:
    assert urlparse(build_connection_string(_connection())).port == 10000


def test_credentials_with_reserved_characters_survive_the_round_trip() -> None:
    """A client secret containing ``/`` or ``@`` must not corrupt the URL."""
    built = build_connection_string(_connection(password="se/cret@1"))

    assert _parse_connection_string(built)["secret"] == "se/cret@1"


def test_ssa_url_selects_service_account_auth() -> None:
    settings = _parse_connection_string(build_connection_string(_connection()))

    assert settings["auth"] == "ssa"
    assert settings["ssa_url"] == SSA_URL


def test_without_ssa_url_the_password_is_treated_as_a_static_token() -> None:
    """Nothing can refresh a pasted JWT, so it must not be taken for a secret."""
    built = build_connection_string(_connection(ssa_url=""))

    assert parse_qs(urlparse(built).query)["auth"] == ["token"]
    assert _parse_connection_string(built)["auth"] == "token"


def test_scheme_url_host_is_normalised() -> None:
    built = build_connection_string(_connection(host=f"https://{HOST}"))

    assert urlparse(built).hostname == HOST


def test_truststore_is_carried_through() -> None:
    built = build_connection_string(
        _connection(truststore="/etc/ssl/ca.jks", truststore_password="pw")
    )

    settings = _parse_connection_string(built)
    assert settings["truststore"] == "/etc/ssl/ca.jks"
    assert settings["truststore_password"] == "pw"


def test_ssa_auth_requires_a_token_url() -> None:
    """auth=ssa without an endpoint would mint nothing and hang at connect time."""
    with pytest.raises(ValueError, match="ssa_url"):
        _parse_connection_string(f"kyuubi://user:secret@{HOST}:10000/nvdp?auth=ssa")


def test_catalog_is_required() -> None:
    with pytest.raises(ValueError, match="catalog"):
        _parse_connection_string(f"kyuubi://user:secret@{HOST}:10000?auth=token")


def test_non_kyuubi_url_is_rejected() -> None:
    with pytest.raises(ValueError, match="Not a Kyuubi URL"):
        _parse_connection_string("postgresql://user:pw@host:5432/db")


# ----------------------------------------------------------------------
# Identity
# ----------------------------------------------------------------------


def test_dialect_is_spark() -> None:
    """Kyuubi runs Spark SQL; the agent must not be told to emit Hive SQL."""
    assert _database().dialect == "spark"


def test_database_name_is_the_catalog() -> None:
    """NeMo routes SQL by database_name, so it has to be the catalog."""
    assert _database(database="kratos").database_name == "kratos"


# ----------------------------------------------------------------------
# Token refresh
# ----------------------------------------------------------------------


class _FakeTokenResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode()

    def __enter__(self) -> "_FakeTokenResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None


def test_ssa_token_is_reused_until_it_nears_expiry(monkeypatch) -> None:
    calls = []

    def fake_urlopen(request, timeout=None):
        calls.append(request.full_url)
        return _FakeTokenResponse(
            {"access_token": f"tok{len(calls)}", "expires_in": 3600}
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    db = _database()

    assert db._password() == "tok1"
    assert db._password() == "tok1"
    assert len(calls) == 1
    assert calls[0].startswith(f"{SSA_URL}/token")
    assert "grant_type=client_credentials" in calls[0]


def test_expiring_ssa_token_is_refreshed(monkeypatch) -> None:
    """A cached connector outlives an hourly token, so it must mint a new one."""
    calls = []

    def fake_urlopen(request, timeout=None):
        calls.append(request.full_url)
        # Already inside the refresh margin, so the next call must re-mint.
        return _FakeTokenResponse(
            {"access_token": f"tok{len(calls)}", "expires_in": 10}
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    db = _database()

    assert db._password() == "tok1"
    assert db._password() == "tok2"


def test_static_token_mode_never_calls_the_token_endpoint(monkeypatch) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("auth=token must not contact the SSA endpoint")

    monkeypatch.setattr("urllib.request.urlopen", fail)
    db = _database(ssa_url="", password="a.jwt.value")

    assert db._password() == "a.jwt.value"


# ----------------------------------------------------------------------
# Introspection
# ----------------------------------------------------------------------


def _stub_execute(db: KyuubiDatabase, responses: dict[str, pd.DataFrame]) -> list[str]:
    """Route execute() to canned frames, recording the SQL that was run."""
    issued: list[str] = []

    def fake_execute(sql: str, parameters=None) -> pd.DataFrame:
        issued.append(sql)
        for fragment, frame in responses.items():
            if fragment in sql:
                return frame
        return pd.DataFrame()

    db.execute = fake_execute  # type: ignore[method-assign]
    return issued


DESCRIBED = pd.DataFrame({"col_name": ["id"], "data_type": ["bigint"]})


def test_get_tables_marks_views(monkeypatch) -> None:
    db = _database()
    _stub_execute(
        db,
        {
            "SHOW DATABASES": pd.DataFrame({"namespace": ["raw"]}),
            "SHOW TABLES": pd.DataFrame(
                {"namespace": ["raw", "raw"], "tableName": ["events", "events_v"]}
            ),
            "SHOW VIEWS": pd.DataFrame(
                {"namespace": ["raw"], "viewName": ["events_v"]}
            ),
            "DESCRIBE TABLE": DESCRIBED,
        },
    )

    tables = db.get_tables()

    assert dict(zip(tables["table_name"], tables["table_type"])) == {
        "events": "BASE TABLE",
        "events_v": "VIEW",
    }


def _stub_with_failing_describe(db: KyuubiDatabase, failing: set[str]) -> list[str]:
    """Stub execute() so DESCRIBE raises for the named tables."""
    issued: list[str] = []

    def fake_execute(sql: str, parameters=None) -> pd.DataFrame:
        issued.append(sql)
        if sql.startswith("DESCRIBE TABLE"):
            if any(f"`{name}`" in sql for name in failing):
                raise RuntimeError("NessieForbiddenException: Forbidden (HTTP/403)")
            return DESCRIBED
        if "SHOW DATABASES" in sql:
            return pd.DataFrame({"namespace": ["kpi", "system"]})
        if "SHOW TABLES" in sql:
            table = "events" if "`kpi`" in sql else "table_creation_locks"
            return pd.DataFrame({"namespace": ["x"], "tableName": [table]})
        return pd.DataFrame()

    db.execute = fake_execute  # type: ignore[method-assign]
    return issued


def test_undescribable_table_is_dropped_from_tables_too() -> None:
    """A table with no columns must not be advertised in the tables frame.

    Ingestion groups columns by the schemas found in the tables frame, so a
    schema listing a table it has no columns for reaches the graph writer with
    an empty column set and aborts the entire run.
    """
    db = _database()
    _stub_with_failing_describe(db, {"table_creation_locks"})

    tables = db.get_tables()
    columns = db.get_columns()

    assert set(zip(tables["table_schema"], tables["table_name"])) == {("kpi", "events")}
    assert set(tables["table_schema"]) == set(columns["table_schema"])
    assert "system" not in set(tables["table_schema"])


def test_describing_nothing_at_all_raises_instead_of_emptying_the_catalog() -> None:
    """Dropping one unreadable table is a skip; dropping every one is a failure.

    A dead session or a revoked catalog grant fails every DESCRIBE alike. Silently
    dropping the lot yields an empty catalog that is indistinguishable from a
    database with nothing in it.
    """
    db = _database()
    _stub_with_failing_describe(db, {"events", "table_creation_locks"})

    with pytest.raises(IncompleteCatalogExtractionError, match="could not describe"):
        db.get_tables()


def test_every_reported_schema_has_columns() -> None:
    """The invariant nemo's schemas_parser relies on, stated directly."""
    db = _database()
    _stub_with_failing_describe(db, {"table_creation_locks"})

    tables = db.get_tables()
    columns = db.get_columns()

    for schema in set(tables["table_schema"]):
        assert not columns[columns["table_schema"] == schema].empty


def test_introspection_is_cached_across_getters() -> None:
    """DESCRIBE is a Spark round trip per table; sweeping twice doubles ingest."""
    db = _database()
    issued = _stub_with_failing_describe(db, set())

    db.get_tables()
    db.get_columns()
    db.get_views()

    assert sum(1 for sql in issued if sql.startswith("DESCRIBE TABLE")) == 2


def test_schema_allowlist_limits_introspection() -> None:
    db = KyuubiDatabase(build_connection_string(_connection()), schemas=["Raw"])
    issued = _stub_execute(
        db,
        {
            "SHOW DATABASES": pd.DataFrame({"namespace": ["raw", "kpi"]}),
            "SHOW TABLES": pd.DataFrame({"namespace": ["raw"], "tableName": ["t"]}),
            "SHOW VIEWS": pd.DataFrame(),
            "DESCRIBE TABLE": DESCRIBED,
        },
    )

    tables = db.get_tables()

    # Spark lower-cases unquoted identifiers, so the filter is case-insensitive.
    assert set(tables["table_schema"]) == {"raw"}
    assert not any("`kpi`" in sql for sql in issued)


def test_get_columns_stops_at_the_metadata_section() -> None:
    """DESCRIBE appends partition/table sections that are not columns."""
    db = _database()
    _stub_execute(
        db,
        {
            "SHOW DATABASES": pd.DataFrame({"namespace": ["raw"]}),
            "SHOW TABLES": pd.DataFrame({"namespace": ["raw"], "tableName": ["t"]}),
            "SHOW VIEWS": pd.DataFrame(),
            "DESCRIBE TABLE": pd.DataFrame(
                {
                    "col_name": ["id", "ts", "", "# Partition Information", "ts"],
                    "data_type": ["bigint", "timestamp", "", "", "timestamp"],
                }
            ),
        },
    )

    columns = db.get_columns()

    assert list(columns["column_name"]) == ["id", "ts"]
    assert list(columns["ordinal_position"]) == [1, 2]


def test_introspection_returns_typed_empty_frames() -> None:
    """Empty results still need their columns, or ingestion raises on access."""
    db = _database()
    _stub_execute(db, {"SHOW DATABASES": pd.DataFrame()})

    assert list(db.get_columns().columns) == [
        "table_schema",
        "table_name",
        "column_name",
        "data_type",
        "is_nullable",
        "ordinal_position",
    ]
    assert list(db.get_pks().columns) == [
        "table_schema",
        "table_name",
        "column_name",
        "ordinal_position",
    ]
    assert list(db.get_fks().columns) == [
        "table_schema",
        "table_name",
        "column_name",
        "referenced_schema",
        "referenced_table",
        "referenced_column",
    ]
    assert list(db.get_queries().columns) == ["end_time", "query_text"]


# ----------------------------------------------------------------------
# Session recovery
# ----------------------------------------------------------------------


def test_execute_reopens_after_a_broken_pipe() -> None:
    """Kyuubi retires idle engines, so a cached connector meets a dead socket.

    pyhive's ``cursor()`` builds a local object without touching the network, so
    the loss only surfaces when a statement is sent -- the retry has to live in
    execute(), not around cursor creation.
    """
    db = _database()
    calls: list[str] = []
    opened: list[int] = []

    def fake_execute_once(sql: str, parameters=None, **_kwargs) -> pd.DataFrame:
        calls.append(sql)
        if len(calls) == 1:
            raise BrokenPipeError(32, "Broken pipe")
        return pd.DataFrame({"ok": [1]})

    def fake_reset() -> None:
        opened.append(1)

    db._execute_once = fake_execute_once  # type: ignore[method-assign]
    db._reset_connection = fake_reset  # type: ignore[method-assign]

    frame = db.execute("SELECT 1")

    assert frame.to_dict("records") == [{"ok": 1}]
    assert len(calls) == 2, "the statement should be retried once"
    assert opened == [1], "the dead session should be dropped before retrying"


def test_execute_reopens_after_an_invalid_session_handle() -> None:
    """A retired engine usually leaves the socket up and the handle unknown.

    Kyuubi answers the next statement normally, reporting ``Invalid
    SessionHandle`` for a session it no longer has. That is an ordinary driver
    error on a healthy connection, so recognising it by exception type is not
    enough. Nothing else clears the cached connection, so missing this case
    strands the connector on a dead handle for every later query.
    """
    db = _database()
    calls: list[str] = []
    opened: list[int] = []

    def fake_execute_once(sql: str, parameters=None, **_kwargs) -> pd.DataFrame:
        calls.append(sql)
        if len(calls) == 1:
            raise RuntimeError(
                "TExecuteStatementResp(status=TStatus(statusCode=3, "
                "errorMessage='Invalid SessionHandle: "
                "1f9c2c1e-0000-4c1e-9f00-2b0d5a7c9e11'))"
            )
        return pd.DataFrame({"ok": [1]})

    db._execute_once = fake_execute_once  # type: ignore[method-assign]
    db._reset_connection = lambda: opened.append(1)  # type: ignore[method-assign]

    frame = db.execute("SELECT 1")

    assert frame.to_dict("records") == [{"ok": 1}]
    assert len(calls) == 2, "the statement should be retried once"
    assert opened == [1], "the stale session should be dropped before retrying"


def test_execute_does_not_retry_a_rejected_statement() -> None:
    """A SQL error is the server answering, not the transport failing."""
    db = _database()
    calls: list[str] = []

    def fake_execute_once(sql: str, parameters=None, **_kwargs) -> pd.DataFrame:
        calls.append(sql)
        raise ValueError("Table or view not found: nope")

    db._execute_once = fake_execute_once  # type: ignore[method-assign]

    with pytest.raises(ValueError, match="not found"):
        db.execute("SELECT * FROM nope")

    assert len(calls) == 1, "a SQL error must surface, not trigger a reconnect"


def test_execute_surfaces_a_session_error_that_survives_the_retry() -> None:
    """If reopening does not help, the error is reported rather than looped on."""
    db = _database()
    calls: list[str] = []

    def fake_execute_once(sql: str, parameters=None, **_kwargs) -> pd.DataFrame:
        calls.append(sql)
        raise RuntimeError("Invalid SessionHandle: deadbeef")

    db._execute_once = fake_execute_once  # type: ignore[method-assign]
    db._reset_connection = lambda: None  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="Invalid SessionHandle"):
        db.execute("SELECT 1")

    assert len(calls) == 2, "exactly one retry, then give up"


# ----------------------------------------------------------------------
# Uploaded truststore
# ----------------------------------------------------------------------


def test_uploaded_truststore_travels_as_data_not_a_path() -> None:
    """A path would have to exist on every pod; the bytes ride along instead."""
    encoded = base64.b64encode(b"fake-keystore-bytes").decode()
    built = build_connection_string(
        _connection(truststore_file=encoded, truststore_password="pw")
    )

    settings = _parse_connection_string(built)
    assert settings["truststore_data"] == encoded
    assert settings["truststore"] is None
    assert settings["truststore_password"] == "pw"


def test_uploaded_truststore_wins_over_a_path() -> None:
    built = build_connection_string(
        _connection(truststore_file="QUJD", truststore="/etc/ssl/ca.jks")
    )

    settings = _parse_connection_string(built)
    assert settings["truststore_data"] == "QUJD"
    assert settings["truststore"] is None


def test_keystore_bytes_are_kept_out_of_the_logs() -> None:
    """The blob is hundreds of KB; logging it verbatim on failure is unusable."""
    from auto_ontology.connectors.registry import _redact

    built = build_connection_string(_connection(truststore_file="A" * 5000))

    redacted = _redact(built)
    assert "truststore_data=<keystore>" in redacted
    assert "A" * 100 not in redacted


def test_invalid_uploaded_keystore_is_rejected_clearly() -> None:
    from auto_ontology.connectors.kyuubi import _write_temp_jks

    with pytest.raises(ValueError, match="not valid base64"):
        _write_temp_jks("not!base64!")


# ----------------------------------------------------------------------
# Qualification
# ----------------------------------------------------------------------


def test_qualify_prepends_the_bound_catalog() -> None:
    # Spark resolves a bare schema.table against spark_catalog, not the Iceberg
    # catalog this connection binds, so dropping the catalog made every
    # profiling probe fail with TABLE_OR_VIEW_NOT_FOUND.
    assert _database().qualify("raw", "events") == "`nvdp`.`raw`.`events`"


def test_qualify_rejects_a_missing_schema() -> None:
    # `nvdp`.`events` would not mean "catalog nvdp": Spark reads a two-part
    # name as schema.table, so the catalog would land in the schema position.
    with pytest.raises(ValueError, match="requires a schema"):
        _database().qualify(None, "events")


def test_qualify_quotes_names_with_spaces() -> None:
    assert _database().qualify("my schema", "my table") == (
        "`nvdp`.`my schema`.`my table`"
    )


SQL = "SELECT *\n  FROM `nvdp`.`raw`.`events` LIMIT 1000"


class _FakeCursor:
    """A pyhive-shaped cursor whose operation state the test drives.

    ``states`` is consumed one entry per ``poll()``; the last one repeats, so a
    single RUNNING entry stands in for a statement that never finishes.
    """

    description = [("events.id", "bigint")]

    def __init__(self, states: list[int]) -> None:
        self._states = states
        self.executed_async: bool | None = None
        self.cancelled = False
        self.polls = 0

    def execute(self, sql: str, parameters=None, async_: bool = False) -> None:
        self.executed_async = async_

    def poll(self):
        state = self._states[min(self.polls, len(self._states) - 1)]
        self.polls += 1
        return SimpleNamespace(operationState=state, errorMessage="boom")

    def cancel(self) -> None:
        self.cancelled = True

    def fetchall(self) -> list[tuple[int]]:
        return [(1,), (2,)]


def _stub_cursor(db: KyuubiDatabase, states: list[int]) -> _FakeCursor:
    cursor = _FakeCursor(states)

    @contextmanager
    def fake_cursor():
        yield cursor

    db._cursor = fake_cursor  # type: ignore[method-assign]
    return cursor


def test_execute_logs_the_query_and_its_phase_timings(caplog) -> None:
    """A statement is silent until it returns, so time it like Databricks does.

    Without this a 900s read timeout and a healthy query look identical in the
    logs, and the queueing behind the session lock is invisible.
    """
    db = _database()
    cursor = _stub_cursor(db, [TOperationState.FINISHED_STATE])

    with caplog.at_level(logging.INFO, logger="auto_ontology.connectors.kyuubi"):
        frame = db.execute(SQL)

    assert list(frame.columns) == ["id"]
    assert cursor.executed_async is True, "a blocking call cannot report its state"
    message = next(m for m in caplog.messages if m.startswith("kyuubi: "))
    assert "2 row(s)" in message
    for phase in ("wait ", "connect ", "execute ", "fetch "):
        assert phase in message
    assert "SELECT * FROM `nvdp`.`raw`.`events` LIMIT 1000" in message


def test_execute_cancels_a_statement_that_outlives_its_timeout() -> None:
    """A capped query fails fast instead of riding the 900s socket timeout.

    The engine never gets past RUNNING here, which is what a Spark job that
    cannot get executors looks like from the client.
    """
    db = _database()
    cursor = _stub_cursor(db, [TOperationState.RUNNING_STATE])
    reopened: list[int] = []
    db._reset_connection = lambda: reopened.append(1)  # type: ignore[method-assign]

    with pytest.raises(KyuubiQueryTimeout, match="exceeded 1s"):
        db.execute(SQL, timeout_s=1)

    assert cursor.cancelled, "the server should be told to stop the statement"
    assert not reopened, "a timeout is not a lost session — retrying just re-waits"


def test_execute_reports_a_statement_the_engine_failed() -> None:
    """poll() reports ERROR_STATE without raising; fetching would mask it."""
    db = _database()
    _stub_cursor(db, [TOperationState.ERROR_STATE])

    with pytest.raises(RuntimeError, match="ERROR_STATE: boom"):
        db.execute(SQL)


def test_a_pending_statement_reports_its_state_while_it_waits(
    caplog, monkeypatch
) -> None:
    """PENDING vs RUNNING is the difference between queued and scanning."""
    # The state line is only worth logging once a statement is slow, so pretend
    # this one already is.
    monkeypatch.setattr("auto_ontology.connectors.kyuubi._SLOW_QUERY_SECONDS", 0.0)
    db = _database()
    _stub_cursor(db, [TOperationState.PENDING_STATE, TOperationState.FINISHED_STATE])

    with caplog.at_level(logging.INFO, logger="auto_ontology.connectors.kyuubi"):
        db.execute(SQL)

    assert any("PENDING_STATE after" in m for m in caplog.messages)


def test_an_uncapped_statement_still_has_a_ceiling(monkeypatch) -> None:
    """Polling removed the implicit bound the blocking socket read gave us.

    Without a default ceiling an engine parked in PENDING spins here forever,
    holding the session lock and wedging every other caller with it.
    """
    monkeypatch.setattr("auto_ontology.connectors.kyuubi._SOCKET_TIMEOUT_SECONDS", 1)
    db = _database()
    cursor = _stub_cursor(db, [TOperationState.PENDING_STATE])

    with pytest.raises(KyuubiQueryTimeout, match="exceeded 1s"):
        db.execute(SQL)

    assert cursor.cancelled


def test_a_failed_statement_reports_what_the_server_said() -> None:
    """is_session_lost and the repair loop classify by message text.

    A bare state name matches nothing, so a session lost mid-statement would be
    raised instead of retried.
    """
    db = _database()

    class _Failed(_FakeCursor):
        def poll(self):
            return SimpleNamespace(
                operationState=TOperationState.ERROR_STATE,
                errorMessage="Invalid SessionHandle: deadbeef",
                sqlState="08S01",
                infoMessages=["org.apache.kyuubi.KyuubiSQLException"],
            )

    cursor = _Failed([TOperationState.ERROR_STATE])

    @contextmanager
    def fake_cursor():
        yield cursor

    db._cursor = fake_cursor  # type: ignore[method-assign]
    db._reset_connection = lambda: None  # type: ignore[method-assign]

    with pytest.raises(RuntimeError) as excinfo:
        db.execute(SQL)

    message = str(excinfo.value)
    assert "Invalid SessionHandle" in message
    assert "08S01" in message
    assert "KyuubiSQLException" in message
    # The statement is left out on purpose: callers match markers against this
    # text, and an echoed identifier could match one.
    assert "elk_log" not in message and "SELECT" not in message


def test_a_failure_with_no_detail_says_so() -> None:
    """Falling back to the SQL would put an identifier where markers are matched."""
    db = _database()

    class _Bare(_FakeCursor):
        def poll(self):
            return SimpleNamespace(
                operationState=TOperationState.CANCELED_STATE,
                errorMessage=None,
                sqlState=None,
                infoMessages=None,
            )

    cursor = _Bare([TOperationState.CANCELED_STATE])

    @contextmanager
    def fake_cursor():
        yield cursor

    db._cursor = fake_cursor  # type: ignore[method-assign]
    db._reset_connection = lambda: None  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="no error detail reported"):
        db.execute(SQL)
