# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every connector honours ``execute(..., timeout_s=...)``.

The text-to-SQL agent caps each statement at the SQL Query Timeout from Agent
Settings, but only on connectors advertising ``supports_statement_timeout``.
SQLite and DuckDB run in-process, so they are exercised against a real slow
query; the network engines are checked against a fake driver for the setting
they send, plus a live Postgres check when a fixture database is reachable.
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

import httpx
import mysql.connector
import pandas as pd
import psycopg
import pytest
import snowflake.connector
from pytest import MonkeyPatch

from auto_ontology.connectors import heavydb as heavydb_module
from auto_ontology.connectors.base import StatementTimeout
from auto_ontology.connectors.clickhouse import ClickHouseDatabase, ClickHouseError
from auto_ontology.connectors.connection_string_factory import build_connection_string
from auto_ontology.connectors.databricks import DatabricksDatabase
from auto_ontology.connectors.duckdb import DuckDBDatabase
from auto_ontology.connectors.heavydb import HeavyDBDatabase
from auto_ontology.connectors.mysql import MySQLDatabase
from auto_ontology.connectors.postgres import PostgresDatabase
from auto_ontology.connectors.registry import CONNECTOR_REGISTRY
from auto_ontology.connectors.snowflake import SnowflakeDatabase
from auto_ontology.connectors.sqlite import SQLiteDatabase
from auto_ontology.connectors.trino import TrinoDatabase

# A query that never finishes on its own: SQLite and DuckDB both evaluate the
# recursive CTE until interrupted.
_ENDLESS_SQL = """
    WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n)
    SELECT count(*) FROM n
"""


def test_every_registered_connector_supports_a_timeout() -> None:
    missing = [
        name
        for name, cls in CONNECTOR_REGISTRY.items()
        if getattr(cls, "supports_statement_timeout", False) is not True
    ]
    assert missing == []


def test_timeout_message_is_not_mistaken_for_a_dead_connection() -> None:
    """A cancelled query should be rewritten, not reported as unreachable."""
    from auto_ontology.connectors.db_errors import is_infrastructure_error

    assert not is_infrastructure_error(str(StatementTimeout(30)), "SELECT 1")


# ----------------------------------------------------------------------
# In-process engines: real slow queries
# ----------------------------------------------------------------------


@pytest.fixture
def sqlite_db(tmp_path: Path) -> Iterator[SQLiteDatabase]:
    path = tmp_path / "t.sqlite"
    sqlite3.connect(path).close()
    database = SQLiteDatabase(str(path))
    yield database
    database.close()


def test_sqlite_statement_is_aborted_at_the_deadline(
    sqlite_db: SQLiteDatabase,
) -> None:
    started = time.monotonic()
    with pytest.raises(StatementTimeout):
        sqlite_db.execute(_ENDLESS_SQL, timeout_s=0.2)
    assert time.monotonic() - started < 5


def test_sqlite_handler_does_not_outlive_the_statement(
    sqlite_db: SQLiteDatabase,
) -> None:
    with pytest.raises(StatementTimeout):
        sqlite_db.execute(_ENDLESS_SQL, timeout_s=0.1)
    time.sleep(0.15)

    # Past the old deadline; a leftover handler would abort this immediately.
    result = sqlite_db.execute(
        "WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n "
        "WHERE i < 200000) SELECT count(*) AS c FROM n"
    )
    assert result["c"].iloc[0] == 200000


def test_sqlite_fast_query_is_unaffected(sqlite_db: SQLiteDatabase) -> None:
    assert sqlite_db.execute("SELECT 1 AS v", timeout_s=5)["v"].iloc[0] == 1


@pytest.fixture
def duckdb_db() -> Iterator[DuckDBDatabase]:
    database = DuckDBDatabase(":memory:", read_only=False)
    yield database
    database.close()


def test_duckdb_statement_is_interrupted_at_the_deadline(
    duckdb_db: DuckDBDatabase,
) -> None:
    started = time.monotonic()
    with pytest.raises(StatementTimeout):
        duckdb_db.execute(_ENDLESS_SQL, timeout_s=0.2)
    assert time.monotonic() - started < 5


def test_duckdb_connection_is_usable_after_a_timeout(
    duckdb_db: DuckDBDatabase,
) -> None:
    with pytest.raises(StatementTimeout):
        duckdb_db.execute(_ENDLESS_SQL, timeout_s=0.1)
    time.sleep(0.15)

    assert duckdb_db.execute("SELECT 42 AS v", timeout_s=5)["v"].iloc[0] == 42


# ----------------------------------------------------------------------
# Postgres
# ----------------------------------------------------------------------


class _PgConnection:
    def __init__(self) -> None:
        self.log: list[Any] = []

    @contextmanager
    def transaction(self) -> Iterator[None]:
        self.log.append("BEGIN")
        yield
        self.log.append("COMMIT")

    def execute(self, sql: str, params: Any = None) -> None:
        self.log.append((sql, params))

    def cursor(self, **kwargs: Any) -> "_PgCursor":
        return _PgCursor(self)


class _PgCursor:
    description = [("v",)]

    def __init__(self, conn: _PgConnection) -> None:
        self._conn = conn

    def __enter__(self) -> "_PgCursor":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def execute(self, sql: str, params: Any = None) -> None:
        self._conn.log.append(("query", sql))

    def fetchall(self) -> list[dict[str, int]]:
        return [{"v": 1}]


class _PgPool:
    def __init__(self) -> None:
        self.conn = _PgConnection()

    @contextmanager
    def connection(self) -> Iterator[_PgConnection]:
        yield self.conn


def _fake_postgres() -> tuple[PostgresDatabase, _PgConnection]:
    database = PostgresDatabase.__new__(PostgresDatabase)
    pool = _PgPool()
    database._pool = pool  # noqa: SLF001 - test double for the pool
    return database, pool.conn


def test_postgres_scopes_the_timeout_to_one_transaction() -> None:
    database, conn = _fake_postgres()

    database.execute("SELECT 1", timeout_s=1.5)

    assert conn.log == [
        "BEGIN",
        ("SELECT set_config('statement_timeout', %s, true)", ("1500ms",)),
        ("query", "SELECT 1"),
        "COMMIT",
    ]


def test_postgres_without_timeout_sends_no_setting() -> None:
    database, conn = _fake_postgres()

    database.execute("SELECT 1")

    assert conn.log == [("query", "SELECT 1")]


def test_live_postgres_cancels_and_does_not_leak_the_cap() -> None:
    if not os.environ.get("POSTGRES_USER"):
        pytest.skip("POSTGRES_* not set")
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    user = os.environ["POSTGRES_USER"]
    password = os.environ["POSTGRES_PASSWORD"]
    try:
        database = PostgresDatabase(
            f"postgresql://{user}:{password}@{host}:{port}/postgres"
        )
    except Exception as exc:  # noqa: BLE001 — any failure means "no fixture DB"
        pytest.skip(f"postgres not reachable: {exc}")
    try:
        with pytest.raises(StatementTimeout):
            database.execute("SELECT pg_sleep(5)", timeout_s=0.3)
        # The pool holds one idle connection; the cap must not have stuck to it.
        setting = database.execute("SHOW statement_timeout").iloc[0, 0]
        assert setting == "0"
    finally:
        database.close()


# ----------------------------------------------------------------------
# MySQL
# ----------------------------------------------------------------------


class _MySQLCursor:
    description = [("v",)]

    def __init__(self, reject: str | None) -> None:
        self.executed: list[tuple[str, Any]] = []
        self._reject = reject

    def execute(self, sql: str, params: Any = None) -> None:
        if self._reject and self._reject in sql:
            import mysql.connector

            raise mysql.connector.Error(msg=f"Unknown system variable '{self._reject}'")
        self.executed.append((sql, params))

    def fetchall(self) -> list[dict[str, int]]:
        return [{"v": 1}]

    def close(self) -> None:
        return None


def _fake_mysql(
    monkeypatch: MonkeyPatch, reject: str | None = None
) -> tuple[MySQLDatabase, _MySQLCursor]:
    cursor = _MySQLCursor(reject)

    class _Conn:
        def cursor(self, **kwargs: Any) -> _MySQLCursor:
            return cursor

        def ping(self, **kwargs: Any) -> None:
            return None

        def close(self) -> None:
            return None

    monkeypatch.setattr(
        "auto_ontology.connectors.mysql.mysql.connector.connect",
        lambda **kwargs: _Conn(),
    )
    return MySQLDatabase("mysql://u:p@db.example.com:3306/beaver"), cursor


def test_mysql_sets_max_execution_time(monkeypatch: MonkeyPatch) -> None:
    database, cursor = _fake_mysql(monkeypatch)

    database.execute("SELECT 1", timeout_s=1.5)

    assert cursor.executed == [
        ("SET SESSION max_execution_time = %s", (1500,)),
        ("SELECT 1", None),
    ]


def test_mariadb_falls_back_to_max_statement_time(monkeypatch: MonkeyPatch) -> None:
    database, cursor = _fake_mysql(monkeypatch, reject="max_execution_time")

    database.execute("SELECT 1", timeout_s=30)

    assert cursor.executed == [
        ("SET SESSION max_statement_time = %s", (30.0,)),
        ("SELECT 1", None),
    ]


def test_mysql_without_timeout_sends_no_setting(monkeypatch: MonkeyPatch) -> None:
    database, cursor = _fake_mysql(monkeypatch)

    database.execute("SELECT 1")

    assert cursor.executed == [("SELECT 1", None)]


# ----------------------------------------------------------------------
# Snowflake
# ----------------------------------------------------------------------


def _fake_snowflake(monkeypatch: MonkeyPatch) -> list[dict[str, Any]]:
    seen: list[dict[str, Any]] = []

    class _Cursor:
        description = [("V",)]

        def __enter__(self) -> "_Cursor":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def execute(self, sql: str, params: Any = None) -> None:
            return None

        def fetchall(self) -> list[tuple[int]]:
            return [(1,)]

    class _Conn:
        def __enter__(self) -> "_Conn":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def cursor(self) -> _Cursor:
            return _Cursor()

    def connect(**kwargs: Any) -> _Conn:
        seen.append(kwargs)
        return _Conn()

    monkeypatch.setattr(
        "auto_ontology.connectors.snowflake.snowflake.connector.connect", connect
    )
    return seen


def test_snowflake_sets_statement_timeout_session_parameter(
    monkeypatch: MonkeyPatch,
) -> None:
    seen = _fake_snowflake(monkeypatch)
    database = SnowflakeDatabase("snowflake://u:p@acct?warehouse=WH&database=DB")

    database.execute("SELECT 1", timeout_s=45)
    database.execute("SELECT 1")

    assert seen[0]["session_parameters"] == {"STATEMENT_TIMEOUT_IN_SECONDS": 45}
    # Per-call kwargs: the cap must not stick to the connector.
    assert "session_parameters" not in seen[1]


# ----------------------------------------------------------------------
# Trino
# ----------------------------------------------------------------------


class _TrinoCursor:
    def __init__(self, conn: "_TrinoConnection") -> None:
        self._conn = conn
        self.description: list[tuple[str, str]] | None = None

    def execute(self, sql: str, parameters: Any = None) -> None:
        self._conn.statements.append(sql)
        if sql in self._conn.fail_on:
            raise RuntimeError(f"failed: {sql}")
        self.description = None if sql.startswith(("SET", "RESET")) else [("v", "")]

    def fetchall(self) -> list[list[int]]:
        return [[1]]

    def close(self) -> None:
        return None


class _TrinoConnection:
    def __init__(self, fail_on: tuple[str, ...] = ()) -> None:
        self.statements: list[str] = []
        self.fail_on = fail_on
        self.closed = False

    def cursor(self) -> _TrinoCursor:
        return _TrinoCursor(self)

    def close(self) -> None:
        self.closed = True


def _trino(fake: _TrinoConnection) -> TrinoDatabase:
    database = TrinoDatabase(
        build_connection_string(
            {"type": "trino", "host": "t.example.com", "user": "a", "database": "hive"}
        )
    )
    database._connection = fake  # noqa: SLF001 - test double for the driver
    return database


def test_trino_sets_and_resets_query_max_run_time() -> None:
    fake = _TrinoConnection()
    database = _trino(fake)

    result = database.execute("SELECT 1", timeout_s=30)

    assert result["v"].tolist() == [1]
    assert fake.statements == [
        "SET SESSION query_max_run_time = '30s'",
        "SELECT 1",
        "RESET SESSION query_max_run_time",
    ]


def test_trino_resets_even_when_the_query_fails() -> None:
    fake = _TrinoConnection(fail_on=("SELECT boom",))
    database = _trino(fake)

    with pytest.raises(RuntimeError, match="SELECT boom"):
        database.execute("SELECT boom", timeout_s=30)

    assert fake.statements[-1] == "RESET SESSION query_max_run_time"


def test_trino_drops_the_connection_if_the_reset_fails() -> None:
    """Otherwise the cap would leak onto later, uncapped metadata scans."""
    fake = _TrinoConnection(fail_on=("RESET SESSION query_max_run_time",))
    database = _trino(fake)

    database.execute("SELECT 1", timeout_s=30)

    assert fake.closed
    assert database._connection is None  # noqa: SLF001


# ----------------------------------------------------------------------
# HeavyDB
# ----------------------------------------------------------------------


class _HeavyClient:
    def __init__(self) -> None:
        self.interrupts: list[tuple[str, str]] = []

    def interrupt(self, query_session: str, interrupt_session: str) -> None:
        self.interrupts.append((query_session, interrupt_session))


def _fake_heavydb(
    monkeypatch: MonkeyPatch, query_seconds: float
) -> tuple[HeavyDBDatabase, _HeavyClient]:
    client = _HeavyClient()
    sessions = iter(f"s{i}" for i in range(100))
    release = threading.Event()

    class _Cursor:
        description = [("v",)]

        def __enter__(self) -> "_Cursor":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def execute(self, sql: str, params: Any = None) -> None:
            release.wait(query_seconds)

        def fetchall(self) -> list[tuple[int]]:
            return [(1,)]

    class _Conn:
        def __init__(self) -> None:
            self._session = next(sessions)
            self._client = client

        def cursor(self) -> _Cursor:
            return _Cursor()

        def close(self) -> None:
            return None

    monkeypatch.setattr(heavydb_module, "_connect_with_timeout", lambda *args: _Conn())
    database = HeavyDBDatabase("heavydb://admin:pw@h.example.com:6274/heavyai")
    return database, client


def test_heavydb_stops_waiting_and_interrupts_the_query(
    monkeypatch: MonkeyPatch,
) -> None:
    database, client = _fake_heavydb(monkeypatch, query_seconds=5)

    started = time.monotonic()
    with pytest.raises(StatementTimeout):
        database.execute("SELECT 1", timeout_s=0.2)

    assert time.monotonic() - started < 2
    assert client.interrupts == [("s0", "s1")]


def test_heavydb_fast_query_returns_normally(monkeypatch: MonkeyPatch) -> None:
    database, client = _fake_heavydb(monkeypatch, query_seconds=0)

    result = database.execute("SELECT 1", timeout_s=5)

    assert result.equals(pd.DataFrame([(1,)], columns=["v"]))
    assert client.interrupts == []


# ----------------------------------------------------------------------
# Every engine's cancellation surfaces as StatementTimeout
# ----------------------------------------------------------------------
#
# Callers recognise a timeout by type: semantic compilation's sampling circuit
# breaker counts ``TimeoutError`` and no server-reported cancellation message,
# so an engine whose cancellation stayed a driver error would never trip it.


def _raise(error: BaseException) -> Any:
    raise error


def test_sampling_breaker_counts_a_statement_timeout() -> None:
    from auto_ontology.semantic.visit_enter import _SamplingCircuitBreaker

    breaker = _SamplingCircuitBreaker()
    connector = SimpleNamespace(database_name="warehouse")

    assert breaker.record_failure(connector, StatementTimeout(120)) is True


def _raising_postgres(error: BaseException) -> PostgresDatabase:
    database, conn = _fake_postgres()

    class _Cursor(_PgCursor):
        def execute(self, sql: str, params: Any = None) -> None:
            raise error

    conn.cursor = lambda **kwargs: _Cursor(conn)  # type: ignore[method-assign]
    return database


def test_postgres_statement_timeout_becomes_statement_timeout() -> None:
    database = _raising_postgres(
        psycopg.errors.QueryCanceled("canceling statement due to statement timeout")
    )

    with pytest.raises(StatementTimeout) as caught:
        database.execute("SELECT 1", timeout_s=30)
    assert caught.value.timeout_s == 30


def test_postgres_other_cancellation_is_left_alone() -> None:
    """pg_cancel_backend() is an operator's decision, not our cap."""
    database = _raising_postgres(
        psycopg.errors.QueryCanceled("canceling statement due to user request")
    )

    with pytest.raises(psycopg.errors.QueryCanceled):
        database.execute("SELECT 1", timeout_s=30)


def _mysql_failing_with(monkeypatch: MonkeyPatch, errno: int) -> MySQLDatabase:
    database, cursor = _fake_mysql(monkeypatch)
    error = mysql.connector.Error(msg="statement failed", errno=errno)
    monkeypatch.setattr(
        cursor,
        "execute",
        lambda sql, params=None: _raise(error) if sql.startswith("SELECT") else None,
    )
    return database


@pytest.mark.parametrize("errno", [3024, 1969])
def test_mysql_and_mariadb_timeouts_become_statement_timeout(
    monkeypatch: MonkeyPatch, errno: int
) -> None:
    database = _mysql_failing_with(monkeypatch, errno)

    with pytest.raises(StatementTimeout):
        database.execute("SELECT 1", timeout_s=30)


def test_mysql_other_errors_are_left_alone(monkeypatch: MonkeyPatch) -> None:
    database = _mysql_failing_with(monkeypatch, 1054)

    with pytest.raises(mysql.connector.Error):
        database.execute("SELECT 1", timeout_s=30)


@pytest.mark.parametrize(("errno", "expected"), [(630, StatementTimeout), (2003, None)])
def test_snowflake_only_its_timeout_becomes_statement_timeout(
    monkeypatch: MonkeyPatch, errno: int, expected: type | None
) -> None:
    database = SnowflakeDatabase("snowflake://u:p@acct?warehouse=WH&database=DB")
    error = snowflake.connector.errors.ProgrammingError(msg="failed", errno=errno)
    monkeypatch.setattr(database, "_execute", lambda *args: _raise(error))

    with pytest.raises(expected or snowflake.connector.errors.ProgrammingError):
        database.execute("SELECT 1", timeout_s=30)


def test_trino_time_limit_becomes_statement_timeout() -> None:
    class _TimeLimit(Exception):
        error_name = "EXCEEDED_TIME_LIMIT"

    class _Cursor(_TrinoCursor):
        def execute(self, sql: str, parameters: Any = None) -> None:
            super().execute(sql, parameters)
            if sql == "SELECT slow":
                raise _TimeLimit("Query exceeded maximum time limit of 30.00s")

    fake = _TrinoConnection()
    fake.cursor = lambda: _Cursor(fake)  # type: ignore[method-assign]
    database = _trino(fake)

    with pytest.raises(StatementTimeout):
        database.execute("SELECT slow", timeout_s=30)
    assert fake.statements[-1] == "RESET SESSION query_max_run_time"


def _clickhouse_answering(response: httpx.Response) -> ClickHouseDatabase:
    database = ClickHouseDatabase("clickhouse://u@ch.example.com:8123/db")
    database._client = httpx.Client(  # noqa: SLF001 - test double for the server
        base_url="http://test",
        transport=httpx.MockTransport(lambda request: response),
    )
    return database


@pytest.mark.parametrize(
    ("headers", "body"),
    [
        ({"X-ClickHouse-Exception-Code": "159"}, "Code: 159. DB::Exception: x"),
        ({}, "Code: 159. DB::Exception: Timeout exceeded. (TIMEOUT_EXCEEDED)"),
    ],
)
def test_clickhouse_timeout_becomes_statement_timeout(
    headers: dict[str, str], body: str
) -> None:
    database = _clickhouse_answering(httpx.Response(500, headers=headers, text=body))

    with pytest.raises(StatementTimeout):
        database.execute("SELECT 1", timeout_s=30)


def test_clickhouse_other_errors_stay_clickhouse_errors() -> None:
    database = _clickhouse_answering(
        httpx.Response(
            404,
            headers={"X-ClickHouse-Exception-Code": "60"},
            text="Code: 60. DB::Exception: Table does not exist. (UNKNOWN_TABLE)",
        )
    )

    with pytest.raises(ClickHouseError):
        database.execute("SELECT 1", timeout_s=30)


def test_databricks_timeout_becomes_statement_timeout(
    monkeypatch: MonkeyPatch,
) -> None:
    error = RuntimeError(
        "[QUERY_EXECUTION_TIMEOUT_EXCEEDED] Query execution was cancelled due to "
        "exceeding the timeout (30s). SQLSTATE: 57KD0"
    )

    class _Cursor:
        def __enter__(self) -> "_Cursor":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def execute(self, sql: str, params: Any = None) -> None:
            raise error

    class _Conn:
        def cursor(self) -> _Cursor:
            return _Cursor()

        def close(self) -> None:
            return None

    monkeypatch.setattr(
        "auto_ontology.connectors.databricks.sql.connect", lambda **kwargs: _Conn()
    )
    database = DatabricksDatabase(
        "databricks://token:t@example.databricks.com/main"
        "?http_path=%2Fsql%2F1.0%2Fwarehouses%2Fw"
    )

    with pytest.raises(StatementTimeout):
        database.execute("SELECT 1", timeout_s=30)
