# SPDX-FileCopyrightText: Copyright (c) 2026 Mario Nascimento
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from typing import Any

import oracledb
import pytest
from pytest import MonkeyPatch
from sqlglot import exp

from auto_ontology.connectors.base import StatementTimeout
from auto_ontology.connectors.connection_string_factory import build_connection_string
from auto_ontology.connectors.oracle import OracleDatabase, _parse_connection_string
from auto_ontology.connectors.registry import create_connector
from auto_ontology.retrieval.text_to_sql.prompts import format_dialect_rules
from auto_ontology.semantic.visit_enter import probe_ascending_distinct_values

_CONNECTION_STRING = "oracle://scott:tiger@localhost:1521/FREEPDB1"


def _oracle_error(code: str) -> oracledb.DatabaseError:
    """A driver error carrying *code* the way python-oracledb reports it."""
    return oracledb.DatabaseError(SimpleNamespace(full_code=code, message=code))


class Cursor:
    """Answers each statement from *results*, a list of ``(columns, rows)``.

    An exception in place of a result is raised instead.
    """

    arraysize = 100

    def __init__(self, connection: "Connection") -> None:
        self._connection = connection
        self.description: list[tuple[str]] | None = None
        self._rows: list[tuple[Any, ...]] = []
        self.closed = False

    def execute(self, statement: str, parameters: list[Any] | None = None) -> None:
        self._connection.executed.append((statement, parameters))
        result = self._connection.next_result()
        if isinstance(result, BaseException):
            raise result
        columns, rows = result
        self.description = [(name,) for name in columns]
        self._rows = rows

    def fetchall(self) -> list[tuple[Any, ...]]:
        return self._rows

    def close(self) -> None:
        self.closed = True


class Connection:
    def __init__(self, results: list[Any], connect_kwargs: dict[str, Any]) -> None:
        self._results = results
        self.connect_kwargs = connect_kwargs
        self.executed: list[tuple[str, list[Any] | None]] = []
        self.call_timeout: int | None = None
        self.outputtypehandler: Any = None
        self.pinged = False
        self.closed = False

    def next_result(self) -> Any:
        if self._results:
            return self._results.pop(0)
        return (["VALUE"], [(1,)])

    def cursor(self) -> Cursor:
        return Cursor(self)

    def ping(self) -> None:
        self.pinged = True

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def driver(monkeypatch: MonkeyPatch) -> SimpleNamespace:
    """Replace ``oracledb.connect``; exposes the connections it handed out.

    Results queued on ``driver.results`` are consumed by statements in order,
    across connections, since the connector opens one per call.
    """
    state = SimpleNamespace(connections=[], results=[])

    def connect(**kwargs: Any) -> Connection:
        connection = Connection(state.results, kwargs)
        state.connections.append(connection)
        return connection

    monkeypatch.setattr("auto_ontology.connectors.oracle.oracledb.connect", connect)
    return state


def test_parse_connection_string() -> None:
    kwargs = _parse_connection_string(
        "oracle://user%40domain:secret%2Fvalue@example.com:1522/sales.example.com"
    )

    assert kwargs == {
        "host": "example.com",
        "port": 1522,
        "user": "user@domain",
        "password": "secret/value",
        "service_name": "sales.example.com",
    }


def test_parse_connection_string_defaults_the_port() -> None:
    assert _parse_connection_string("oracle://u:p@db/XEPDB1")["port"] == 1521


@pytest.mark.parametrize(
    "connection_string",
    [
        "mysql://u:p@localhost/db",
        "oracle://u:p@/FREEPDB1",
        "oracle://u:p@localhost:1521",
    ],
)
def test_parse_connection_string_rejects_incomplete_urls(
    connection_string: str,
) -> None:
    with pytest.raises(ValueError):
        _parse_connection_string(connection_string)


def test_parse_error_does_not_leak_the_password() -> None:
    with pytest.raises(ValueError) as raised:
        _parse_connection_string("mysql://user:hunter2@localhost/db")

    assert "hunter2" not in str(raised.value)


def test_build_connection_string() -> None:
    connection_string = build_connection_string(
        {
            "type": "oracle",
            "host": "localhost",
            "port": "1522",
            "user": "user@domain",
            "password": "secret/value",
            "database": "FREEPDB1",
        }
    )

    assert (
        connection_string
        == "oracle://user%40domain:secret%2Fvalue@localhost:1522/FREEPDB1"
    )


def test_build_connection_string_defaults_the_port() -> None:
    connection_string = build_connection_string(
        {
            "type": "oracle",
            "host": "db.example.com",
            "user": "scott",
            "password": "tiger",
            "database": "ORCLPDB1",
        }
    )

    assert connection_string == "oracle://scott:tiger@db.example.com:1521/ORCLPDB1"


def test_registry_creates_oracle_connector(driver: SimpleNamespace) -> None:
    database = create_connector(_CONNECTION_STRING)

    assert isinstance(database, OracleDatabase)
    assert database.dialect == "oracle"
    assert database.database_name == "FREEPDB1"
    assert driver.connections[0].pinged
    assert driver.connections[0].closed


def test_registry_passes_the_schema_allowlist(driver: SimpleNamespace) -> None:
    database = create_connector(_CONNECTION_STRING, schemas=["hr", " Sales "])

    assert isinstance(database, OracleDatabase)
    database.get_tables()

    statement = driver.connections[-1].executed[0][0]
    assert "UPPER(t.OWNER) IN ('HR', 'SALES')" in statement
    assert "CURRENT_SCHEMA" not in statement


def test_introspection_defaults_to_the_connecting_users_schema(
    driver: SimpleNamespace,
) -> None:
    database = OracleDatabase(_CONNECTION_STRING)
    driver.results.append((["X"], []))

    database.get_columns()

    statement = driver.connections[-1].executed[0][0]
    assert "OWNER = SYS_CONTEXT('USERENV', 'CURRENT_SCHEMA')" in statement


def test_execute_returns_dataframe_and_closes_connection(
    driver: SimpleNamespace,
) -> None:
    database = OracleDatabase(_CONNECTION_STRING)

    result = database.execute("SELECT :1 AS VALUE FROM DUAL;", [1])

    connection = driver.connections[-1]
    assert result.to_dict(orient="records") == [{"VALUE": 1}]
    assert connection.connect_kwargs["service_name"] == "FREEPDB1"
    # The trailing semicolon is dropped, and valid Oracle SQL is otherwise sent
    # exactly as written.
    assert connection.executed == [("SELECT :1 AS VALUE FROM DUAL", [1])]
    assert connection.call_timeout is None
    assert connection.closed


def test_execute_sends_statements_as_written(driver: SimpleNamespace) -> None:
    # No rewriting: callers write Oracle SQL because ``dialect`` says so, and
    # what the agent wrote is what the user is shown, so it must be what ran.
    database = OracleDatabase(_CONNECTION_STRING)
    statements = [
        'SELECT * FROM "HR"."EMPLOYEES" LIMIT 1000',
        "SELECT COUNT(*) AS n FROM (SELECT 1 FROM DUAL) AS t",
        'SELECT e."NAME" FROM FREEPDB1.HR."EMPLOYEES" e',
    ]

    for statement in statements:
        database.execute(statement)

    sent = [connection.executed[0][0] for connection in driver.connections[1:]]
    assert sent == statements


def test_a_syntax_error_reaches_the_caller_without_a_retry(
    driver: SimpleNamespace,
) -> None:
    # The agent's run_sql hands Oracle's error back to the model, which is how
    # a LIMIT it should not have written gets fixed.
    database = OracleDatabase(_CONNECTION_STRING)
    driver.results.append(_oracle_error("ORA-00933"))

    with pytest.raises(oracledb.DatabaseError):
        database.execute('SELECT * FROM "HR"."EMPLOYEES" LIMIT 5')

    assert len(driver.connections[-1].executed) == 1
    assert driver.connections[-1].closed


def test_execute_does_not_retry_other_errors(driver: SimpleNamespace) -> None:
    database = OracleDatabase(_CONNECTION_STRING)
    driver.results.append(_oracle_error("ORA-00942"))

    with pytest.raises(oracledb.DatabaseError):
        database.execute("SELECT * FROM missing_table")

    assert len(driver.connections[-1].executed) == 1
    assert driver.connections[-1].closed


def test_core_profiling_queries_reach_oracle_as_valid_oracle(
    driver: SimpleNamespace,
) -> None:
    # End to end through the real connector: the core writes the row limit in
    # Oracle's syntax itself, so nothing has to be rewritten on the way.
    database = OracleDatabase(_CONNECTION_STRING)
    driver.results.append((["ID"], [(1,), (2,)]))

    values = probe_ascending_distinct_values(database, "HR", "EMPLOYEES", "ID")

    assert values == [1, 2]
    statement = driver.connections[-1].executed[0][0]
    # sqlglot's Oracle output spells IS NOT NULL as NOT ... IS NULL; same meaning.
    assert statement == (
        'SELECT DISTINCT "ID" FROM "HR"."EMPLOYEES" WHERE NOT "ID" IS NULL '
        'ORDER BY "ID" FETCH FIRST 20 ROWS ONLY'
    )


def test_text_cast_type_is_comparable_in_oracle() -> None:
    # The base default, TEXT, becomes CLOB in Oracle, which ``=`` rejects.
    assert OracleDatabase.text_cast_type == "VARCHAR2(4000)"
    cast = exp.cast(exp.column("x"), OracleDatabase.text_cast_type)
    assert cast.sql(dialect="oracle") == "CAST(x AS VARCHAR2(4000))"


def test_timeout_is_set_per_call_and_becomes_statement_timeout(
    driver: SimpleNamespace,
) -> None:
    database = OracleDatabase(_CONNECTION_STRING)
    driver.results.append(_oracle_error("DPY-4024"))

    with pytest.raises(StatementTimeout) as raised:
        database.execute("SELECT 1 FROM DUAL", timeout_s=2.5)

    assert raised.value.timeout_s == 2.5
    assert driver.connections[-1].call_timeout == 2500
    assert driver.connections[-1].closed


def test_timeout_code_without_a_cap_is_not_a_statement_timeout(
    driver: SimpleNamespace,
) -> None:
    database = OracleDatabase(_CONNECTION_STRING)
    driver.results.append(_oracle_error("DPY-4024"))

    with pytest.raises(oracledb.DatabaseError):
        database.execute("SELECT 1 FROM DUAL")


def test_get_columns_returns_boolean_nullability(driver: SimpleNamespace) -> None:
    database = OracleDatabase(_CONNECTION_STRING)
    driver.results.append(
        (
            [
                "table_schema",
                "table_name",
                "column_name",
                "data_type",
                "is_nullable",
                "ordinal_position",
            ],
            [
                ("HR", "EMPLOYEES", "ID", "NUMBER", 0, 1),
                ("HR", "EMPLOYEES", "NAME", "VARCHAR2", 1, 2),
            ],
        )
    )

    columns = database.get_columns()

    assert columns.is_nullable.dtype == bool
    assert columns.is_nullable.tolist() == [False, True]
    statement = driver.connections[-1].executed[0][0]
    assert "CASE WHEN NULLABLE = 'Y' THEN 1 ELSE 0 END" in statement


def test_empty_introspection_keeps_the_contract_columns(
    driver: SimpleNamespace,
) -> None:
    database = OracleDatabase(_CONNECTION_STRING)
    empty = (["X"], [])
    driver.results.extend([empty] * 6)

    assert list(database.get_tables().columns) == [
        "table_schema",
        "table_name",
        "table_type",
    ]
    assert "is_nullable" in database.get_columns().columns
    assert list(database.get_views().columns) == [
        "table_schema",
        "table_name",
        "view_definition",
    ]
    assert "ordinal_position" in database.get_pks().columns
    assert "referenced_column" in database.get_fks().columns
    assert list(database.get_queries().columns) == ["end_time", "query_text"]


def test_get_views_includes_materialized_views(driver: SimpleNamespace) -> None:
    database = OracleDatabase(_CONNECTION_STRING)
    columns = ["table_schema", "table_name", "view_definition"]
    driver.results.extend(
        [
            (columns, [("HR", "V_ACTIVE", "SELECT 1 FROM DUAL")]),
            (columns, [("HR", "MV_TOTALS", "SELECT 2 FROM DUAL")]),
        ]
    )

    views = database.get_views()

    assert views.table_name.tolist() == ["V_ACTIVE", "MV_TOTALS"]


def test_get_schemas_lists_only_user_schemas(driver: SimpleNamespace) -> None:
    database = OracleDatabase(_CONNECTION_STRING)
    driver.results.append((["USERNAME"], [("HR",), ("SALES",)]))

    assert database.get_schemas() == ["HR", "SALES"]
    assert "ORACLE_MAINTAINED = 'N'" in driver.connections[-1].executed[0][0]


def test_qualify_quotes_schema_and_table(driver: SimpleNamespace) -> None:
    database = OracleDatabase(_CONNECTION_STRING)

    assert database.qualify("HR", "EMPLOYEES") == '"HR"."EMPLOYEES"'


def test_oracle_dialect_rules_replace_limit() -> None:
    rules = format_dialect_rules("Oracle")

    assert rules == format_dialect_rules("oracle")
    assert "FETCH FIRST n ROWS ONLY" in rules
    assert "FROM DUAL" in rules
    for dialect in ("sqlite", "postgres", "snowflake", "mysql", None):
        assert "Oracle-specific" not in format_dialect_rules(dialect)
