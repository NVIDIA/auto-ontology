# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from typing import Any

import oracledb
import pytest
from pytest import MonkeyPatch

from auto_ontology.connectors.base import StatementTimeout
from auto_ontology.connectors.connection_string_factory import build_connection_string
from auto_ontology.connectors.oracle import (
    OracleDatabase,
    _parse_connection_string,
    drop_database_qualifier,
    to_oracle_sql,
)
from auto_ontology.connectors.registry import create_connector
from auto_ontology.retrieval.text_to_sql.prompts import format_dialect_rules

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


def test_execute_drops_the_database_qualifier(driver: SimpleNamespace) -> None:
    # The catalog knows a table as database.schema.table, and generated SQL can
    # carry all three; Oracle rejects that spelling (ORA-03048 on 23ai).
    database = OracleDatabase(_CONNECTION_STRING)

    database.execute(
        'SELECT e."NAME" FROM FREEPDB1.HR."EMPLOYEES" e '
        'ORDER BY e."SALARY" DESC FETCH FIRST 2 ROWS ONLY'
    )

    assert driver.connections[-1].executed == [
        (
            'SELECT e."NAME" FROM HR."EMPLOYEES" e '
            'ORDER BY e."SALARY" DESC FETCH FIRST 2 ROWS ONLY',
            [],
        )
    ]


def test_database_qualifier_is_dropped_from_tables_and_columns() -> None:
    rewritten = drop_database_qualifier(
        'SELECT "FREEPDB1"."HR"."EMPLOYEES"."NAME" FROM "FREEPDB1"."HR"."EMPLOYEES" '
        "JOIN freepdb1.hr.departments d ON d.id = 1",
        "FREEPDB1",
    )

    assert "FREEPDB1" not in rewritten.upper()
    assert 'FROM "HR"."EMPLOYEES"' in rewritten
    assert "JOIN hr.departments d" in rewritten


def test_only_this_connections_database_is_dropped() -> None:
    sql = "SELECT * FROM OTHERDB.HR.EMPLOYEES"

    assert drop_database_qualifier(sql, "FREEPDB1") == sql


def test_a_schema_named_like_the_database_is_left_alone(
    driver: SimpleNamespace,
) -> None:
    database = OracleDatabase("oracle://hr:pw@localhost:1521/HR")

    database.execute("SELECT * FROM HR.EMPLOYEES")

    assert driver.connections[-1].executed == [("SELECT * FROM HR.EMPLOYEES", [])]


def test_execute_rewrites_limit_before_sending(driver: SimpleNamespace) -> None:
    database = OracleDatabase(_CONNECTION_STRING)

    database.execute('SELECT * FROM "HR"."EMPLOYEES" LIMIT 1000')

    assert driver.connections[-1].executed == [
        ('SELECT * FROM "HR"."EMPLOYEES" FETCH FIRST 1000 ROWS ONLY', [])
    ]


@pytest.mark.parametrize("code", ["ORA-00933", "ORA-00907", "ORA-03048"])
def test_execute_retries_once_when_oracle_rejects_the_syntax(
    driver: SimpleNamespace, code: str
) -> None:
    database = OracleDatabase(_CONNECTION_STRING)
    driver.results.extend([_oracle_error(code), (["N"], [(3,)])])

    result = database.execute("SELECT COUNT(*) AS n FROM (SELECT 1 FROM DUAL) AS t")

    connection = driver.connections[-1]
    assert [statement for statement, _ in connection.executed] == [
        "SELECT COUNT(*) AS n FROM (SELECT 1 FROM DUAL) AS t",
        "SELECT COUNT(*) AS n FROM (SELECT 1 FROM DUAL) t",
    ]
    assert result.iloc[0, 0] == 3
    assert connection.closed


def test_execute_does_not_retry_other_errors(driver: SimpleNamespace) -> None:
    database = OracleDatabase(_CONNECTION_STRING)
    driver.results.append(_oracle_error("ORA-00942"))

    with pytest.raises(oracledb.DatabaseError):
        database.execute("SELECT * FROM missing_table")

    assert len(driver.connections[-1].executed) == 1
    assert driver.connections[-1].closed


def test_core_probe_queries_become_valid_oracle() -> None:
    # The shapes issued by semantic/visit_enter.py, semantic/semantic_fk.py and
    # retrieval/text_to_sql/db_probe, none of which Oracle accepts as written.
    assert (
        to_oracle_sql('SELECT DISTINCT "A" FROM "HR"."T" WHERE "A" IS NOT NULL LIMIT 5')
        == 'SELECT DISTINCT "A" FROM "HR"."T" WHERE NOT "A" IS NULL '
        "FETCH FIRST 5 ROWS ONLY"
    )
    overlap = to_oracle_sql(
        'SELECT COUNT(*) AS n FROM (SELECT "A" FROM "HR"."T" LIMIT 1) AS present '
        'UNION ALL SELECT COUNT(*) FROM (SELECT "A" FROM "HR"."T" LIMIT 1) AS shared'
    )
    assert "LIMIT" not in overlap
    assert " AS present" not in overlap and " AS shared" not in overlap
    assert overlap.count("FETCH FIRST 1 ROWS ONLY") == 2


def test_unparseable_sql_is_left_for_oracle_to_report() -> None:
    sql = "SELECT owner FROM all_tables WHERE owner = :1 LIMIT"

    assert to_oracle_sql(sql) == sql


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
