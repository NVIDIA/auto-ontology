# SPDX-FileCopyrightText: Copyright (c) 2026 Mario Nascimento
# SPDX-License-Identifier: Apache-2.0

import base64
import datetime
import re
import ssl
from types import SimpleNamespace
from typing import Any
from urllib.parse import quote

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

    def fetchmany(self) -> list[tuple[Any, ...]]:
        self._connection.fetches += 1
        self._connection.clock.now += self._connection.fetch_seconds
        batch, self._rows = self._rows[: self.arraysize], self._rows[self.arraysize :]
        return batch

    def close(self) -> None:
        self.closed = True


class Connection:
    def __init__(self, results: list[Any], connect_kwargs: dict[str, Any]) -> None:
        self._results = results
        self.connect_kwargs = connect_kwargs
        self.executed: list[tuple[str, list[Any] | None]] = []
        self.call_timeout: int | None = None
        self.call_timeouts: list[int] = []
        self.fetches = 0
        # Each fetch round trip moves *clock* on by this long.
        self.clock = SimpleNamespace(now=0.0)
        self.fetch_seconds = 0.0
        self.outputtypehandler: Any = None
        self.pinged = False
        self.closed = False

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "call_timeout" and value is not None:
            self.call_timeouts.append(value)
        super().__setattr__(name, value)

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
    state = SimpleNamespace(connections=[], results=[], connect=None)

    def connect(**kwargs: Any) -> Connection:
        connection = Connection(state.results, kwargs)
        state.connections.append(connection)
        return connection

    state.connect = connect
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


def test_parse_connection_string_is_plain_tcp_by_default() -> None:
    kwargs = _parse_connection_string("oracle://u:p@db:1521/FREEPDB1?protocol=tcp")

    assert "protocol" not in kwargs
    assert "ssl_context" not in kwargs


def test_tcps_verifies_the_server_with_the_system_trust_store() -> None:
    kwargs = _parse_connection_string("oracle://u:p@db:2484/FREEPDB1?protocol=TCPS")

    assert kwargs["protocol"] == "tcps"
    assert kwargs["ssl_server_dn_match"] is True
    context = kwargs["ssl_context"]
    assert isinstance(context, ssl.SSLContext)
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname is True
    assert "ssl_server_cert_dn" not in kwargs


def _self_signed_pem() -> bytes:
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Test CA")])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(1)
        .not_valid_before(now)
        .not_valid_after(now + datetime.timedelta(days=1))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), True)
        .sign(key, hashes.SHA256())
    )
    return cert.public_bytes(serialization.Encoding.PEM)


@pytest.mark.parametrize("encoding", ["pem", "der"])
def test_tcps_trusts_an_uploaded_ca(encoding: str) -> None:
    pem = _self_signed_pem()
    data = pem if encoding == "pem" else ssl.PEM_cert_to_DER_cert(pem.decode())
    encoded = quote(base64.b64encode(data).decode(), safe="")

    kwargs = _parse_connection_string(
        f"oracle://u:p@db:2484/FREEPDB1?protocol=tcps&ssl_ca_data={encoded}"
    )

    assert kwargs["ssl_context"].get_ca_certs()[0]["subject"] == (
        (("commonName", "Test CA"),),
    )


def test_tcps_trusts_a_ca_file(tmp_path: Any) -> None:
    ca = tmp_path / "ca.pem"
    ca.write_bytes(_self_signed_pem())

    kwargs = _parse_connection_string(
        f"oracle://u:p@db:2484/FREEPDB1?protocol=tcps&ssl_ca={quote(str(ca))}"
    )

    assert len(kwargs["ssl_context"].get_ca_certs()) == 1


def test_tcps_matches_a_server_dn_instead_of_the_host_name() -> None:
    dn = quote("CN=adb.example.com,O=Oracle Corporation", safe="")

    kwargs = _parse_connection_string(
        f"oracle://u:p@10.0.0.5:1522/svc?protocol=tcps&ssl_server_cert_dn={dn}"
    )

    assert kwargs["ssl_server_cert_dn"] == "CN=adb.example.com,O=Oracle Corporation"
    assert kwargs["ssl_server_dn_match"] is True
    # The driver checks the DN; the chain is still verified.
    assert kwargs["ssl_context"].check_hostname is False
    assert kwargs["ssl_context"].verify_mode == ssl.CERT_REQUIRED


@pytest.mark.parametrize(
    ("query", "message"),
    [
        # A typo must not leave the connection unencrypted.
        ("protocl=tcps", "Unsupported Oracle connection option"),
        ("ssl_verify=false", "Unsupported Oracle connection option"),
        ("protocol=ssl", "Unsupported Oracle protocol"),
        ("protocol=tcps&protocol=tcp", "given twice"),
        ("ssl_ca_data=AAAA", "add protocol=tcps"),
        ("ssl_server_cert_dn=CN%3Dx", "add protocol=tcps"),
        ("protocol=tcps&ssl_ca=a.pem&ssl_ca_data=AAAA", "not both"),
        ("protocol=tcps&ssl_ca_data=not*base64", "not valid base64"),
        ("protocol=tcps&ssl_ca_data=AAAA", "could not be loaded"),
        ("protocol=tcps&ssl_ca=/no/such/ca.pem", "could not be loaded"),
    ],
)
def test_bad_tls_options_are_rejected(query: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _parse_connection_string(f"oracle://u:p@db:2484/FREEPDB1?{query}")


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


def test_build_connection_string_carries_tls_options() -> None:
    pem = _self_signed_pem()
    uploaded = base64.b64encode(pem).decode()

    connection_string = build_connection_string(
        {
            "type": "oracle",
            "host": "adb.example.com",
            "port": "1522",
            "user": "admin",
            "password": "pw",
            "database": "svc_high",
            "protocol": "TCPS",
            "ssl_ca_file": uploaded,
            "ssl_server_cert_dn": "CN=adb.example.com,O=Oracle Corporation",
        }
    )

    kwargs = _parse_connection_string(connection_string)
    assert kwargs["protocol"] == "tcps"
    assert len(kwargs["ssl_context"].get_ca_certs()) == 1
    assert kwargs["ssl_server_cert_dn"] == "CN=adb.example.com,O=Oracle Corporation"


def test_build_connection_string_passes_a_ca_without_tcps_to_fail() -> None:
    # Dropping the CA here would silently connect without TLS; the connector
    # rejects it instead, so the connection test shows the mistake.
    connection_string = build_connection_string(
        {
            "type": "oracle",
            "host": "db",
            "user": "u",
            "password": "p",
            "database": "svc",
            "ssl_ca_file": "AAAA",
        }
    )

    with pytest.raises(ValueError, match="add protocol=tcps"):
        _parse_connection_string(connection_string)


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
    driver.results.append((["USERNAME"], [("HR",), ("SALES",)]))

    assert isinstance(database, OracleDatabase)
    database.get_tables()

    statement = driver.connections[-1].executed[0][0]
    assert "t.OWNER IN ('HR', 'SALES')" in statement
    assert "UPPER(" not in statement
    assert "CURRENT_SCHEMA" not in statement


@pytest.mark.parametrize(
    ("requested", "existing", "expected"),
    [
        # Picked from get_schemas(): already exact.
        (["HR"], ["HR", "hr"], ["HR"]),
        (["hr"], ["HR", "hr"], ["hr"]),
        # Typed bare, no schema by that exact name: Oracle reads it as upper case.
        (["hr"], ["HR"], ["HR"]),
        # Typed in double quotes: exact, never folded.
        (['"hr"'], ["HR", "hr"], ["hr"]),
        # Unknown names stay as given and match nothing.
        (["nobody"], [], ["nobody"]),
    ],
)
def test_schema_allowlist_matches_exact_schema_names(
    driver: SimpleNamespace,
    requested: list[str],
    existing: list[str],
    expected: list[str],
) -> None:
    database = OracleDatabase(_CONNECTION_STRING, schemas=requested)
    driver.results.append((["USERNAME"], [(name,) for name in existing]))
    driver.results.append((["X"], []))

    database.get_columns()

    lookup = driver.connections[-2].executed[0][0]
    assert lookup.startswith("SELECT USERNAME FROM ALL_USERS WHERE USERNAME IN (")
    statement = driver.connections[-1].executed[0][0]
    names = ", ".join(f"'{name}'" for name in expected)
    assert f"OWNER IN ({names})" in statement


def test_a_large_allowlist_stays_within_oracles_in_list_limit(
    driver: SimpleNamespace,
) -> None:
    # 600 lowercase names look up 1,200 candidates; one IN list over 1,000
    # fails with ORA-01795.
    requested = [f"s{i:03d}" for i in range(600)]
    database = OracleDatabase(_CONNECTION_STRING, schemas=requested)
    # Half exist as typed, half only in upper case.
    existing = requested[:300] + [name.upper() for name in requested[300:]]
    driver.results.append((["USERNAME"], [(name,) for name in existing]))
    driver.results.append((["X"], []))

    database.get_columns()

    lookup = driver.connections[-2].executed[0][0]
    statement = driver.connections[-1].executed[0][0]
    for sql in (lookup, statement):
        lists = re.findall(r"IN \(([^)]*)\)", sql)
        assert all(group.count("'") // 2 <= 1000 for group in lists)
    assert len(re.findall(r"IN \(", lookup)) == 2
    # Exact names keep precedence; the rest resolve to upper case.
    assert "'s000'" in statement and "'S000'" not in statement
    assert "'S599'" in statement and "'s599'" not in statement


def test_schema_names_are_resolved_once(driver: SimpleNamespace) -> None:
    database = OracleDatabase(_CONNECTION_STRING, schemas=["hr"])
    driver.results.append((["USERNAME"], [("HR",)]))
    driver.results.extend([(["X"], [])] * 2)

    database.get_columns()
    database.get_columns()

    statements = [c.executed[0][0] for c in driver.connections if c.executed]
    assert sum("FROM ALL_USERS" in s for s in statements) == 1


def test_get_tables_keeps_index_organized_tables(driver: SimpleNamespace) -> None:
    database = OracleDatabase(_CONNECTION_STRING)
    driver.results.append((["X"], []))

    database.get_tables()

    statement = driver.connections[-1].executed[0][0]
    # IOT_TYPE is 'IOT' for the table itself, and 'IOT_OVERFLOW' or
    # 'IOT_MAPPING' for its storage, which stay out.
    assert "(t.IOT_TYPE IS NULL OR t.IOT_TYPE = 'IOT')" in statement


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


def test_timeout_covers_the_fetches_as_well_as_the_execute(
    driver: SimpleNamespace, monkeypatch: MonkeyPatch
) -> None:
    # call_timeout bounds one round trip. A result fetched in several must
    # still stop at timeout_s overall, each trip getting only what is left.
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(
        "auto_ontology.connectors.oracle.time.monotonic", lambda: clock.now
    )
    database = OracleDatabase(_CONNECTION_STRING)
    driver.results.append((["ID"], [(n,) for n in range(350)]))
    connect = driver.connect

    def slow_connect(**kwargs: Any) -> Connection:
        connection = connect(**kwargs)
        connection.clock = clock
        connection.fetch_seconds = 1.0
        return connection

    monkeypatch.setattr(
        "auto_ontology.connectors.oracle.oracledb.connect", slow_connect
    )

    with pytest.raises(StatementTimeout) as raised:
        database.execute("SELECT ID FROM T", timeout_s=2.5)

    connection = driver.connections[-1]
    assert raised.value.timeout_s == 2.5
    # Execute and two fetches got 2.5s, 2.5s and 1.5s; the third fetch had
    # 0.5s left, and the fourth was never sent.
    assert connection.call_timeouts == [2500, 2500, 1500, 500]
    assert connection.fetches == 3
    assert connection.closed


def test_a_result_fetched_within_the_cap_is_returned_whole(
    driver: SimpleNamespace,
) -> None:
    database = OracleDatabase(_CONNECTION_STRING)
    driver.results.append((["ID"], [(n,) for n in range(250)]))

    result = database.execute("SELECT ID FROM T", timeout_s=30)

    assert result.ID.tolist() == list(range(250))
    assert driver.connections[-1].fetches == 4


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
    statement = driver.connections[-1].executed[0][0]
    assert "ORACLE_MAINTAINED = 'N'" in statement
    # Every database user is a schema; only those with something to read are
    # offered.
    for view in ("ALL_TABLES", "ALL_VIEWS", "ALL_MVIEWS"):
        assert f"EXISTS (SELECT 1 FROM {view}" in statement


def test_qualify_quotes_schema_and_table(driver: SimpleNamespace) -> None:
    database = OracleDatabase(_CONNECTION_STRING)

    assert database.qualify("HR", "EMPLOYEES") == '"HR"."EMPLOYEES"'


def test_oracle_dialect_rules_replace_limit() -> None:
    rules = format_dialect_rules("Oracle")

    assert rules == format_dialect_rules("oracle")
    assert "FETCH FIRST n ROWS ONLY" in rules
    assert "FROM DUAL" in rules
    # 23ai has a native BOOLEAN type; only flags stored as text or numbers
    # are compared against their stored values.
    assert "BOOLEAN (Oracle 23ai and later) takes TRUE / FALSE" in rules
    assert "There is no BOOLEAN column type" not in rules
    for dialect in ("sqlite", "postgres", "snowflake", "mysql", None):
        assert "Oracle-specific" not in format_dialect_rules(dialect)
