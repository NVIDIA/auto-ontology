# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trino connector tests.

Everything here is driver-free: the connector is exercised against a fake
``trino.dbapi`` connection, so no cluster has to be reachable.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import parse_qs, urlparse

import pandas as pd
import pytest

from auto_ontology.connectors.connection_string_factory import build_connection_string
from auto_ontology.connectors.registry import CONNECTOR_REGISTRY
from auto_ontology.connectors.trino import (
    TrinoDatabase,
    _parse_connection_string,
    _qualified,
    _quoted_identifier,
    _quoted_literal,
)

HOST = "trino.example.com"


def _connection(**overrides: object) -> dict[str, object]:
    connection: dict[str, object] = {
        "type": "trino",
        "host": HOST,
        "user": "analyst",
        "database": "hive",
    }
    connection.update(overrides)
    return connection


def _database(**overrides: object) -> TrinoDatabase:
    return TrinoDatabase(build_connection_string(_connection(**overrides)))


# ----------------------------------------------------------------------
# Fake driver
# ----------------------------------------------------------------------


class _FakeCursor:
    def __init__(self, recorder: "_FakeConnection") -> None:
        self._recorder = recorder
        self.description: list[tuple[str, str]] | None = None

    def execute(self, sql: str, parameters: Any = None) -> None:
        self._recorder.statements.append(sql)
        self.description = self._recorder.description

    def fetchall(self) -> list[list[Any]]:
        return self._recorder.rows

    def close(self) -> None:
        self._recorder.closed_cursors += 1


class _FakeConnection:
    def __init__(self) -> None:
        self.statements: list[str] = []
        self.rows: list[list[Any]] = []
        self.description: list[tuple[str, str]] | None = None
        self.closed_cursors = 0
        self.closed = False

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self)

    def close(self) -> None:
        self.closed = True


def _wire(database: TrinoDatabase) -> _FakeConnection:
    """Attach a fake connection so no HTTP call is ever made."""
    fake = _FakeConnection()
    database._connection = fake  # noqa: SLF001 - test double for the driver
    return fake


# ----------------------------------------------------------------------
# Connection string
# ----------------------------------------------------------------------


def test_trino_is_registered() -> None:
    assert CONNECTOR_REGISTRY["trino"] is TrinoDatabase


def test_default_port_is_8080() -> None:
    assert urlparse(build_connection_string(_connection())).port == 8080
    assert _parse_connection_string(f"trino://analyst@{HOST}/hive")["port"] == 8080


def test_an_unauthenticated_url_needs_no_password() -> None:
    """A dev cluster runs without auth, so the username stands alone."""
    built = build_connection_string(_connection())

    assert built == f"trino://analyst@{HOST}:8080/hive"
    assert _parse_connection_string(built)["password"] is None


def test_password_is_carried_through() -> None:
    settings = _parse_connection_string(
        build_connection_string(_connection(password="s3cret", port="443"))
    )

    assert settings["password"] == "s3cret"


def test_credentials_with_reserved_characters_survive_the_round_trip() -> None:
    built = build_connection_string(_connection(password="pa/ss@word", port="443"))

    assert _parse_connection_string(built)["password"] == "pa/ss@word"


def test_schema_is_carried_through() -> None:
    built = build_connection_string(_connection(schema="default"))

    assert parse_qs(urlparse(built).query)["schema"] == ["default"]
    assert _parse_connection_string(built)["schema"] == "default"


def test_scheme_url_host_is_normalised() -> None:
    built = build_connection_string(_connection(host=f"https://{HOST}"))

    assert urlparse(built).hostname == HOST


def test_plain_port_defaults_to_http() -> None:
    assert (
        _parse_connection_string(f"trino://analyst@{HOST}:8080/hive")["http_scheme"]
        == "http"
    )


def test_a_password_implies_tls() -> None:
    """The Trino client refuses to send basic credentials over cleartext HTTP."""
    settings = _parse_connection_string(f"trino://analyst:pw@{HOST}:8080/hive")

    assert settings["http_scheme"] == "https"


def test_tls_port_implies_https() -> None:
    assert (
        _parse_connection_string(f"trino://analyst@{HOST}:443/hive")["http_scheme"]
        == "https"
    )


def test_explicit_http_scheme_wins() -> None:
    settings = _parse_connection_string(
        f"trino://analyst@{HOST}:443/hive?http_scheme=http"
    )

    assert settings["http_scheme"] == "http"


def test_unknown_http_scheme_is_rejected() -> None:
    with pytest.raises(ValueError, match="http_scheme"):
        _parse_connection_string(f"trino://analyst@{HOST}:8080/hive?http_scheme=ftp")


def test_catalog_is_required() -> None:
    with pytest.raises(ValueError, match="catalog"):
        _parse_connection_string(f"trino://analyst@{HOST}:8080")


def test_username_is_required() -> None:
    with pytest.raises(ValueError, match="username"):
        _parse_connection_string(f"trino://{HOST}:8080/hive")


def test_non_trino_url_is_rejected() -> None:
    with pytest.raises(ValueError, match="Not a Trino URL"):
        _parse_connection_string("postgresql://user:pw@host:5432/db")


# ----------------------------------------------------------------------
# Identity
# ----------------------------------------------------------------------


def test_dialect_is_trino() -> None:
    assert _database().dialect == "trino"


def test_database_name_is_the_catalog() -> None:
    """NeMo routes SQL by database_name, so it has to be the catalog."""
    assert _database(database="iceberg").database_name == "iceberg"


# ----------------------------------------------------------------------
# Identifier quoting
# ----------------------------------------------------------------------


def test_identifiers_are_double_quoted() -> None:
    assert _quoted_identifier("orders") == '"orders"'


def test_embedded_quotes_are_doubled() -> None:
    assert _quoted_identifier('we"ird') == '"we""ird"'


def test_qualified_reference_quotes_every_part() -> None:
    assert _qualified("hive", "my schema", "order") == '"hive"."my schema"."order"'


def test_literals_escape_single_quotes() -> None:
    assert _quoted_literal("O'Brien") == "'O''Brien'"


def test_generated_metadata_sql_quotes_a_catalog_with_a_space() -> None:
    """The catalog name reaches the SQL text, so it must never go in bare."""
    database = _database(database="my catalog")
    fake = _wire(database)
    fake.description = [("table_schema", "varchar")]

    database.get_tables()

    assert '"my catalog"."information_schema"."tables"' in fake.statements[0]


def test_schema_allowlist_is_pushed_down_as_escaped_literals() -> None:
    database = TrinoDatabase(
        build_connection_string(_connection()), schemas=["it's", "sales"]
    )
    fake = _wire(database)
    fake.description = [("table_schema", "varchar")]

    database.get_columns()

    assert "table_schema IN ('it''s', 'sales')" in fake.statements[0]


def test_no_allowlist_still_excludes_information_schema() -> None:
    database = _database()
    fake = _wire(database)
    fake.description = [("table_schema", "varchar")]

    database.get_tables()

    assert "table_schema <> 'information_schema'" in fake.statements[0]


def test_a_pinned_schema_seeds_the_allowlist() -> None:
    database = _database(schema="sales")
    fake = _wire(database)
    fake.description = [("table_schema", "varchar")]

    database.get_tables()

    assert "table_schema IN ('sales')" in fake.statements[0]


# ----------------------------------------------------------------------
# Execution and introspection shapes
# ----------------------------------------------------------------------


def test_execute_returns_a_frame_with_the_result_columns() -> None:
    database = _database()
    fake = _wire(database)
    fake.description = [("n", "bigint")]
    fake.rows = [[1], [2]]

    frame = database.execute('SELECT "n" FROM "t"')

    assert list(frame.columns) == ["n"]
    assert frame["n"].tolist() == [1, 2]


def test_a_statement_with_no_result_set_yields_an_empty_frame() -> None:
    database = _database()
    fake = _wire(database)
    fake.description = None

    assert database.execute("SET SESSION x = 1").empty


def test_get_schemas_lists_the_catalogs_schemas() -> None:
    database = _database()
    fake = _wire(database)
    fake.description = [("schema_name", "varchar")]
    fake.rows = [["default"], ["sales"]]

    assert database.get_schemas() == ["default", "sales"]


def test_views_are_empty_when_the_catalog_cannot_store_them() -> None:
    """Kafka/JMX catalogs error on information_schema.views; that is not a failure."""

    class _Failing(_FakeConnection):
        def cursor(self) -> _FakeCursor:
            raise RuntimeError("Table 'information_schema.views' does not exist")

    database = _database()
    database._connection = _Failing()  # noqa: SLF001 - test double for the driver

    views = database.get_views()

    assert views.empty
    assert list(views.columns) == ["table_schema", "table_name", "view_definition"]


def test_empty_metadata_results_keep_the_expected_columns() -> None:
    database = _database()
    fake = _wire(database)
    fake.description = None

    assert list(database.get_tables().columns) == [
        "table_schema",
        "table_name",
        "table_type",
    ]
    assert list(database.get_columns().columns) == [
        "table_schema",
        "table_name",
        "column_name",
        "data_type",
        "is_nullable",
        "ordinal_position",
    ]


# ----------------------------------------------------------------------
# Constraints
# ----------------------------------------------------------------------


def test_primary_keys_are_an_empty_frame_not_an_error() -> None:
    """Trino declares no constraints; ingestion must see no keys, not a failure."""
    pks = _database().get_pks()

    assert isinstance(pks, pd.DataFrame)
    assert pks.empty
    assert list(pks.columns) == [
        "table_schema",
        "table_name",
        "column_name",
        "ordinal_position",
    ]


def test_foreign_keys_are_an_empty_frame_not_an_error() -> None:
    fks = _database().get_fks()

    assert fks.empty
    assert list(fks.columns) == [
        "table_schema",
        "table_name",
        "column_name",
        "referenced_schema",
        "referenced_table",
        "referenced_column",
    ]


def test_query_history_is_empty() -> None:
    queries = _database().get_queries()

    assert queries.empty
    assert list(queries.columns) == ["end_time", "query_text"]


# ----------------------------------------------------------------------
# Lifecycle
# ----------------------------------------------------------------------


def test_close_releases_the_connection() -> None:
    database = _database()
    fake = _wire(database)

    database.close()

    assert fake.closed
    assert database._connection is None  # noqa: SLF001 - asserting on the test double


def test_qualify_prepends_the_bound_catalog() -> None:
    # Trino objects are catalog.schema.table; a two-level name resolves against
    # the session default catalog instead of the one this connection binds.
    assert _database().qualify("raw", "events") == '"hive"."raw"."events"'


def test_qualify_rejects_a_missing_schema() -> None:
    # "hive"."events" would be read as schema.table, not catalog.table.
    with pytest.raises(ValueError, match="requires a schema"):
        _database().qualify(None, "events")
