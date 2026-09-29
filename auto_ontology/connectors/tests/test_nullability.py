# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``is_nullable`` is a boolean at every connector's edge.

Each connector converts differently — Postgres reads ``NOT attnotnull``
straight from ``pg_attribute``, SQLite negates ``PRAGMA table_info``'s
``notnull`` flag, and the ``information_schema`` connectors compare against the
literal ``'YES'``. What they must agree on is the *type* they hand back, since
both ``'YES'`` and ``'NO'`` are truthy and a single connector leaking strings
marks its whole database nullable.

Postgres has its own coverage in ``test_postgres.py`` against the Pagila
fixture. Snowflake, Databricks and HeavyDB need live warehouses, so their
conversions are asserted against the emitted SQL here and verified against a
real warehouse separately.
"""

from __future__ import annotations

import sqlite3

import duckdb
import pytest

from auto_ontology.connectors.databricks import DatabricksDatabase
from auto_ontology.connectors.duckdb import DuckDBDatabase
from auto_ontology.connectors.snowflake import SnowflakeDatabase
from auto_ontology.connectors.sqlite import SQLiteDatabase


@pytest.fixture
def sqlite_db(tmp_path):
    path = tmp_path / "t.sqlite"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE t (a INTEGER NOT NULL, b TEXT, c REAL NOT NULL)")
    conn.commit()
    conn.close()
    db = SQLiteDatabase(f"sqlite:///{path}")
    yield db
    db.close()


@pytest.fixture
def duckdb_db(tmp_path):
    path = tmp_path / "t.duckdb"
    conn = duckdb.connect(str(path))
    conn.execute("CREATE TABLE t (a INTEGER NOT NULL, b TEXT, c DATE NOT NULL)")
    conn.close()
    db = DuckDBDatabase(str(path))
    yield db
    db.close()


def _nullability(connector) -> dict[str, object]:
    columns = connector.get_columns()
    return dict(zip(columns.column_name, columns.is_nullable))


def test_sqlite_returns_booleans(sqlite_db) -> None:
    values = _nullability(sqlite_db)
    assert values == {"a": False, "b": True, "c": False}
    assert {type(v) for v in values.values()} == {bool}


def test_duckdb_returns_booleans(duckdb_db) -> None:
    """DuckDB's ``information_schema`` reports the ``'YES'``/``'NO'`` text.

    The connector compares in SQL rather than passing it through, so what
    reaches pandas is already a boolean column.
    """
    values = _nullability(duckdb_db)
    assert values == {"a": False, "b": True, "c": False}
    assert {type(v) for v in values.values()} == {bool}


def test_duckdb_column_dtype_is_bool_not_object(duckdb_db) -> None:
    """An object column of Python bools would still be a latent string column."""
    assert duckdb_db.get_columns().is_nullable.dtype == bool


@pytest.mark.parametrize("connector_cls", [SnowflakeDatabase, DatabricksDatabase])
def test_information_schema_connectors_compare_rather_than_select(
    connector_cls,
) -> None:
    """Snowflake and Databricks both document ``IS_NULLABLE`` as ``'YES'``/``'NO'``.

    Neither can be exercised without a live warehouse, so this pins the one
    thing that makes them wrong: selecting the column bare instead of
    comparing it. Kept as a source assertion so a refactor that reverts the
    conversion fails here rather than in production.

    https://docs.snowflake.com/en/sql-reference/info-schema/columns
    https://docs.databricks.com/aws/en/sql/language-manual/information-schema/columns
    """
    import inspect

    source = inspect.getsource(connector_cls.get_columns)
    normalised = " ".join(source.split()).upper()
    assert "IS_NULLABLE = 'YES'" in normalised, (
        f"{connector_cls.__name__}.get_columns must convert IS_NULLABLE to a "
        "boolean in SQL; selecting it bare yields 'YES'/'NO', both truthy."
    )
