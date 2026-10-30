# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DuckDB reports the keys its tables declare.

The catalog keys prediction-graph entities on a table's ``pk`` and builds its
edges from foreign keys, so a connector that returns no keys leaves every
DuckDB source without either. Exercised against a real file, since DuckDB runs
in-process.
"""

from __future__ import annotations

from collections.abc import Iterator

import duckdb
import pytest

from auto_ontology.connectors.duckdb import DuckDBDatabase

_PK_COLUMNS = ["table_schema", "table_name", "column_name", "ordinal_position"]
_FK_COLUMNS = [
    "table_schema",
    "table_name",
    "column_name",
    "referenced_schema",
    "referenced_table",
    "referenced_column",
]


@pytest.fixture
def keyed_db(tmp_path) -> Iterator[DuckDBDatabase]:
    path = tmp_path / "keyed.duckdb"
    conn = duckdb.connect(str(path))
    conn.execute(
        """
        CREATE SCHEMA "Sales Ops";
        CREATE TABLE "Sales Ops".customers (id INTEGER PRIMARY KEY, email TEXT);
        -- Key declared in a different order from the table's columns.
        CREATE TABLE "Sales Ops"."Order Lines" (
            order_id INTEGER,
            line_no INTEGER,
            "Cust Id" INTEGER REFERENCES "Sales Ops".customers,
            PRIMARY KEY (line_no, order_id)
        );
        CREATE TABLE "Sales Ops".shipments (
            sid INTEGER PRIMARY KEY,
            o_id INTEGER,
            l_no INTEGER,
            FOREIGN KEY (l_no, o_id)
                REFERENCES "Sales Ops"."Order Lines" (line_no, order_id)
        );
        CREATE TABLE "Sales Ops".unkeyed (x INTEGER);
        """
    )
    conn.close()
    db = DuckDBDatabase(str(path))
    yield db
    db.close()


@pytest.fixture
def unkeyed_db() -> Iterator[DuckDBDatabase]:
    db = DuckDBDatabase(":memory:", read_only=False)
    db.execute("CREATE TABLE t (a INTEGER)")
    yield db
    db.close()


def _rows(frame) -> list[tuple]:
    return [tuple(row) for row in frame.itertuples(index=False)]


def test_composite_primary_key_keeps_declared_order(keyed_db) -> None:
    pks = keyed_db.get_pks()
    assert list(pks.columns) == _PK_COLUMNS
    assert _rows(pks) == [
        ("Sales Ops", "Order Lines", "line_no", 1),
        ("Sales Ops", "Order Lines", "order_id", 2),
        ("Sales Ops", "customers", "id", 1),
        ("Sales Ops", "shipments", "sid", 1),
    ]


def test_composite_foreign_key_pairs_columns_by_position(keyed_db) -> None:
    """``(l_no, o_id)`` references ``(line_no, order_id)``, not their cross product.

    The single-column key names no target column, so DuckDB resolves it to the
    referenced table's primary key.
    """
    fks = keyed_db.get_fks()
    assert list(fks.columns) == _FK_COLUMNS
    assert _rows(fks) == [
        ("Sales Ops", "Order Lines", "Cust Id", "Sales Ops", "customers", "id"),
        ("Sales Ops", "shipments", "l_no", "Sales Ops", "Order Lines", "line_no"),
        ("Sales Ops", "shipments", "o_id", "Sales Ops", "Order Lines", "order_id"),
    ]


def test_attached_catalog_keys_are_not_reported(keyed_db, tmp_path) -> None:
    """Same-named tables in another catalog must not contribute their keys.

    Every column the attached keys name also exists here, so a leaked row
    would be stored against this database's tables rather than dropped.
    """
    other = tmp_path / "other.duckdb"
    conn = duckdb.connect(str(other))
    conn.execute(
        """
        CREATE SCHEMA "Sales Ops";
        CREATE TABLE "Sales Ops".customers (email TEXT PRIMARY KEY);
        CREATE TABLE "Sales Ops"."Order Lines" (
            "Cust Id" TEXT REFERENCES "Sales Ops".customers
        );
        """
    )
    conn.close()
    keyed_db.execute(f"ATTACH '{other}' AS other (READ_ONLY)")
    pks = keyed_db.get_pks()
    assert _rows(pks[pks.table_name == "customers"]) == [
        ("Sales Ops", "customers", "id", 1)
    ]
    fks = keyed_db.get_fks()
    assert _rows(fks[fks.table_name == "Order Lines"]) == [
        ("Sales Ops", "Order Lines", "Cust Id", "Sales Ops", "customers", "id")
    ]


def test_database_without_keys_returns_shaped_empty_frames(unkeyed_db) -> None:
    """Empty, but with the columns ``add_pks``/``add_fks`` read."""
    pks = unkeyed_db.get_pks()
    fks = unkeyed_db.get_fks()
    assert pks.empty and list(pks.columns) == _PK_COLUMNS
    assert fks.empty and list(fks.columns) == _FK_COLUMNS
