# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``SQLiteDatabase.get_fks`` names columns the catalog can resolve.

``add_fks`` matches each row against the ingested catalog by exact
``(schema, table, column)``, and skips a key it can't find, so a reference
SQLite left implicit or spelled in another case was dropped silently.
"""

from __future__ import annotations

import sqlite3

import pytest

from auto_ontology.connectors.sqlite import SQLiteDatabase


def _connector(tmp_path, ddl: str):
    path = tmp_path / "shop.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript(ddl)
    conn.close()
    return SQLiteDatabase(f"sqlite:///{path}")


@pytest.fixture
def connector_factory(tmp_path):
    opened: list[SQLiteDatabase] = []

    def make(ddl: str) -> SQLiteDatabase:
        db = _connector(tmp_path, ddl)
        opened.append(db)
        return db

    yield make
    for db in opened:
        db.close()


def _references(db: SQLiteDatabase) -> set[tuple[str, str, str, str]]:
    fks = db.get_fks()
    return {
        (r.table_name, r.column_name, r.referenced_table, r.referenced_column)
        for r in fks.itertuples()
    }


def test_explicit_reference_is_reported_as_declared(connector_factory) -> None:
    db = connector_factory(
        """
        CREATE TABLE customer (id INTEGER PRIMARY KEY);
        CREATE TABLE orders (id INTEGER PRIMARY KEY,
                             customer_id INTEGER REFERENCES customer(id));
        """
    )
    assert _references(db) == {("orders", "customer_id", "customer", "id")}


def test_reference_without_columns_targets_the_parent_primary_key(
    connector_factory,
) -> None:
    """``REFERENCES customer`` means customer's primary key; SQLite reports NULL."""
    db = connector_factory(
        """
        CREATE TABLE customer (customer_key INTEGER PRIMARY KEY, name TEXT);
        CREATE TABLE orders (id INTEGER PRIMARY KEY,
                             customer_id INTEGER REFERENCES customer);
        """
    )
    assert _references(db) == {("orders", "customer_id", "customer", "customer_key")}


def test_composite_reference_without_columns_follows_key_order(
    connector_factory,
) -> None:
    """Each child column pairs with the parent key column in key order."""
    db = connector_factory(
        """
        CREATE TABLE stock (sku TEXT, warehouse TEXT, qty INTEGER,
                            PRIMARY KEY (warehouse, sku));
        CREATE TABLE pick (wh TEXT, item TEXT,
                           FOREIGN KEY (wh, item) REFERENCES stock);
        """
    )
    assert _references(db) == {
        ("pick", "wh", "stock", "warehouse"),
        ("pick", "item", "stock", "sku"),
    }


def test_names_are_resolved_to_their_declared_case(connector_factory) -> None:
    db = connector_factory(
        """
        CREATE TABLE Customer (Id INTEGER PRIMARY KEY);
        CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER,
                             FOREIGN KEY (CUSTOMER_ID) REFERENCES customer(ID));
        """
    )
    assert _references(db) == {("orders", "customer_id", "Customer", "Id")}


def test_unresolvable_parent_is_passed_through(connector_factory) -> None:
    """A parent that isn't in the database keeps the name the DDL gave it."""
    db = connector_factory(
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, x INTEGER REFERENCES gone);"
    )
    fks = db.get_fks()
    assert fks.referenced_table.tolist() == ["gone"]
    assert fks.referenced_column.isna().all()
