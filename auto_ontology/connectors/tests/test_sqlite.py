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


def test_non_ascii_names_are_not_folded_together(connector_factory) -> None:
    """SQLite folds ASCII letters only, so "Ä" and "ä" are different tables."""
    db = connector_factory(
        """
        CREATE TABLE "Ä" (upper_key INTEGER PRIMARY KEY);
        CREATE TABLE "ä" (lower_key INTEGER PRIMARY KEY);
        CREATE TABLE child (x INTEGER REFERENCES "Ä");
        """
    )
    assert _references(db) == {("child", "x", "Ä", "upper_key")}


def test_unresolvable_parent_is_passed_through(connector_factory) -> None:
    """A parent that isn't in the database keeps the name the DDL gave it."""
    db = connector_factory(
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, x INTEGER REFERENCES gone);"
    )
    fks = db.get_fks()
    assert fks.referenced_table.tolist() == ["gone"]
    assert fks.referenced_column.isna().all()


def test_generated_columns_are_reported_in_catalog(connector_factory) -> None:
    """PRAGMA table_xinfo reports generated columns; table_info omits them."""
    db = connector_factory(
        """
        CREATE TABLE sales (
            sale_id    INTEGER PRIMARY KEY,
            qty        INTEGER NOT NULL,
            unit_price REAL    NOT NULL,
            revenue    REAL    GENERATED ALWAYS AS (qty * unit_price)       STORED,
            margin     REAL    GENERATED ALWAYS AS (qty * unit_price * 0.25) VIRTUAL
        );
        """
    )
    cols = db.get_columns()
    sales_cols = cols[cols["table_name"] == "sales"]
    assert sales_cols["column_name"].tolist() == [
        "sale_id",
        "qty",
        "unit_price",
        "revenue",
        "margin",
    ]
    assert sales_cols["data_type"].tolist() == [
        "INTEGER",
        "INTEGER",
        "REAL",
        "REAL",
        "REAL",
    ]
    assert sales_cols["is_nullable"].tolist() == [True, False, False, True, True]
    assert sales_cols["ordinal_position"].tolist() == [1, 2, 3, 4, 5]


def test_virtual_table_hidden_columns_are_excluded(connector_factory) -> None:
    """Hidden shadow columns on virtual tables (hidden=1) must not enter the catalog."""
    db = connector_factory("CREATE VIRTUAL TABLE documents USING fts5(title, body);")
    cols = db.get_columns()
    doc_cols = cols[cols["table_name"] == "documents"]
    assert doc_cols["column_name"].tolist() == ["title", "body"]
    assert doc_cols["ordinal_position"].tolist() == [1, 2]


def test_foreign_key_sourced_on_generated_column_is_reported(connector_factory) -> None:
    """Foreign keys sourced on a generated column are kept with their declared column name."""
    db = connector_factory(
        """
        CREATE TABLE parent (id INTEGER PRIMARY KEY);
        CREATE TABLE child (
            a INTEGER,
            b INTEGER,
            gen_id INTEGER GENERATED ALWAYS AS (a + b) STORED REFERENCES parent(id)
        );
        """
    )
    assert _references(db) == {("child", "gen_id", "parent", "id")}


def test_foreign_key_referencing_generated_column_resolves_declared_case(
    connector_factory,
) -> None:
    """REFERENCES parent(GEN_KEY) resolves to the parent's declared column name."""
    db = connector_factory(
        """
        CREATE TABLE parent (
            id INTEGER PRIMARY KEY,
            code TEXT,
            gen_key TEXT GENERATED ALWAYS AS (code) STORED UNIQUE
        );
        CREATE TABLE child (
            id INTEGER PRIMARY KEY,
            ref_val TEXT,
            FOREIGN KEY (ref_val) REFERENCES parent(GEN_KEY)
        );
        """
    )
    assert _references(db) == {("child", "ref_val", "parent", "gen_key")}
