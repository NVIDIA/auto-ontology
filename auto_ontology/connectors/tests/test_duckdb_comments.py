# SPDX-FileCopyrightText: Copyright (c) 2026, Infinidatum LLC.
# SPDX-License-Identifier: Apache-2.0

"""``DuckDBDatabase`` reports ``COMMENT ON`` text as descriptions.

A documented DuckDB source used to reach the catalog with every description
blank, so compilation asked a model to describe columns the source had
already described.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import duckdb
import pytest

from auto_ontology.catalog.extract import data_for_populate_tabular
from auto_ontology.connectors.duckdb import DuckDBDatabase

_DDL = """
CREATE SCHEMA shop;
CREATE TABLE shop.customer (id INTEGER, name VARCHAR);
CREATE TABLE shop.orders (id INTEGER, customer_id INTEGER, total DOUBLE);
CREATE VIEW shop.big_orders AS SELECT * FROM shop.orders WHERE total > 100;
COMMENT ON TABLE shop.customer IS 'People who buy from the shop.';
COMMENT ON VIEW shop.big_orders IS 'Orders over 100.';
COMMENT ON COLUMN shop.orders.total IS 'Order total in dollars.';
"""


@pytest.fixture
def db(tmp_path: Path) -> Iterator[DuckDBDatabase]:
    path = tmp_path / "store.duckdb"
    conn = duckdb.connect(str(path))
    conn.execute(_DDL)
    conn.close()
    database = DuckDBDatabase(str(path))
    yield database
    database.close()


def test_table_and_view_comments_are_descriptions(db: DuckDBDatabase) -> None:
    tables = db.get_tables().set_index("table_name")["description"]
    assert tables["customer"] == "People who buy from the shop."
    assert tables["big_orders"] == "Orders over 100."
    assert tables.isna()["orders"]


def test_column_comment_is_description(db: DuckDBDatabase) -> None:
    cols = db.get_columns()
    orders = cols[cols.table_name == "orders"].set_index("column_name")
    assert orders.loc["total", "description"] == "Order total in dollars."
    assert orders["description"].isna()["customer_id"]
    assert list(orders.index) == ["id", "customer_id", "total"]


def test_undocumented_source_has_blank_descriptions(tmp_path: Path) -> None:
    path = tmp_path / "bare.duckdb"
    duckdb.connect(str(path)).execute("CREATE TABLE t (a INTEGER)").close()
    with DuckDBDatabase(str(path)) as bare:
        assert bare.get_columns()["description"].isna().all()
        assert bare.get_tables()["description"].isna().all()


def test_descriptions_survive_catalog_normalisation(db: DuckDBDatabase) -> None:
    data = data_for_populate_tabular(db)
    cols = data["columns"]
    total = cols[(cols.table_name == "orders") & (cols.column_name == "total")]
    assert total["description"].iloc[0] == "Order total in dollars."
    tables = data["tables"].set_index("table_name")["description"]
    assert tables["customer"] == "People who buy from the shop."
