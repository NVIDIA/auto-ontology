# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A composite primary key is stored in key order, whatever order its rows arrive in."""

from __future__ import annotations

import os
import sqlite3
import uuid
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("psycopg")

from sqlalchemy import select  # noqa: E402
from sqlalchemy.exc import OperationalError  # noqa: E402

from auto_ontology.catalog.store.rows import upsert_row  # noqa: E402
from auto_ontology.catalog.write import populate_pks  # noqa: E402
from auto_ontology.connectors.sqlite import SQLiteDatabase  # noqa: E402
from auto_ontology.dal import schema as s  # noqa: E402
from auto_ontology.dal.session import store  # noqa: E402

_PK_COLUMNS = ["table_schema", "table_name", "column_name", "ordinal_position"]


@pytest.fixture(scope="module", autouse=True)
def require_schema() -> None:
    if not os.environ.get("POSTGRES_USER"):
        pytest.skip("POSTGRES_* not set")
    try:
        store().query_read("SELECT 1 FROM catalog_database LIMIT 1")
    except OperationalError as exc:
        pytest.skip(f"catalog database not reachable: {exc}")


@pytest.fixture
def db_name() -> Iterator[str]:
    name = f"pk-{uuid.uuid4().hex[:8]}"
    yield name
    store().query_write(
        s.catalog_database.delete().where(s.catalog_database.c.name == name)
    )


def _tables(db_name: str, schema: str, *names: str) -> dict[str, str]:
    db = upsert_row("Database", {"name": db_name}, {"name": db_name})
    schema_id = upsert_row("Schema", {"name": schema}, {"name": schema}, parent_id=db)
    return {
        name: upsert_row("Table", {"name": name}, {"name": name}, parent_id=schema_id)
        for name in names
    }


def _stored_pk(table_id: str) -> list[str] | None:
    rows = store().query_read(
        select(s.catalog_table.c.pk).where(s.catalog_table.c.id == table_id)
    )
    return rows[0]["pk"]


def test_a_key_listed_alphabetically_is_stored_in_key_order(db_name: str) -> None:
    """Snowflake's ``SHOW PRIMARY KEYS`` lists a table's key columns by name.

    These are the rows it returned for TPC-H, whose ``LINEITEM`` is keyed
    ``(L_ORDERKEY, L_LINENUMBER)``. ``PARTSUPP``'s key happens to be in
    alphabetical order already.
    """
    ids = _tables(db_name, "TPCH_SF1", "LINEITEM", "PARTSUPP")
    pks = pd.DataFrame(
        [
            ("TPCH_SF1", "LINEITEM", "L_LINENUMBER", 2),
            ("TPCH_SF1", "LINEITEM", "L_ORDERKEY", 1),
            ("TPCH_SF1", "PARTSUPP", "PS_PARTKEY", 1),
            ("TPCH_SF1", "PARTSUPP", "PS_SUPPKEY", 2),
        ],
        columns=_PK_COLUMNS,
    )

    populate_pks(pks, db_name)

    assert _stored_pk(ids["LINEITEM"]) == ["L_ORDERKEY", "L_LINENUMBER"]
    assert _stored_pk(ids["PARTSUPP"]) == ["PS_PARTKEY", "PS_SUPPKEY"]


def test_a_sqlite_key_declared_out_of_column_order_is_stored_in_key_order(
    db_name: str, tmp_path: Path
) -> None:
    """SQLite reports a table's key columns in the table's column order."""
    path = tmp_path / "shop.sqlite"
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE line (item_id INTEGER, order_id INTEGER, "
        "PRIMARY KEY (order_id, item_id))"
    )
    conn.commit()
    conn.close()
    ids = _tables(db_name, "main", "line")

    populate_pks(SQLiteDatabase(f"sqlite:///{path}").get_pks(), db_name)

    assert _stored_pk(ids["line"]) == ["order_id", "item_id"]


def test_rows_without_a_position_keep_the_order_they_arrived_in(db_name: str) -> None:
    ids = _tables(db_name, "s", "orders")
    pks = pd.DataFrame(
        [("s", "orders", "region"), ("s", "orders", "id")],
        columns=_PK_COLUMNS[:3],
    )

    populate_pks(pks, db_name)

    assert _stored_pk(ids["orders"]) == ["region", "id"]
