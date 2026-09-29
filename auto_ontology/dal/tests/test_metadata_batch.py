# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Batched description / sample-value writes.

``apply_metadata_batch`` has no caller inside Auto Ontology -- it is DAL surface, kept
because it is part of the published interface -- so nothing else exercises it.
That is precisely why it is tested here: it was rewritten from a row-at-a-time
loop into two ``UPDATE ... FROM (VALUES ...)`` statements, and a set-based
rewrite is exactly the kind that can start matching more rows than it should
without any existing test noticing.

The scoping cases carry the weight. Two databases with identically named
schemas, tables and columns is the normal state of a Auto Ontology deployment, and the
statement's ``WHERE`` is the only thing keeping one batch out of the other's
catalog. ``coalesce`` direction gets its own tests for the same reason: the
whole point of the function is that a curated description outlives a batch that
has nothing to say about it.
"""

from __future__ import annotations

import os
import uuid

import pytest

pytest.importorskip("sqlalchemy")

from sqlalchemy import select  # noqa: E402

from auto_ontology.dal import schema as s  # noqa: E402
from auto_ontology.dal.datasources import apply_metadata_batch  # noqa: E402
from auto_ontology.dal.session import store  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def require_schema():
    if not os.environ.get("POSTGRES_USER"):
        pytest.skip("POSTGRES_* not set")
    try:
        store().query_read("SELECT 1 FROM catalog_column LIMIT 1")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"auto_ontology schema unavailable (alembic upgrade head): {exc}")


def _add(table, **values) -> str:
    return store().query_write(table.insert().values(**values).returning(table.c.id))[
        0
    ]["id"]


def _row(table, row_id: str) -> dict:
    return dict(store().query_read(select(table).where(table.c.id == row_id))[0])


class _Catalog:
    """One database, ``public.orders(id, total)`` and ``public.items(id)``.

    ``items.id`` shares a name with ``orders.id`` on purpose: a column update
    that forgets to qualify by table would hit both.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.db = _add(s.catalog_database, name=name)
        self.schema = _add(s.catalog_schema, database_id=self.db, name="public")
        self.orders = _add(s.catalog_table, schema_id=self.schema, name="orders")
        self.items = _add(s.catalog_table, schema_id=self.schema, name="items")
        self.orders_id = _add(s.catalog_column, table_id=self.orders, name="id")
        self.orders_total = _add(s.catalog_column, table_id=self.orders, name="total")
        self.items_id = _add(s.catalog_column, table_id=self.items, name="id")


@pytest.fixture
def catalogs():
    """Two catalogs with identical names inside them."""
    prefix = f"mdb-{uuid.uuid4().hex[:8]}"
    target = _Catalog(f"{prefix}-target")
    other = _Catalog(f"{prefix}-other")
    yield target, other
    for db in (target.db, other.db):
        store().query_write(
            s.catalog_database.delete().where(s.catalog_database.c.id == db)
        )


def test_fills_empty_descriptions(catalogs):
    target, _ = catalogs
    apply_metadata_batch(
        target.name,
        [{"table_name": "orders", "description": "customer orders"}],
        [
            {
                "table_name": "orders",
                "column_name": "total",
                "description": "order total",
                "sample_values": "1,2,3",
            }
        ],
    )

    assert _row(s.catalog_table, target.orders)["description"] == "customer orders"
    column = _row(s.catalog_column, target.orders_total)
    assert column["description"] == "order total"
    assert column["sample_values"] == "1,2,3"


def test_curated_values_survive_a_batch_with_nothing_to_say(catalogs):
    """The direction of ``coalesce``: absent means "leave it", not "clear it"."""
    target, _ = catalogs
    apply_metadata_batch(
        target.name,
        [{"table_name": "orders", "description": "curated"}],
        [
            {
                "table_name": "orders",
                "column_name": "total",
                "description": "curated",
                "sample_values": "a,b",
            }
        ],
    )

    # A second batch naming the same rows, carrying nothing for them.
    apply_metadata_batch(
        target.name,
        [{"table_name": "orders"}],
        [{"table_name": "orders", "column_name": "total"}],
    )

    assert _row(s.catalog_table, target.orders)["description"] == "curated"
    column = _row(s.catalog_column, target.orders_total)
    assert column["description"] == "curated"
    assert column["sample_values"] == "a,b"


def test_columns_are_scoped_to_their_table(catalogs):
    """``orders.id`` and ``items.id`` share a name and must not share a write."""
    target, _ = catalogs
    apply_metadata_batch(
        target.name,
        [],
        [{"table_name": "orders", "column_name": "id", "description": "order id"}],
    )

    assert _row(s.catalog_column, target.orders_id)["description"] == "order id"
    assert _row(s.catalog_column, target.items_id)["description"] is None


def test_never_crosses_a_database(catalogs):
    """The other catalog has the same names and must be untouched."""
    target, other = catalogs
    apply_metadata_batch(
        target.name,
        [
            {"table_name": "orders", "description": "t"},
            {"table_name": "items", "description": "t"},
        ],
        [
            {"table_name": "orders", "column_name": "id", "description": "c"},
            {"table_name": "items", "column_name": "id", "description": "c"},
        ],
    )

    for row_id in (other.orders, other.items):
        assert _row(s.catalog_table, row_id)["description"] is None
    for row_id in (other.orders_id, other.items_id):
        assert _row(s.catalog_column, row_id)["description"] is None


def test_multi_row_batch_applies_to_every_row(catalogs):
    """The set-based form has to write all of them, not just the first."""
    target, _ = catalogs
    apply_metadata_batch(
        target.name,
        [
            {"table_name": "orders", "description": "orders desc"},
            {"table_name": "items", "description": "items desc"},
        ],
        [
            {"table_name": "orders", "column_name": "id", "description": "oid"},
            {"table_name": "orders", "column_name": "total", "description": "tot"},
            {"table_name": "items", "column_name": "id", "description": "iid"},
        ],
    )

    assert _row(s.catalog_table, target.orders)["description"] == "orders desc"
    assert _row(s.catalog_table, target.items)["description"] == "items desc"
    assert _row(s.catalog_column, target.orders_id)["description"] == "oid"
    assert _row(s.catalog_column, target.orders_total)["description"] == "tot"
    assert _row(s.catalog_column, target.items_id)["description"] == "iid"


@pytest.mark.parametrize("empty", [[], None])
def test_empty_batch_is_a_no_op(catalogs, empty):
    """No rows must mean no statement, not a statement matching everything."""
    target, _ = catalogs
    apply_metadata_batch(
        target.name, [{"table_name": "orders", "description": "d"}], []
    )
    apply_metadata_batch(target.name, empty, empty)

    assert _row(s.catalog_table, target.orders)["description"] == "d"


def test_unknown_names_are_ignored(catalogs):
    """A batch naming a table that is not in the catalog writes nothing."""
    target, _ = catalogs
    apply_metadata_batch(
        target.name,
        [{"table_name": "not_a_table", "description": "x"}],
        [{"table_name": "not_a_table", "column_name": "id", "description": "x"}],
    )

    assert _row(s.catalog_table, target.orders)["description"] is None
