# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Incremental re-ingest: what changes in the catalog when the source changes.

``update_diff_from_existing_schema`` decides which tables and columns are added,
dropped and updated on every scheduled re-ingest, so a wrong answer either loses
catalog entries users were querying or leaves ghosts pointing at columns that no
longer exist. The end-to-end fixture ingest exercises only the *empty database*
case, where diffing has nothing to do — and re-ingest is where the bugs were.

**These run against whichever backend the process was started with.** Assertions
go through :mod:`auto_ontology.catalog.tests.catalog_inspector`, which asks the same
question of both stores, so the file is one specification rather than two. The
switch is read once at import, so covering both means two processes::

    uv run pytest auto_ontology/catalog/tests/test_incremental_ingest.py

CI has to run both. One passing does not imply the other.

Each test owns a throwaway source database, so it can add and drop tables
without disturbing the Pagila fixture. Skips when its stores are unreachable.
"""

from __future__ import annotations

import os
import uuid

import pytest

pytest.importorskip("psycopg")
import psycopg  # noqa: E402

from auto_ontology.catalog.tests import catalog_inspector as catalog  # noqa: E402


def _pg_dsn(dbname: str) -> str:
    user = os.environ["POSTGRES_USER"]
    password = os.environ["POSTGRES_PASSWORD"]
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    return f"postgresql://{user}:{password}@{host}:{port}/{dbname}"


@pytest.fixture(scope="module", autouse=True)
def require_store():
    """Skip unless the *configured* store is actually reachable."""
    if not os.environ.get("POSTGRES_USER"):
        pytest.skip("POSTGRES_* not set (needed for the source database)")

    try:
        from auto_ontology.dal.session import store

        store().query_read("SELECT 1 FROM catalog_database LIMIT 1")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"auto_ontology schema unavailable (alembic upgrade head): {exc}")


@pytest.fixture
def source_db():
    """A throwaway source database this test may freely mutate."""
    name = f"difftest_{uuid.uuid4().hex[:12]}"
    admin = psycopg.connect(_pg_dsn("postgres"), autocommit=True, connect_timeout=5)
    try:
        admin.execute(f'CREATE DATABASE "{name}"')
    finally:
        admin.close()

    with psycopg.connect(_pg_dsn(name), autocommit=True) as conn:
        conn.execute("CREATE SCHEMA IF NOT EXISTS shop")
        conn.execute(
            "CREATE TABLE shop.customer ("
            " customer_id integer PRIMARY KEY,"
            " name varchar(100),"
            " email varchar(200))"
        )
        conn.execute(
            "CREATE TABLE shop.orders ("
            " order_id integer PRIMARY KEY,"
            " customer_id integer REFERENCES shop.customer(customer_id),"
            " total numeric(10,2))"
        )
        conn.execute("INSERT INTO shop.customer VALUES (1, 'ada', 'a@b.c')")
        conn.execute("INSERT INTO shop.orders VALUES (1, 1, 9.99)")

    yield name

    catalog.wipe(name)
    admin = psycopg.connect(_pg_dsn("postgres"), autocommit=True, connect_timeout=5)
    try:
        admin.execute(
            "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
            "WHERE datname = %s AND pid <> pg_backend_pid()",
            (name,),
        )
        admin.execute(f'DROP DATABASE IF EXISTS "{name}"')
    finally:
        admin.close()


def _ingest(database_name: str) -> None:
    from auto_ontology.catalog import ingest_catalog
    from auto_ontology.connectors.registry import create_connector

    connector = create_connector(_pg_dsn(database_name))
    try:
        ingest_catalog(connector)
    finally:
        connector.close()


def _mutate(database_name: str, *statements: str) -> None:
    with psycopg.connect(_pg_dsn(database_name), autocommit=True) as conn:
        for statement in statements:
            conn.execute(statement)


# ---------------------------------------------------------------------------


def test_first_ingest_creates_the_catalog(source_db: str) -> None:
    _ingest(source_db)
    assert catalog.tables(source_db) == {"customer", "orders"}
    assert catalog.columns(source_db, "customer") == {
        "customer_id",
        "name",
        "email",
    }


def test_reingest_without_changes_is_a_no_op(source_db: str) -> None:
    """The scheduler re-ingests every 24h; an unchanged source must not churn.

    Compared on counts, not names: a duplicate-creating bug keeps every name
    and simply doubles the catalog.
    """
    _ingest(source_db)
    before = catalog.entity_counts(source_db)

    _ingest(source_db)
    after = catalog.entity_counts(source_db)

    assert after == before, f"re-ingest changed the catalog: {before} -> {after}"


def test_added_table_appears_with_its_columns(source_db: str) -> None:
    _ingest(source_db)
    _mutate(
        source_db,
        "CREATE TABLE shop.invoice (invoice_id integer PRIMARY KEY, amount numeric)",
    )
    _ingest(source_db)

    assert "invoice" in catalog.tables(source_db)
    assert catalog.columns(source_db, "invoice") == {"invoice_id", "amount"}


def test_dropped_table_is_removed(source_db: str) -> None:
    """A ghost table is worse than a missing one: it is offered to the SQL
    generator, which then writes queries against a relation that is gone."""
    _ingest(source_db)
    _mutate(source_db, "DROP TABLE shop.orders")
    _ingest(source_db)

    assert catalog.tables(source_db) == {"customer"}


def test_dropped_table_takes_its_columns_with_it(source_db: str) -> None:
    _ingest(source_db)
    _mutate(source_db, "DROP TABLE shop.orders")
    _ingest(source_db)

    assert catalog.columns(source_db, "orders") == set()


def test_added_column_appears(source_db: str) -> None:
    _ingest(source_db)
    _mutate(source_db, "ALTER TABLE shop.customer ADD COLUMN phone varchar(40)")
    _ingest(source_db)

    assert "phone" in catalog.columns(source_db, "customer")


def test_dropped_column_is_removed(source_db: str) -> None:
    _ingest(source_db)
    _mutate(source_db, "ALTER TABLE shop.customer DROP COLUMN email")
    _ingest(source_db)

    assert catalog.columns(source_db, "customer") == {"customer_id", "name"}


def test_renamed_column_is_add_plus_drop(source_db: str) -> None:
    """Rename is indistinguishable from drop+add at the catalog level.

    Worth pinning: it means any curation attached to the old column — a
    description, a ColumnAttribute — does not follow the rename.
    """
    _ingest(source_db)
    _mutate(source_db, "ALTER TABLE shop.customer RENAME COLUMN email TO contact")
    _ingest(source_db)

    names = catalog.columns(source_db, "customer")
    assert "contact" in names
    assert "email" not in names


def test_changed_column_type_is_updated_in_place(source_db: str) -> None:
    """The column keeps its identity; only data_type changes."""
    _ingest(source_db)
    before_id = catalog.column_property(source_db, "customer", "name", "id")
    assert (
        catalog.column_property(source_db, "customer", "name", "data_type")
        == "character varying"
    )

    _mutate(source_db, "ALTER TABLE shop.customer ALTER COLUMN name TYPE text")
    _ingest(source_db)

    assert catalog.column_property(source_db, "customer", "name", "data_type") == "text"
    assert catalog.column_property(source_db, "customer", "name", "id") == before_id, (
        "the column was recreated rather than updated; anything referencing its "
        "id would now dangle"
    )


def test_added_then_dropped_table_leaves_no_trace(source_db: str) -> None:
    _ingest(source_db)
    baseline = catalog.entity_counts(source_db)

    _mutate(source_db, "CREATE TABLE shop.temp_thing (id integer PRIMARY KEY)")
    _ingest(source_db)
    assert "temp_thing" in catalog.tables(source_db)

    _mutate(source_db, "DROP TABLE shop.temp_thing")
    _ingest(source_db)

    assert "temp_thing" not in catalog.tables(source_db)
    assert catalog.entity_counts(source_db) == baseline


def test_new_schema_is_picked_up(source_db: str) -> None:
    _ingest(source_db)
    _mutate(
        source_db,
        "CREATE SCHEMA warehouse",
        "CREATE TABLE warehouse.stock (sku text PRIMARY KEY, qty integer)",
    )
    _ingest(source_db)

    assert "stock" in catalog.tables(source_db)


def test_foreign_key_survives_reingest(source_db: str) -> None:
    _ingest(source_db)
    _ingest(source_db)

    assert catalog.foreign_keys(source_db) == {("customer_id", "customer_id")}
