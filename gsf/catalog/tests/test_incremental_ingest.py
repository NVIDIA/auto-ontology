# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Incremental re-ingest: what changes in the catalog when the source changes.

``update_diff_from_existing_schema`` is the most consequential untested code in
the write path. It decides which tables and columns are added, dropped and
updated on every scheduled re-ingest, so a wrong answer either loses catalog
entries users were querying or leaves ghosts pointing at columns that no longer
exist. The end-to-end fixture ingest exercises only the *empty database* case,
where diffing has nothing to do.

These tests exist **before** the Postgres rewrite deliberately: they are the
specification the Postgres implementation has to satisfy, written against the
Neo4j behaviour that is being preserved. Written afterwards they would only
describe whatever the new code happened to do.

Each test owns a throwaway source database, so it can add and drop tables
without disturbing the Pagila fixture.

Needs both stores and skips without them::

    docker compose up -d postgres neo4j
"""

from __future__ import annotations

import os
import uuid

import pytest

pytest.importorskip("psycopg")
import psycopg  # noqa: E402


def _pg_dsn(dbname: str) -> str:
    user = os.environ["POSTGRES_USER"]
    password = os.environ["POSTGRES_PASSWORD"]
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    return f"postgresql://{user}:{password}@{host}:{port}/{dbname}"


@pytest.fixture(scope="module")
def _require_stores():
    if not (os.environ.get("POSTGRES_USER") and os.environ.get("NEO4J_URI")):
        pytest.skip("POSTGRES_* and NEO4J_URI must both be set")
    try:
        from gsf.catalog.store.neo4j.connection import get_neo4j_conn

        get_neo4j_conn().query_read("RETURN 1 AS ok")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"neo4j unavailable: {exc}")


@pytest.fixture
def source_db(_require_stores):
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

    _wipe_graph(name)
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


def _wipe_graph(database_name: str) -> None:
    from gsf.catalog.store.neo4j.connection import get_neo4j_conn

    get_neo4j_conn().query_write(
        "MATCH (d:Database {name:$name}) "
        "CALL apoc.path.subgraphNodes(d, {}) YIELD node "
        "DETACH DELETE node",
        {"name": database_name},
    )


def _ingest(database_name: str) -> None:
    from gsf.catalog import ingest_catalog
    from gsf.connectors.registry import create_connector

    connector = create_connector(_pg_dsn(database_name))
    try:
        ingest_catalog(connector)
    finally:
        connector.close()


def _mutate(database_name: str, *statements: str) -> None:
    with psycopg.connect(_pg_dsn(database_name), autocommit=True) as conn:
        for statement in statements:
            conn.execute(statement)


def _tables(database_name: str) -> set[str]:
    from gsf.catalog.store.neo4j.connection import get_neo4j_conn

    rows = get_neo4j_conn().query_read(
        "MATCH (:Database {name:$name})-[:CONTAINS]->(:Schema)"
        "-[:CONTAINS]->(t:Table) RETURN t.name AS name",
        {"name": database_name},
    )
    return {r["name"] for r in rows}


def _columns(database_name: str, table: str) -> set[str]:
    from gsf.catalog.store.neo4j.connection import get_neo4j_conn

    rows = get_neo4j_conn().query_read(
        "MATCH (:Database {name:$name})-[:CONTAINS]->(:Schema)"
        "-[:CONTAINS]->(t:Table {name:$table})-[:CONTAINS]->(c:Column) "
        "RETURN c.name AS name",
        {"name": database_name, "table": table},
    )
    return {r["name"] for r in rows}


def _column_prop(database_name: str, table: str, column: str, prop: str):
    from gsf.catalog.store.neo4j.connection import get_neo4j_conn

    rows = get_neo4j_conn().query_read(
        f"MATCH (:Database {{name:$name}})-[:CONTAINS]->(:Schema)"
        f"-[:CONTAINS]->(:Table {{name:$table}})-[:CONTAINS]->(c:Column {{name:$column}}) "
        f"RETURN c.`{prop}` AS value",
        {"name": database_name, "table": table, "column": column},
    )
    return rows[0]["value"] if rows else None


def _counts(database_name: str) -> dict[str, int]:
    from gsf.catalog.store.neo4j.connection import get_neo4j_conn

    rows = get_neo4j_conn().query_read(
        "MATCH (d:Database {name:$name}) "
        "CALL apoc.path.subgraphNodes(d, {}) YIELD node "
        "RETURN labels(node)[0] AS label, count(*) AS n",
        {"name": database_name},
    )
    return {r["label"]: r["n"] for r in rows}


# ---------------------------------------------------------------------------


def test_first_ingest_creates_the_catalog(source_db: str) -> None:
    _ingest(source_db)
    assert _tables(source_db) == {"customer", "orders"}
    assert _columns(source_db, "customer") == {"customer_id", "name", "email"}


def test_reingest_without_changes_is_a_no_op(source_db: str) -> None:
    """The scheduler re-ingests every 24h; an unchanged source must not churn.

    Node counts are compared rather than just names, because a duplicate-
    creating bug keeps every name and simply doubles the graph.
    """
    _ingest(source_db)
    before = _counts(source_db)

    _ingest(source_db)
    after = _counts(source_db)

    assert after == before, f"re-ingest changed the graph: {before} -> {after}"


def test_added_table_appears_with_its_columns(source_db: str) -> None:
    _ingest(source_db)
    _mutate(
        source_db,
        "CREATE TABLE shop.invoice (invoice_id integer PRIMARY KEY, amount numeric)",
    )
    _ingest(source_db)

    assert "invoice" in _tables(source_db)
    assert _columns(source_db, "invoice") == {"invoice_id", "amount"}


def test_dropped_table_is_removed(source_db: str) -> None:
    """A ghost table is worse than a missing one: it is offered to the SQL
    generator, which then writes queries against a relation that is gone."""
    _ingest(source_db)
    _mutate(source_db, "DROP TABLE shop.orders")
    _ingest(source_db)

    assert _tables(source_db) == {"customer"}


def test_dropped_table_takes_its_columns_with_it(source_db: str) -> None:
    _ingest(source_db)
    _mutate(source_db, "DROP TABLE shop.orders")
    _ingest(source_db)

    assert _columns(source_db, "orders") == set()


def test_added_column_appears(source_db: str) -> None:
    _ingest(source_db)
    _mutate(source_db, "ALTER TABLE shop.customer ADD COLUMN phone varchar(40)")
    _ingest(source_db)

    assert "phone" in _columns(source_db, "customer")


def test_dropped_column_is_removed(source_db: str) -> None:
    _ingest(source_db)
    _mutate(source_db, "ALTER TABLE shop.customer DROP COLUMN email")
    _ingest(source_db)

    assert _columns(source_db, "customer") == {"customer_id", "name"}


def test_renamed_column_is_add_plus_drop(source_db: str) -> None:
    """Rename is indistinguishable from drop+add at the catalog level.

    Worth pinning: it means any curation attached to the old column — a
    description, a ColumnAttribute — does not follow the rename.
    """
    _ingest(source_db)
    _mutate(source_db, "ALTER TABLE shop.customer RENAME COLUMN email TO contact")
    _ingest(source_db)

    columns = _columns(source_db, "customer")
    assert "contact" in columns
    assert "email" not in columns


def test_changed_column_type_is_updated_in_place(source_db: str) -> None:
    """The column keeps its identity; only data_type changes."""
    _ingest(source_db)
    before_id = _column_prop(source_db, "customer", "name", "id")
    assert _column_prop(source_db, "customer", "name", "data_type") == (
        "character varying"
    )

    _mutate(source_db, "ALTER TABLE shop.customer ALTER COLUMN name TYPE text")
    _ingest(source_db)

    assert _column_prop(source_db, "customer", "name", "data_type") == "text"
    assert _column_prop(source_db, "customer", "name", "id") == before_id, (
        "the column was recreated rather than updated; anything referencing its "
        "id would now dangle"
    )


def test_added_then_dropped_table_leaves_no_trace(source_db: str) -> None:
    _ingest(source_db)
    baseline = _counts(source_db)

    _mutate(source_db, "CREATE TABLE shop.temp_thing (id integer PRIMARY KEY)")
    _ingest(source_db)
    assert "temp_thing" in _tables(source_db)

    _mutate(source_db, "DROP TABLE shop.temp_thing")
    _ingest(source_db)

    assert "temp_thing" not in _tables(source_db)
    assert _counts(source_db) == baseline


def test_new_schema_is_picked_up(source_db: str) -> None:
    _ingest(source_db)
    _mutate(
        source_db,
        "CREATE SCHEMA warehouse",
        "CREATE TABLE warehouse.stock (sku text PRIMARY KEY, qty integer)",
    )
    _ingest(source_db)

    assert "stock" in _tables(source_db)


def test_foreign_key_survives_reingest(source_db: str) -> None:
    from gsf.catalog.store.neo4j.connection import get_neo4j_conn

    _ingest(source_db)
    _ingest(source_db)

    rows = get_neo4j_conn().query_read(
        "MATCH (:Database {name:$name})-[:CONTAINS]->(:Schema)-[:CONTAINS]->"
        "(:Table)-[:CONTAINS]->(c:Column)-[:FOREIGN_KEY]->(t:Column) "
        "RETURN c.name AS src, t.name AS dst",
        {"name": source_db},
    )
    assert {(r["src"], r["dst"]) for r in rows} == {("customer_id", "customer_id")}
