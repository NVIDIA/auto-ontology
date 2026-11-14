# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Catalog introspection tests for the Postgres connector.

Runs against the Pagila fixture, which is the only fixture that has a
partitioned table and a materialized view — the two relation kinds this
connector used to drop silently. Skipped when no fixture database is reachable.

Set up with::

    docker compose up -d postgres
    uv run --no-sync python -m dev_tools.fixtures.seed_fixtures
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator

import psycopg
import pytest

from auto_ontology.catalog.constants import TableTypes
from auto_ontology.connectors.postgres import PostgresDatabase
from auto_ontology.connectors.registry import create_connector


@pytest.fixture(scope="module")
def pagila():
    if not os.environ.get("POSTGRES_USER"):
        pytest.skip("POSTGRES_* not set; run dev_tools.fixtures.seed_fixtures first")
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    user = os.environ["POSTGRES_USER"]
    password = os.environ["POSTGRES_PASSWORD"]
    try:
        connector = create_connector(
            f"postgresql://{user}:{password}@{host}:{port}/pagila"
        )
        connector.get_tables()
    except Exception as exc:  # noqa: BLE001 — any failure means "no fixture DB"
        pytest.skip(f"pagila not reachable: {exc}")
    yield connector
    connector.close()


@pytest.fixture(scope="module")
def public_tables(pagila):
    tables = pagila.get_tables()
    return tables[tables.table_schema == "public"]


def test_partitioned_parent_is_catalogued(public_tables) -> None:
    """``payment`` is partitioned. It used to vanish entirely.

    ``relispartition = false`` correctly hid the children, but ``relkind`` did
    not allow ``'p'``, so the parent was dropped too and an entire queryable
    table was missing from the catalog.
    """
    payment = public_tables[public_tables.table_name == "payment"]
    assert len(payment) == 1, "partitioned parent missing from the catalog"
    assert payment.table_type.iloc[0] == TableTypes.BASE_TABLE


def test_partition_children_are_not_catalogued(public_tables) -> None:
    """Children are an implementation detail; listing both double-counts rows."""
    children = public_tables[public_tables.table_name.str.startswith("payment_p")]
    assert children.empty, f"partition children leaked: {list(children.table_name)}"


def test_materialized_view_is_catalogued(public_tables) -> None:
    """``rental_by_category`` is a matview.

    Matviews are absent from ``information_schema.tables``, so driving the query
    from there meant ``TableTypes.MATERIALIZED_VIEW`` could never be produced —
    the ``relkind = 'm'`` branch was unreachable.
    """
    matview = public_tables[public_tables.table_name == "rental_by_category"]
    assert len(matview) == 1, "materialized view missing from the catalog"
    assert matview.table_type.iloc[0] == TableTypes.MATERIALIZED_VIEW


def test_all_three_table_types_are_produced(public_tables) -> None:
    """Every ``TableTypes`` value is reachable, not just ``base table``."""
    assert set(public_tables.table_type) == {
        TableTypes.BASE_TABLE,
        TableTypes.VIEW,
        TableTypes.MATERIALIZED_VIEW,
    }


def test_columns_reach_every_catalogued_relation(pagila, public_tables) -> None:
    """A relation with no columns is useless downstream.

    Matview columns are missing from ``information_schema.columns`` too, so this
    would have been empty for ``rental_by_category`` even once the table itself
    was listed.
    """
    columns = pagila.get_columns()
    public_columns = columns[columns.table_schema == "public"]
    with_columns = set(public_columns.table_name)
    missing = set(public_tables.table_name) - with_columns
    assert not missing, f"catalogued relations with no columns: {sorted(missing)}"


def test_declared_types_are_reported_not_placeholders(pagila) -> None:
    """Enums, arrays and domains report their real type.

    ``information_schema.columns.data_type`` collapses these to
    ``USER-DEFINED`` and ``ARRAY``, which tells a SQL-generating model nothing.
    """
    columns = pagila.get_columns()
    film = columns[
        (columns.table_schema == "public") & (columns.table_name == "film")
    ].set_index("column_name")

    assert film.loc["rating", "data_type"] == "mpaa_rating"
    assert film.loc["special_features", "data_type"] == "text[]"
    assert "USER-DEFINED" not in set(columns.data_type)
    assert "ARRAY" not in set(columns.data_type)


def test_system_schemas_are_excluded(pagila) -> None:
    schemas = set(pagila.get_tables().table_schema)
    assert schemas == {"public", "analytics"}, schemas


def test_nullability_is_a_real_boolean(pagila) -> None:
    """Not the ``'YES'``/``'NO'`` text ``information_schema`` reports.

    Both spellings are truthy, so leaking them makes every column read as
    nullable — the bug this connector's ``NOT a.attnotnull`` avoids by never
    producing a string in the first place.
    """
    columns = pagila.get_columns()
    assert columns.is_nullable.dtype == bool
    assert set(columns.is_nullable) <= {True, False}

    film = columns[
        (columns.table_schema == "public") & (columns.table_name == "film")
    ].set_index("column_name")
    assert bool(film.loc["film_id", "is_nullable"]) is False, "film_id is NOT NULL"
    assert bool(film.loc["description", "is_nullable"]) is True

    # The real assertion: a mix, not a column of all-True. The bug's signature
    # was every column agreeing.
    assert 0 < int(columns.is_nullable.sum()) < len(columns)


def test_foreign_key_into_another_schema_is_reported(pagila: PostgresDatabase) -> None:
    """``analytics.film_performance.film_id`` references ``public.film``.

    ``constraint_column_usage.table_schema`` is the *referenced* table's schema,
    so joining it to the constraint's own schema dropped every foreign key that
    crosses a schema boundary, and with it the join path between the schemas.
    """
    fks = pagila.get_fks()
    film_performance = fks[
        (fks.table_schema == "analytics") & (fks.table_name == "film_performance")
    ]
    assert film_performance[
        ["column_name", "referenced_schema", "referenced_table", "referenced_column"]
    ].values.tolist() == [["film_id", "public", "film", "film_id"]]


def test_cross_and_same_schema_foreign_keys_on_one_table(
    pagila: PostgresDatabase,
) -> None:
    """``analytics.category_rollup`` has one of each: ``category_id`` into
    ``public.category`` and the ``parent_rollup_id`` self-reference."""
    fks = pagila.get_fks()
    rollup = fks[
        (fks.table_schema == "analytics") & (fks.table_name == "category_rollup")
    ]
    assert rollup[
        ["column_name", "referenced_schema", "referenced_table"]
    ].values.tolist() == [
        ["category_id", "public", "category"],
        ["parent_rollup_id", "analytics", "category_rollup"],
    ]


# ----------------------------------------------------------------------------
# Foreign-key shapes the Pagila fixture does not have. Each test gets a scratch
# database so the shared fixture (and the DAL goldens recorded from it) stay put.
# ----------------------------------------------------------------------------

_FK_SHAPES_DDL = """
CREATE SCHEMA s;
CREATE SCHEMA w;
CREATE TABLE w.warehouses (id int PRIMARY KEY);
CREATE TABLE s.customers (id int PRIMARY KEY);
-- Same constraint name on two tables of one schema: legal, names are per table.
CREATE TABLE s.orders (
    cust_id int CONSTRAINT fk_ref REFERENCES s.customers (id)
);
CREATE TABLE s.shipments (
    warehouse_id int CONSTRAINT fk_ref REFERENCES w.warehouses (id)
);
-- A composite key that crosses schemas, declared in a different column order
-- from the referenced table's.
CREATE TABLE w.bins (bin_no int, warehouse_id int, PRIMARY KEY (warehouse_id, bin_no));
CREATE TABLE s.picks (
    pick_bin int,
    pick_warehouse int,
    FOREIGN KEY (pick_warehouse, pick_bin) REFERENCES w.bins (warehouse_id, bin_no)
);
"""


def _admin_dsn(database: str) -> str:
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    user = os.environ["POSTGRES_USER"]
    password = os.environ["POSTGRES_PASSWORD"]
    return f"postgresql://{user}:{password}@{host}:{port}/{database}"


@pytest.fixture(scope="module")
def fk_shapes_db() -> Iterator[str]:
    """A scratch database with the shapes above, plus a SELECT-only role."""
    if not os.environ.get("POSTGRES_USER"):
        pytest.skip("POSTGRES_* not set")
    name = f"fk_shapes_{uuid.uuid4().hex[:8]}"
    reader = f"{name}_reader"
    maintenance = os.environ.get("POSTGRES_DATABASE", "postgres")
    try:
        admin = psycopg.connect(_admin_dsn(maintenance), autocommit=True)
    except Exception as exc:  # noqa: BLE001 — any failure means "no fixture DB"
        pytest.skip(f"postgres not reachable: {exc}")
    with admin:
        # Both CREATEs sit inside the try: the DROP ... IF EXISTS cleanup
        # tolerates a partial setup, so a failed CREATE ROLE can't leak the
        # scratch database.
        try:
            admin.execute(f'CREATE DATABASE "{name}"')
            admin.execute(f"CREATE ROLE \"{reader}\" LOGIN PASSWORD 'reader'")
            with psycopg.connect(_admin_dsn(name), autocommit=True) as scratch:
                scratch.execute(_FK_SHAPES_DDL)
                for schema in ("s", "w"):
                    scratch.execute(f'GRANT USAGE ON SCHEMA {schema} TO "{reader}"')
                    scratch.execute(
                        f'GRANT SELECT ON ALL TABLES IN SCHEMA {schema} TO "{reader}"'
                    )
            yield name
        finally:
            admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
            admin.execute(f'DROP ROLE IF EXISTS "{reader}"')


def _fk_rows(connector: PostgresDatabase) -> list[list[str]]:
    fks = connector.get_fks()
    return fks[
        [
            "table_schema",
            "table_name",
            "column_name",
            "referenced_schema",
            "referenced_table",
            "referenced_column",
        ]
    ].values.tolist()


_EXPECTED_FK_ROWS = [
    ["s", "orders", "cust_id", "s", "customers", "id"],
    ["s", "picks", "pick_bin", "w", "bins", "bin_no"],
    ["s", "picks", "pick_warehouse", "w", "bins", "warehouse_id"],
    ["s", "shipments", "warehouse_id", "w", "warehouses", "id"],
]


def test_foreign_keys_pair_by_owning_constraint_and_position(fk_shapes_db: str) -> None:
    """Two tables' ``fk_ref`` constraints stay separate, and the composite key
    pairs each column with the one it references -- not the cross product."""
    connector = create_connector(_admin_dsn(fk_shapes_db))
    try:
        assert _fk_rows(connector) == _EXPECTED_FK_ROWS
    finally:
        connector.close()


def test_select_only_role_sees_foreign_keys(fk_shapes_db: str) -> None:
    """``information_schema.table_constraints`` hides constraints from a role
    that holds only ``SELECT``; a read-only catalog user must still see them."""
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    connector = create_connector(
        f"postgresql://{fk_shapes_db}_reader:reader@{host}:{port}/{fk_shapes_db}"
    )
    try:
        assert _fk_rows(connector) == _EXPECTED_FK_ROWS
    finally:
        connector.close()
