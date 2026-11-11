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
import secrets
from collections.abc import Iterator

import psycopg
import pytest
from psycopg import sql

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

    ``constraint_column_usage.table_schema`` was the *referenced* table's
    schema, so joining it to the constraint's own schema dropped every foreign
    key that crosses a schema boundary, and with it the join path between the
    schemas.
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


def test_foreign_keys_on_partitions_are_not_reported(pagila: PostgresDatabase) -> None:
    """Pagila declares ``payment``'s keys on each partition, and
    :meth:`get_tables` hides partitions, so no row may name one."""
    fks = pagila.get_fks()
    on_partitions = fks[
        fks.table_name.str.startswith("payment_p")
        | fks.referenced_table.str.startswith("payment_p")
    ]
    assert on_partitions.empty, on_partitions.values.tolist()


def _admin_connection() -> psycopg.Connection:
    return psycopg.connect(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=os.environ.get("POSTGRES_PORT", "5432"),
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        dbname="pagila",
        autocommit=True,
    )


@pytest.fixture
def fk_probe(pagila: PostgresDatabase) -> Iterator[str]:
    """A throwaway schema holding the key shapes ``information_schema`` got
    wrong, plus the cases the ``pg_constraint`` filters exist for:

    - a composite key (``child`` -> ``parent``);
    - two tables whose constraints share a name (``to_film``, ``to_actor``);
    - a table the read-only role is not granted (``hidden``);
    - a partitioned table with a key out of it and a key into it, which
      Postgres clones onto its partition (``events``, ``event_notes``).

    The name is unique per test, so parallel runs against one fixture database
    cannot drop each other's schema.
    """
    schema = f"ao_test_fk_probe_{secrets.token_hex(4)}"
    ddl = sql.SQL("""
        CREATE SCHEMA {s};
        CREATE TABLE {s}.parent (a int, b int, PRIMARY KEY (a, b));
        CREATE TABLE {s}.child (
            id int PRIMARY KEY, pa int, pb int,
            FOREIGN KEY (pa, pb) REFERENCES {s}.parent (a, b));
        CREATE TABLE {s}.to_film (
            id int PRIMARY KEY,
            film_id int CONSTRAINT shared_name REFERENCES public.film (film_id));
        CREATE TABLE {s}.to_actor (
            id int PRIMARY KEY,
            actor_id int CONSTRAINT shared_name REFERENCES public.actor (actor_id));
        CREATE TABLE {s}.hidden (id int PRIMARY KEY);
        CREATE TABLE {s}.to_hidden (
            id int PRIMARY KEY, hidden_id int REFERENCES {s}.hidden (id));
        CREATE TABLE {s}.events (
            id int PRIMARY KEY, film_id int REFERENCES public.film (film_id)
        ) PARTITION BY RANGE (id);
        CREATE TABLE {s}.events_p1 PARTITION OF {s}.events
            FOR VALUES FROM (0) TO (1000);
        CREATE TABLE {s}.event_notes (
            id int PRIMARY KEY, event_id int REFERENCES {s}.events (id));
    """).format(s=sql.Identifier(schema))
    try:
        # Inside the try: the DDL autocommits statement by statement, so a
        # failure partway leaves a half-built schema the cleanup must drop.
        with _admin_connection() as conn:
            conn.execute(ddl)
        yield schema
    finally:
        with _admin_connection() as conn:
            conn.execute(
                sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
                    sql.Identifier(schema)
                )
            )


@pytest.fixture
def probe_reader(fk_probe: str) -> Iterator[PostgresDatabase]:
    """A connector for a role with ``SELECT`` on everything but ``hidden``.

    The role is cluster-wide, so it is dropped whatever step fails.
    """
    role, password = f"ao_test_reader_{secrets.token_hex(4)}", secrets.token_hex(16)
    ident, schema = sql.Identifier(role), sql.Identifier(fk_probe)
    created = False
    reader = None
    try:
        with _admin_connection() as conn:
            try:
                conn.execute(
                    sql.SQL("CREATE ROLE {} LOGIN PASSWORD {}").format(
                        ident, sql.Literal(password)
                    )
                )
            except psycopg.errors.InsufficientPrivilege:
                pytest.skip("fixture user cannot create roles")
            created = True
            for statement in (
                "GRANT USAGE ON SCHEMA public, {s} TO {r}",
                "GRANT SELECT ON ALL TABLES IN SCHEMA public, {s} TO {r}",
                "REVOKE SELECT ON {s}.hidden FROM {r}",
            ):
                conn.execute(sql.SQL(statement).format(s=schema, r=ident))
        host = os.environ.get("POSTGRES_HOST", "localhost")
        port = os.environ.get("POSTGRES_PORT", "5432")
        reader = create_connector(
            f"postgresql://{role}:{password}@{host}:{port}/pagila"
        )
        yield reader
    finally:
        if reader is not None:
            reader.close()
        if created:
            with _admin_connection() as conn:
                conn.execute(sql.SQL("DROP OWNED BY {}").format(ident))
                conn.execute(sql.SQL("DROP ROLE IF EXISTS {}").format(ident))


def _probe_fks(connector: PostgresDatabase, schema: str) -> list[list[str]]:
    fks = connector.get_fks()
    return fks[fks.table_schema == schema][
        ["table_name", "column_name", "referenced_table", "referenced_column"]
    ].values.tolist()


def _probe_pks(connector: PostgresDatabase, schema: str) -> list[list]:
    pks = connector.get_pks()
    return pks[pks.table_schema == schema][
        ["table_name", "column_name", "ordinal_position"]
    ].values.tolist()


# In key order within each table. ``events_p1`` never appears: the copies of
# ``events``' keys that Postgres clones onto it, and onto ``event_notes`` for
# the key into ``events``, are hidden with the partition.
_PROBE_FKS = [
    ["child", "pa", "parent", "a"],
    ["child", "pb", "parent", "b"],
    ["event_notes", "event_id", "events", "id"],
    ["events", "film_id", "film", "film_id"],
    ["to_actor", "actor_id", "actor", "actor_id"],
    ["to_film", "film_id", "film", "film_id"],
    ["to_hidden", "hidden_id", "hidden", "id"],
]
_PROBE_PKS = [
    ["child", "id", 1],
    ["event_notes", "id", 1],
    ["events", "id", 1],
    ["hidden", "id", 1],
    ["parent", "a", 1],
    ["parent", "b", 2],
    ["to_actor", "id", 1],
    ["to_film", "id", 1],
    ["to_hidden", "id", 1],
]


def test_foreign_keys_are_paired_in_key_order(
    pagila: PostgresDatabase, fk_probe: str
) -> None:
    """A composite key used to come back as all four column combinations, each
    ``shared_name`` constraint was also given the other's target, and keys
    cloned onto a partition leaked through."""
    assert _probe_fks(pagila, fk_probe) == _PROBE_FKS


def test_primary_keys_are_in_key_order_without_partitions(
    pagila: PostgresDatabase, fk_probe: str
) -> None:
    assert _probe_pks(pagila, fk_probe) == _PROBE_PKS


def test_read_only_role_sees_keys_it_can_query(
    probe_reader: PostgresDatabase, fk_probe: str
) -> None:
    """``information_schema`` admits a table only for its owner or a privilege
    other than ``SELECT``, so a read-only ingesting role saw no keys at all.

    It must see every key except those touching ``hidden``, which it cannot
    read: the foreign key into it and its primary key.
    """
    assert _probe_fks(probe_reader, fk_probe) == [
        row for row in _PROBE_FKS if row[2] != "hidden"
    ]
    assert _probe_pks(probe_reader, fk_probe) == [
        row for row in _PROBE_PKS if row[0] != "hidden"
    ]
