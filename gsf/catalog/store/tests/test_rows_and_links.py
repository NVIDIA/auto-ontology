# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The generic node/edge layer the rest of the Postgres write path sits on.

Tested directly rather than through the selector, because the remaining store
modules do not have Postgres implementations yet — this is the foundation, and
it needs to be right before anything is built on it.

The cases that matter are the ones where the property-graph assumption and the
relational schema disagree: ``CONTAINS`` being a parent column rather than a
row, and the parser's id winning on match so ids stay stable across re-ingests.

Needs a migrated database and skips without one::

    uv run alembic upgrade head
"""

from __future__ import annotations

import os
import uuid

import pytest

pytest.importorskip("sqlalchemy")

from sqlalchemy import select  # noqa: E402

from gsf.catalog.constants import Edges, Labels, Props  # noqa: E402
from gsf.catalog.model.node import CatalogNode  # noqa: E402
from gsf.catalog.store import queries as pg_queries  # noqa: E402
from gsf.catalog.store import registry  # noqa: E402
from gsf.catalog.store.rows import resolve_id, upsert_row  # noqa: E402
from gsf.dal import schema as s  # noqa: E402
from gsf.dal.session import store  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def require_schema():
    if not os.environ.get("POSTGRES_USER"):
        pytest.skip("POSTGRES_* not set")
    try:
        store().query_read("SELECT 1 FROM catalog_database LIMIT 1")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"gsf schema unavailable (run alembic upgrade head): {exc}")


@pytest.fixture(autouse=True)
def clean():
    """Databases cascade, so removing them clears everything below."""
    yield
    store().query_write(
        s.catalog_database.delete().where(s.catalog_database.c.name.like("t-%"))
    )


def _db_name() -> str:
    return f"t-{uuid.uuid4().hex[:10]}"


def _node(label, props, match):
    return CatalogNode(
        name=props.get("name", ""), label=label, props=props, match_props=match
    )


# --------------------------------------------------------------------------
# Node upserts
# --------------------------------------------------------------------------


def test_insert_then_match_returns_the_same_row() -> None:
    name = _db_name()
    first = upsert_row(Labels.DB, {"name": name}, {"name": name})
    second = upsert_row(Labels.DB, {"name": name}, {"name": name})
    assert first == second

    rows = store().query_read(
        s.catalog_database.select().where(s.catalog_database.c.name == name)
    )
    assert len(rows) == 1, "the natural key did not dedupe"


def test_stored_id_survives_a_match_with_a_different_incoming_id() -> None:
    """Deliberately never overwrites a stored id — see rows.upsert_row.

    An incoming id must not overwrite the matched
    node's id with the parser's. Harmless in a property graph, where
    relationships bind to internal nodes; here ``id`` is the primary key and
    other rows reference it, so overwriting violates their foreign keys.

    Nothing is lost: the parser resolves ids out of the store before writing,
    so for Table and Column the incoming id already *is* the stored one.
    """
    name = _db_name()
    original = upsert_row(Labels.DB, {"name": name}, {"name": name})

    returned = upsert_row(
        Labels.DB, {"name": name}, {"name": name, "id": str(uuid.uuid4())}
    )

    assert returned == original
    rows = store().query_read(
        s.catalog_database.select().where(s.catalog_database.c.name == name)
    )
    assert len(rows) == 1
    assert rows[0]["id"] == original


def test_referencing_rows_survive_a_reingest() -> None:
    """The reason the id is not overwritten, stated as a test."""
    name = _db_name()
    table_id, _ = _column(name, "customer", "customer_id")
    statement = "SELECT customer_id FROM customer"

    def write_sql_edge():
        sql_node = _node(
            Labels.SQL, {"sql_full_query": statement}, {"sql_full_query": statement}
        )
        table = _node(Labels.TABLE, {"id": table_id}, {"id": table_id})
        pg_queries.add_query([(sql_node, table, {Props.SQL_ID: "sql-1"})])

    write_sql_edge()
    write_sql_edge()

    rows = store().query_read(
        s.sql_query_table.select().where(s.sql_query_table.c.table_id == table_id)
    )
    assert len(rows) == 1


def test_unknown_properties_are_dropped_not_rejected() -> None:
    """Unknown keys are dropped; refusing them would fail live ingests."""
    name = _db_name()
    node_id = upsert_row(
        Labels.DB,
        {"name": name},
        {"name": name, "not_a_column": "ignored", "another": 5},
    )
    assert node_id


def test_unknown_label_raises_rather_than_silently_dropping() -> None:
    """A dropped node is a missing catalog entry discovered much later."""
    with pytest.raises(registry.UnknownLabel):
        upsert_row("NotALabel", {"name": "x"}, {"name": "x"})


def test_resolve_id_returns_none_for_absent_node() -> None:
    assert resolve_id(Labels.DB, {"name": _db_name()}) is None


# --------------------------------------------------------------------------
# CONTAINS is a parent column, not a row
#
# Exercised through upsert_row's parent_id, which is the primitive the
# hierarchy writers use. The parent link never reaches add_query:
# add_schemas_edge_batch and merge_schema_edges write it directly, because it
# is the one relationship carried by a column rather than a row.
# --------------------------------------------------------------------------


def _database(name: str) -> str:
    return upsert_row(Labels.DB, {"name": name}, {"name": name})


def _schema(db_name: str, db_id: str, schema_name: str) -> str:
    return upsert_row(
        Labels.SCHEMA,
        {"database_name": db_name, "name": schema_name},
        {"name": schema_name},
        parent_id=db_id,
    )


def test_parent_id_lands_in_the_child_column() -> None:
    name = _db_name()
    db_id = _database(name)
    _schema(name, db_id, "shop")

    rows = store().query_read(
        s.catalog_schema.select().where(s.catalog_schema.c.name == "shop")
    )
    assert len(rows) == 1
    assert rows[0]["database_id"] == db_id


def test_same_schema_name_in_two_databases_stays_separate() -> None:
    """``Schema`` is unique *within* a database, not globally.

    A natural key that forgot the parent would collapse ``public`` from every
    connected database into a single row.
    """
    for _ in range(2):
        name = _db_name()
        _schema(name, _database(name), "public")

    # Scoped to this test's own databases: a real ingest into the same database
    # also creates a `public` schema, and a global count would pick it up.
    rows = store().query_read(
        select(s.catalog_schema.c.id)
        .select_from(
            s.catalog_schema.join(
                s.catalog_database,
                s.catalog_schema.c.database_id == s.catalog_database.c.id,
            )
        )
        .where(
            s.catalog_schema.c.name == "public",
            s.catalog_database.c.name.like("t-%"),
        )
    )
    assert len(rows) == 2, "schemas from different databases were merged"


def test_parent_link_is_idempotent() -> None:
    name = _db_name()
    db_id = _database(name)
    first = _schema(name, db_id, "shop")
    second = _schema(name, db_id, "shop")

    assert first == second
    rows = store().query_read(
        s.catalog_schema.select().where(s.catalog_schema.c.name == "shop")
    )
    assert len(rows) == 1


def _column(db_name: str, table_name: str, column_name: str) -> tuple[str, str]:
    db_id = _database(db_name)
    schema_id = _schema(db_name, db_id, "shop")
    table_id = str(uuid.uuid4())
    upsert_row(
        Labels.TABLE,
        {"id": table_id},
        {"name": table_name, "id": table_id},
        parent_id=schema_id,
    )
    column_id = str(uuid.uuid4())
    upsert_row(
        Labels.COLUMN,
        {"id": column_id},
        {"name": column_name, "id": column_id},
        parent_id=table_id,
    )
    return table_id, column_id


def test_full_hierarchy_hangs_together() -> None:
    name = _db_name()
    _table_id, column_id = _column(name, "customer", "customer_id")

    rows = store().query_read(
        select(s.catalog_column.c.id)
        .select_from(
            s.catalog_column.join(s.catalog_table)
            .join(s.catalog_schema)
            .join(s.catalog_database)
        )
        .where(s.catalog_database.c.name == name)
    )
    assert [r["id"] for r in rows] == [column_id]


# --------------------------------------------------------------------------
# Real links, in the shape add_query actually receives
# --------------------------------------------------------------------------


def test_sql_edge_becomes_a_row() -> None:
    """``Sql`` → ``Table``: the shape ``sql_parse`` hands to ``add_query``."""
    name = _db_name()
    table_id, _ = _column(name, "customer", "customer_id")

    statement = "SELECT customer_id FROM customer"
    sql_node = _node(
        Labels.SQL, {"sql_full_query": statement}, {"sql_full_query": statement}
    )
    table = _node(Labels.TABLE, {"id": table_id}, {"id": table_id})

    pg_queries.add_query([(sql_node, table, {Props.SQL_ID: "sql-1"})])

    rows = store().query_read(
        s.sql_query_table.select().where(s.sql_query_table.c.table_id == table_id)
    )
    assert len(rows) == 1


def test_repeated_sql_link_does_not_duplicate() -> None:
    name = _db_name()
    table_id, _ = _column(name, "customer", "customer_id")
    statement = "SELECT customer_id FROM customer"
    sql_node = _node(
        Labels.SQL, {"sql_full_query": statement}, {"sql_full_query": statement}
    )
    table = _node(Labels.TABLE, {"id": table_id}, {"id": table_id})
    link = (sql_node, table, {Props.SQL_ID: "sql-1"})

    pg_queries.add_query([link, link])

    rows = store().query_read(
        s.sql_query_table.select().where(s.sql_query_table.c.table_id == table_id)
    )
    assert len(rows) == 1


def test_unknown_link_shape_raises() -> None:
    with pytest.raises(registry.UnknownLink):
        registry.link_spec(Edges.FOREIGN_KEY, Labels.DB, Labels.SCHEMA)


# --------------------------------------------------------------------------
# Deletes
# --------------------------------------------------------------------------


def test_delete_cascades_to_children() -> None:
    """No DETACH step: the schema's cascades do that work."""
    name = _db_name()
    table_id, column_id = _column(name, "customer", "customer_id")
    db_id = resolve_id(Labels.DB, {"name": name})

    store().query_write(
        s.catalog_database.delete().where(s.catalog_database.c.id == db_id)
    )

    assert (
        store().query_read(
            s.catalog_table.select().where(s.catalog_table.c.id == table_id)
        )
        == []
    )
    assert (
        store().query_read(
            s.catalog_column.select().where(s.catalog_column.c.id == column_id)
        )
        == []
    )
