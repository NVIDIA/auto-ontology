# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-database reset.

Destructive code, so the tests are written the paranoid way round: every one of
them asserts what **survives**, not only what goes. A delete that removes too
much passes any "is it gone?" assertion.

Two behaviour changes get their own tests, because both are the kind that would
otherwise be discovered by a user:

* **B1 — the deletes are narrower.** ``apoc.path.subgraphNodes`` followed any
  relationship, so a reset of one database could reach a shared Term and, from
  there, delete a *different* database's data. Foreign keys only point downward
  within one database, so that cannot happen now.
* **B2 — a scoped reset still misses ``PqlAnalysis``.** Preserved, not fixed: a
  scoped reset silently wiping every predictive analysis in the deployment is a
  worse surprise than the current gap.

The pgvector side is stubbed. These tests are about which rows the store keeps,
and a real embedding round trip would only add a way for them to fail for an
unrelated reason.

**The unscoped variants run inside a rolled-back transaction.** They are not
scoped to a test fixture by construction — ``delete_all_data(None)`` means every
database in the store — so run directly they delete the shared Pagila/Chinook
catalog every other test suite depends on. That is not hypothetical: the first
version of this file did exactly that. :func:`_rolled_back` contains them.
"""

from __future__ import annotations

import os
import uuid
from contextlib import contextmanager

import pytest

pytest.importorskip("sqlalchemy")

from sqlalchemy import select  # noqa: E402

from gsf.dal.pg import reset as r  # noqa: E402
from gsf.dal.pg import schema as s  # noqa: E402
from gsf.dal.pg.session import store, write_transaction  # noqa: E402


class _Rollback(Exception):
    """Sentinel: unwinds :func:`write_transaction` without reporting a failure."""


@contextmanager
def _rolled_back():
    """Run a destructive block and undo it.

    ``write_transaction`` commits on a clean exit and rolls back on any
    exception, so raising at the end of the block is what makes this work. The
    assertions inside still see the deletion — they are in the same transaction.
    """
    try:
        with write_transaction():
            yield
            raise _Rollback
    except _Rollback:
        pass


@pytest.fixture(scope="module", autouse=True)
def require_schema():
    if not os.environ.get("POSTGRES_USER"):
        pytest.skip("POSTGRES_* not set")
    try:
        store().query_read(f"SELECT 1 FROM {s.SCHEMA}.term LIMIT 1")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"gsf schema unavailable (alembic upgrade head): {exc}")


class _StubVdb:
    def __init__(self) -> None:
        self.deleted_all = 0
        self.deleted_databases: list[str] = []

    def delete_all(self) -> int:
        self.deleted_all += 1
        return 7

    def delete_by_database(self, database_name: str) -> list[str]:
        self.deleted_databases.append(database_name)
        return ["a", "b", "c"]


@pytest.fixture(autouse=True)
def stub_vdbs(monkeypatch):
    data, semantic = _StubVdb(), _StubVdb()
    monkeypatch.setattr(r, "get_data_vdb", lambda: data)
    monkeypatch.setattr(r, "get_semantic_vdb", lambda: semantic)
    return {"data": data, "semantic": semantic}


def _add(table, **values) -> str:
    return store().query_write(table.insert().values(**values).returning(table.c.id))[
        0
    ]["id"]


def _link(table, **values) -> None:
    store().query_write(table.insert().values(**values))


def _exists(table, row_id: str) -> bool:
    return bool(store().query_read(select(table.c.id).where(table.c.id == row_id)))


class Database:
    """One database with a table, a column, and its own semantic rows."""

    def __init__(self, prefix: str, label: str) -> None:
        self.name = f"{prefix}-{label}"
        self.prefix = prefix
        self.id = _add(s.catalog_database, name=self.name)
        self.schema = _add(s.catalog_schema, database_id=self.id, name="public")
        self.table = _add(s.catalog_table, schema_id=self.schema, name="orders")
        self.column = _add(
            s.catalog_column, table_id=self.table, name="id", ordinal_position=1
        )

    def term(self, label: str) -> str:
        term_id = _add(s.term, name=f"{self.prefix}-{label}")
        _link(s.table_term, table_id=self.table, term_id=term_id)
        return term_id

    def column_attribute(self, label: str) -> str:
        return _add(
            s.column_attribute,
            name=f"{self.prefix}-{label}",
            source_column="id",
            term_name=f"{self.prefix}-{label}",
            table_id=self.table,
        )

    def sql_owner(self, owner_table, link_table, owner_column, label: str) -> tuple:
        owner = _add(owner_table, name=f"{self.prefix}-{label}")
        query = _add(s.sql_query, sql_full_query=f"SELECT 1 -- {self.prefix}-{label}")
        _link(link_table, **{owner_column: owner, "sql_query_id": query})
        _link(s.sql_query_table, sql_query_id=query, table_id=self.table)
        return owner, query


@pytest.fixture
def world():
    prefix = f"r-{uuid.uuid4().hex[:8]}"
    created: list[Database] = []

    def make(label: str) -> Database:
        database = Database(prefix, label)
        created.append(database)
        return database

    yield prefix, make

    for database in created:
        store().query_write(
            s.catalog_database.delete().where(s.catalog_database.c.id == database.id)
        )
    for table in (s.term, s.column_attribute, s.sql_attribute, s.custom_analysis):
        store().query_write(table.delete().where(table.c.name.like(f"{prefix}%")))
    store().query_write(
        s.pql_analysis.delete().where(s.pql_analysis.c.name.like(f"{prefix}%"))
    )
    store().query_write(
        s.sql_query.delete().where(s.sql_query.c.sql_full_query.like(f"%{prefix}%"))
    )


# --------------------------------------------------------------------------
# The cascade
# --------------------------------------------------------------------------


def test_deleting_a_database_takes_its_whole_catalog(world) -> None:
    """One DELETE; schemas, tables and columns follow by cascade.

    This is what modelling CONTAINS as a parent FK bought — and it means a child
    table added later cannot be forgotten here.
    """
    _, make = world
    database = make("shop")

    r.delete_data_layer(database.name)

    assert not _exists(s.catalog_database, database.id)
    assert not _exists(s.catalog_schema, database.schema)
    assert not _exists(s.catalog_table, database.table)
    assert not _exists(s.catalog_column, database.column)


def test_deleting_one_database_leaves_the_other_alone(world) -> None:
    _, make = world
    shop, warehouse = make("shop"), make("warehouse")

    r.delete_data_layer(shop.name)

    assert not _exists(s.catalog_database, shop.id)
    assert _exists(s.catalog_database, warehouse.id)
    assert _exists(s.catalog_column, warehouse.column)


# --------------------------------------------------------------------------
# B1 — the deletes are narrower than the traversal was
# --------------------------------------------------------------------------


def test_a_term_shared_with_another_database_survives(world) -> None:
    """B1, and the reason it is a *fix* rather than a regression.

    `subgraphNodes` walked from the Database to a shared Term and onward into
    the other database's tables — so resetting one database could delete
    another's data. A Term that both databases represent belongs to both, and a
    reset of one must leave it standing.
    """
    prefix, make = world
    shop, warehouse = make("shop"), make("warehouse")
    shared = shop.term("Product")
    _link(s.table_term, table_id=warehouse.table, term_id=shared)
    own = shop.term("Order")

    r.delete_semantic_layer(shop.name)

    assert _exists(s.term, shared), "a shared Term must not be deleted"
    assert not _exists(s.term, own)
    assert _exists(s.catalog_column, warehouse.column)


def test_an_analysis_spanning_two_databases_survives(world) -> None:
    """Same rule, reached through SQL rather than REPRESENTS."""
    _, make = world
    shop, warehouse = make("shop"), make("warehouse")
    analysis, query = shop.sql_owner(
        s.custom_analysis,
        s.custom_analysis_sql,
        "analysis_id",
        "cross-database",
    )
    _link(s.sql_query_table, sql_query_id=query, table_id=warehouse.table)

    r.delete_semantic_layer(shop.name)

    assert _exists(s.custom_analysis, analysis)


def test_a_single_database_analysis_does_not_survive(world) -> None:
    """The control for the test above: without the second link, it goes.

    Without this, "survives" could just mean the delete never matched anything.
    """
    _, make = world
    shop = make("shop")
    analysis, _ = shop.sql_owner(
        s.custom_analysis, s.custom_analysis_sql, "analysis_id", "local"
    )

    r.delete_semantic_layer(shop.name)

    assert not _exists(s.custom_analysis, analysis)


def test_a_sql_attribute_is_scoped_the_same_way(world) -> None:
    _, make = world
    shop, warehouse = make("shop"), make("warehouse")
    local, _ = shop.sql_owner(
        s.sql_attribute, s.sql_attribute_sql, "attribute_id", "local"
    )
    crossing, query = shop.sql_owner(
        s.sql_attribute, s.sql_attribute_sql, "attribute_id", "crossing"
    )
    _link(s.sql_query_table, sql_query_id=query, table_id=warehouse.table)

    r.delete_semantic_layer(shop.name)

    assert not _exists(s.sql_attribute, local)
    assert _exists(s.sql_attribute, crossing)


def test_a_column_attribute_goes_with_its_table(world) -> None:
    """Owned outright by one table, so no all-or-nothing question arises."""
    _, make = world
    shop, warehouse = make("shop"), make("warehouse")
    ours = shop.column_attribute("order-id")
    theirs = warehouse.column_attribute("stock-id")

    r.delete_semantic_layer(shop.name)

    assert not _exists(s.column_attribute, ours)
    assert _exists(s.column_attribute, theirs)


# --------------------------------------------------------------------------
# B2 — the scoped gap, preserved
# --------------------------------------------------------------------------


def test_a_scoped_reset_does_not_touch_pql_analyses(world) -> None:
    """B2. Preserved deliberately, and asserted so it cannot change by accident.

    A PqlAnalysis is never attached to a database, so the Cypher traversal never
    reached it. Deleting every predictive analysis in the deployment during a
    *scoped* reset would be a worse surprise than leaving them.
    """
    prefix, make = world
    shop = make("shop")
    analysis = _add(s.pql_analysis, name=f"{prefix}-churn", pql="PREDICT x")

    r.delete_semantic_layer(shop.name)

    assert _exists(s.pql_analysis, analysis)


def test_an_unscoped_reset_does_remove_them(world) -> None:
    """Matching on label alone reached nodes orphaned from every Database."""
    prefix, make = world
    make("shop")
    analysis = _add(s.pql_analysis, name=f"{prefix}-churn", pql="PREDICT x")

    with _rolled_back():
        r.delete_semantic_layer(None)
        assert not _exists(s.pql_analysis, analysis)

    # And the rollback really did put it back -- otherwise this file would be
    # quietly wiping the shared fixture again.
    assert _exists(s.pql_analysis, analysis)


# --------------------------------------------------------------------------
# Layer separation and ordering
# --------------------------------------------------------------------------


def test_the_semantic_reset_leaves_the_catalog_standing(world) -> None:
    _, make = world
    shop = make("shop")
    shop.term("Order")

    r.delete_semantic_layer(shop.name)

    assert _exists(s.catalog_database, shop.id)
    assert _exists(s.catalog_table, shop.table)
    assert _exists(s.catalog_column, shop.column)


def test_delete_all_data_removes_the_semantic_rows_too(world) -> None:
    """Order matters: semantic rows are identified *through* the catalog.

    Delete the catalog first and the links are already gone, so the semantic
    pass would find nothing to scope and leave every Term behind. This is the
    test that fails if the two calls are ever swapped.
    """
    _, make = world
    shop = make("shop")
    term = shop.term("Order")
    attribute = shop.column_attribute("order-id")

    result = r.delete_all_data(shop.name)

    assert not _exists(s.term, term)
    assert not _exists(s.column_attribute, attribute)
    assert not _exists(s.catalog_database, shop.id)
    assert result.database_name == shop.name


def test_the_result_reports_pgvector_rows_from_both_layers(world, stub_vdbs) -> None:
    _, make = world
    shop = make("shop")

    result = r.delete_all_data(shop.name)

    assert result.data_rows == 3
    assert result.semantic_rows == 3
    assert stub_vdbs["data"].deleted_databases == [shop.name]
    assert stub_vdbs["semantic"].deleted_databases == [shop.name]


def test_an_unscoped_reset_clears_both_collections(world, stub_vdbs) -> None:
    """`delete_all` rather than a per-database filter — a different call.

    Getting this wrong would leave every embedding in place after a full wipe,
    and nothing in the store would show it.
    """
    _, make = world
    make("shop")

    with _rolled_back():
        result = r.delete_all_data(None)
        assert result.data_rows == 7

    assert stub_vdbs["data"].deleted_all == 1
    assert stub_vdbs["semantic"].deleted_all == 1
    assert stub_vdbs["data"].deleted_databases == []


def test_resetting_a_database_that_does_not_exist_is_harmless(world) -> None:
    _, make = world
    shop = make("shop")

    r.delete_all_data(f"{shop.prefix}-no-such-database")

    assert _exists(s.catalog_database, shop.id)
