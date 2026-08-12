# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regressions for defects found in review of the Postgres migration.

Each of these was silent: nothing raised, nothing failed, and the wrong value
was a plausible one. They are pinned here because the code paths that produce
them are not otherwise exercised end to end.
"""

from __future__ import annotations

import os
import uuid

import pytest

pytest.importorskip("psycopg")

from sqlalchemy import select  # noqa: E402
from sqlalchemy.dialects.postgresql import insert  # noqa: E402

from gsf.catalog.store import registry  # noqa: E402
from gsf.catalog.store import rows as R  # noqa: E402
from gsf.catalog.store.rows import resolve_id, upsert_row  # noqa: E402
from gsf.dal import schema as s  # noqa: E402
from gsf.dal.session import store  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def require_schema():
    if not os.environ.get("POSTGRES_USER"):
        pytest.skip("POSTGRES_* not set")
    try:
        store().query_read(f"SELECT 1 FROM {s.SCHEMA}.catalog_database LIMIT 1")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"catalog schema not reachable: {exc}")


@pytest.fixture
def db_name():
    name = f"rg-{uuid.uuid4().hex[:8]}"
    yield name
    store().query_write(
        s.catalog_database.delete().where(s.catalog_database.c.name == name)
    )


def test_a_conflicting_upsert_does_not_rewrite_the_primary_key(db_name) -> None:
    """The ``ON CONFLICT`` update set must never contain ``id``.

    It did. Two ingests racing on the same natural key re-keyed the existing
    row to the loser's UUID, so every id the winner was holding — already
    written into child rows and returned to callers — pointed at nothing.
    """
    first = upsert_row(
        "Database", {"name": db_name}, {"id": str(uuid.uuid4()), "name": db_name}
    )

    spec = registry.entity_spec("Database")
    values = registry.projected("Database", {"id": str(uuid.uuid4()), "name": db_name})
    skip = set(spec.natural_key) | {"id"}
    update = {k: v for k, v in values.items() if k not in skip}
    statement = insert(spec.table).values(**values)
    target = R._conflict_target(spec)
    statement = (
        statement.on_conflict_do_update(index_elements=target, set_=update)
        if update
        else statement.on_conflict_do_nothing(index_elements=target)
    )
    store().query_write(statement)

    assert resolve_id("Database", {"name": db_name}) == first


def test_upsert_row_never_offers_id_for_update() -> None:
    """Guards the rule itself, not just one entity's behaviour."""
    for label in ("Database", "Schema", "Table", "Column", "Sql"):
        spec = registry.entity_spec(label)
        values = registry.projected(
            label, {"id": "x", "name": "n", "sql_full_query": "SELECT 1"}
        )
        skip = (
            set(spec.natural_key)
            | {spec.parent_column if p == "parent" else p for p in spec.natural_key}
            | {"id"}
        )
        assert "id" not in {k for k in values if k not in skip}, label


def _column(db_name: str, table: str, column: str) -> tuple[str, str]:
    db = upsert_row("Database", {"name": db_name}, {"name": db_name})
    schema = upsert_row("Schema", {"name": "s"}, {"name": "s"}, parent_id=db)
    tid = upsert_row("Table", {"name": table}, {"name": table}, parent_id=schema)
    cid = upsert_row("Column", {"name": column}, {"name": column}, parent_id=tid)
    return tid, cid


def test_a_join_link_records_its_reference(db_name) -> None:
    """``refs`` is the point of the table and was never written.

    ``sql_parse`` emitted ``join_refs`` while the link spec declared ``refs``,
    so the payload was always empty, every write took
    ``on_conflict_do_nothing``, and the column stayed ``{}`` forever.
    """
    from gsf.catalog.constants import Props
    from gsf.catalog.model.node import CatalogNode
    from gsf.catalog.store.queries import add_query

    _, left = _column(db_name, "orders", "customer_id")
    _, right = _column(db_name, "customers", "id")

    def node(cid: str) -> CatalogNode:
        return CatalogNode("c", label="Column", props={"id": cid}, existing_id=cid)

    add_query([(node(left), node(right), {Props.JOIN: True, "refs": ["sql-1|a = b"]})])
    rows = store().query_read(
        select(s.column_join.c.refs).where(s.column_join.c.source_column_id == left)
    )
    assert rows and rows[0]["refs"] == ["sql-1|a = b"]


def test_repeated_joins_accumulate_and_deduplicate(db_name) -> None:
    """The whole reason ``refs`` is an array rather than a column.

    Also the first exercise of ``_dedupe``: while the payload was empty this
    path never ran, and the expression it generated was invalid SQL.
    """
    from gsf.catalog.constants import Props
    from gsf.catalog.model.node import CatalogNode
    from gsf.catalog.store.queries import add_query

    _, left = _column(db_name, "orders", "customer_id")
    _, right = _column(db_name, "customers", "id")

    def node(cid: str) -> CatalogNode:
        return CatalogNode("c", label="Column", props={"id": cid}, existing_id=cid)

    for ref in ("sql-1|a = b", "sql-2|a = b", "sql-1|a = b"):
        add_query([(node(left), node(right), {Props.JOIN: True, "refs": [ref]})])

    rows = store().query_read(
        select(s.column_join.c.refs).where(s.column_join.c.source_column_id == left)
    )
    assert sorted(rows[0]["refs"]) == ["sql-1|a = b", "sql-2|a = b"]


def test_a_reset_takes_the_statements_with_it(db_name) -> None:
    """``sql_query`` has no foreign key into the catalog, so nothing cascades.

    Left behind, the rows are invisible but not inert: dedup matches on
    ``md5(sql_full_query)``, so the next ingest merges into the stale row and
    ``total_counter`` accumulates across resets.
    """
    from gsf.dal.reset import _delete_orphaned_statements

    tid, cid = _column(db_name, "orders", "customer_id")
    statement = f"SELECT 1 -- {db_name}"
    sql_id = upsert_row(
        "Sql", {"sql_full_query": statement}, {"sql_full_query": statement}
    )
    store().query_write(
        insert(s.sql_query_table)
        .values(sql_query_id=sql_id, table_id=tid)
        .on_conflict_do_nothing()
    )

    store().query_write(
        s.catalog_database.delete().where(s.catalog_database.c.name == db_name)
    )
    # The link cascaded; the statement did not.
    assert store().query_read(
        select(s.sql_query.c.id).where(s.sql_query.c.id == sql_id)
    )

    _delete_orphaned_statements()
    assert not store().query_read(
        select(s.sql_query.c.id).where(s.sql_query.c.id == sql_id)
    )


def test_a_statement_owned_by_an_attribute_survives_the_sweep(db_name) -> None:
    """The sweep must not reach the semantic tier.

    A SqlAttribute's statement has no catalog link, so a sweep keyed on the
    catalog links alone would delete it on every data-layer reset.
    """
    from gsf.dal.reset import _delete_orphaned_statements

    statement = f"SELECT 2 -- {db_name}"
    sql_id = upsert_row(
        "Sql", {"sql_full_query": statement}, {"sql_full_query": statement}
    )
    attr_id = store().query_write(
        s.sql_attribute.insert()
        .values(name=f"{db_name}-attr", expression=statement)
        .returning(s.sql_attribute.c.id)
    )[0]["id"]
    store().query_write(
        insert(s.sql_attribute_sql)
        .values(attribute_id=attr_id, sql_query_id=sql_id)
        .on_conflict_do_nothing()
    )
    try:
        _delete_orphaned_statements()
        assert store().query_read(
            select(s.sql_query.c.id).where(s.sql_query.c.id == sql_id)
        ), "a statement owned by a SqlAttribute must survive"
    finally:
        store().query_write(
            s.sql_attribute.delete().where(s.sql_attribute.c.id == attr_id)
        )
        store().query_write(s.sql_query.delete().where(s.sql_query.c.id == sql_id))
