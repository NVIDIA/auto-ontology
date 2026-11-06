# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Legacy table flags, the compile work lists that skip them, and the purge.

As in ``test_reset``, the purge tests assert what **survives** as well as what
goes, and the pgvector side is stubbed.
"""

from __future__ import annotations

import os
import uuid

import pytest

pytest.importorskip("sqlalchemy")

from sqlalchemy import select  # noqa: E402

from auto_ontology.dal import attributes as a  # noqa: E402
from auto_ontology.dal import datasources as d  # noqa: E402
from auto_ontology.dal import reset as r  # noqa: E402
from auto_ontology.dal import schema as s  # noqa: E402
from auto_ontology.dal import terms as t  # noqa: E402
from auto_ontology.dal.session import store  # noqa: E402
from auto_ontology.dal.tests.semantic_fk_support import (  # noqa: E402
    reference_attribute,
    references_of,
)


@pytest.fixture(scope="module", autouse=True)
def require_schema():
    if not os.environ.get("POSTGRES_USER"):
        pytest.skip("POSTGRES_* not set")
    try:
        store().query_read("SELECT is_legacy FROM catalog_table LIMIT 1")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"auto_ontology schema unavailable (alembic upgrade head): {exc}")


class _StubVdb:
    def __init__(self) -> None:
        self.deleted_ids: list[str] = []

    def delete_by_id(self, node_id: str) -> int:
        self.deleted_ids.append(node_id)
        return 1


@pytest.fixture
def semantic_vdb(monkeypatch) -> _StubVdb:
    vdb = _StubVdb()
    monkeypatch.setattr(r, "get_semantic_vdb", lambda: vdb)
    return vdb


def _add(table, **values) -> str:
    return store().query_write(table.insert().values(**values).returning(table.c.id))[
        0
    ]["id"]


def _link(table, **values) -> None:
    store().query_write(table.insert().values(**values))


def _exists(table, row_id: str) -> bool:
    return bool(store().query_read(select(table.c.id).where(table.c.id == row_id)))


class World:
    """A database with a legacy table, its replacement, and a third table."""

    def __init__(self, prefix: str) -> None:
        self.prefix = prefix
        self.name = f"{prefix}-db"
        self.database = _add(s.catalog_database, name=self.name)
        schema = _add(s.catalog_schema, database_id=self.database, name="public")
        self.tables: dict[str, str] = {}
        self.columns: dict[str, str] = {}
        for table in ("XA_ORDERS", "XB_ORDERS", "LINES"):
            self.tables[table] = _add(s.catalog_table, schema_id=schema, name=table)
        for table, column in (
            ("XA_ORDERS", "ORDER_ID"),
            ("XA_ORDERS", "CUSTOMER_ID"),
            ("XB_ORDERS", "ORDER_ID"),
            ("XB_ORDERS", "CUSTOMER_ID"),
            ("LINES", "ORDER_ID"),
        ):
            self.columns[f"{table}.{column}"] = _add(
                s.catalog_column,
                table_id=self.tables[table],
                name=column,
                ordinal_position=len(self.columns) + 1,
            )

    @property
    def legacy(self) -> str:
        return self.tables["XA_ORDERS"]

    @property
    def current(self) -> str:
        return self.tables["XB_ORDERS"]

    def mark_legacy(self) -> None:
        d.set_legacy_tables(
            self.name,
            [
                {
                    "table_id": self.legacy,
                    "replacement_table_id": self.current,
                    "reason": "test",
                }
            ],
        )

    def term(self, label: str, *tables: str) -> str:
        term_id = _add(s.term, name=f"{self.prefix}-{label}")
        for table in tables:
            _link(s.table__term, table_id=self.tables[table], term_id=term_id)
        return term_id

    def attribute(self, label: str, column: str) -> str:
        table, source_column = column.split(".")
        attr_id = _add(
            s.column_attribute,
            name=f"{self.prefix}-{label}",
            source_column=source_column,
            term_name=f"{self.prefix}-{label}",
            table_id=self.tables[table],
        )
        _link(
            s.column__has_attribute,
            column_id=self.columns[column],
            attribute_id=attr_id,
        )
        return attr_id

    def sql_attribute(self, label: str, source: str, table: str) -> str:
        attr_id = _add(s.sql_attribute, name=f"{self.prefix}-{label}", source=source)
        query = _add(s.sql_query, sql_full_query=f"SELECT 1 -- {self.prefix}-{label}")
        _link(s.sql_attribute__sql, attribute_id=attr_id, sql_query_id=query)
        _link(s.sql_query__table, sql_query_id=query, table_id=self.tables[table])
        return attr_id

    def query(self, label: str, *tables: str, total: int = 1) -> str:
        query = _add(
            s.sql_query,
            sql_full_query=f"SELECT 1 -- {self.prefix}-{label}",
            total_counter=total,
        )
        for table in tables:
            _link(s.sql_query__table, sql_query_id=query, table_id=self.tables[table])
        return query


@pytest.fixture
def world():
    prefix = f"lg-{uuid.uuid4().hex[:8]}"
    worlds: list[World] = []

    def make() -> World:
        w = World(f"{prefix}{len(worlds)}")
        worlds.append(w)
        return w

    yield make

    for w in worlds:
        store().query_write(
            s.catalog_database.delete().where(s.catalog_database.c.id == w.database)
        )
    for table in (s.term, s.column_attribute, s.sql_attribute):
        store().query_write(table.delete().where(table.c.name.like(f"{prefix}%")))
    store().query_write(
        s.sql_query.delete().where(s.sql_query.c.sql_full_query.like(f"%{prefix}%"))
    )


# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------


def test_set_legacy_tables_replaces_flags_within_one_database(world) -> None:
    w, other = world(), world()
    w.mark_legacy()
    other.mark_legacy()

    assert d.fetch_legacy_table_ids(w.name) == {w.legacy}
    row = d.fetch_table_by_id(w.legacy)
    assert row is not None

    d.set_legacy_tables(w.name, [])

    assert d.fetch_legacy_table_ids(w.name) == set()
    assert d.fetch_legacy_table_ids(other.name) == {other.legacy}
    assert {w.legacy, other.legacy} & d.fetch_legacy_table_ids() == {other.legacy}


def test_a_decision_for_another_databases_table_is_ignored(world) -> None:
    w, other = world(), world()

    d.set_legacy_tables(
        w.name,
        [{"table_id": other.legacy, "replacement_table_id": None, "reason": "x"}],
    )

    assert d.fetch_legacy_table_ids(other.name) == set()


def test_replacement_and_reason_are_stored(world) -> None:
    w = world()
    w.mark_legacy()

    [row] = store().query_read(
        select(
            s.catalog_table.c.legacy_replacement_table_id,
            s.catalog_table.c.legacy_reason,
        ).where(s.catalog_table.c.id == w.legacy)
    )

    assert row == {"legacy_replacement_table_id": w.current, "legacy_reason": "test"}


def test_detection_read_carries_columns_and_query_usage(world) -> None:
    w = world()
    w.query("q1", "XA_ORDERS", total=3)
    w.query("q2", "XA_ORDERS", "XB_ORDERS", total=4)

    tables = {row["name"]: row for row in d.fetch_tables_for_legacy_detection(w.name)}

    assert set(tables) == {"XA_ORDERS", "XB_ORDERS", "LINES"}
    assert [c["name"] for c in tables["XA_ORDERS"]["columns"]] == [
        "ORDER_ID",
        "CUSTOMER_ID",
    ]
    assert tables["XA_ORDERS"]["query_count"] == 7
    assert tables["XB_ORDERS"]["query_count"] == 4
    assert tables["LINES"]["query_count"] == 0


# ---------------------------------------------------------------------------
# Compile work lists
# ---------------------------------------------------------------------------


def test_legacy_tables_are_not_compiled(world) -> None:
    w = world()
    w.mark_legacy()

    names = {row["name"] for row in d.fetch_all_tables_without_term(w.name)}

    assert names == {"XB_ORDERS", "LINES"}


def test_legacy_columns_are_not_fk_candidates(world) -> None:
    w = world()
    w.mark_legacy()

    tables = {row["table_id"] for row in a.find_unlinked_fk_columns(w.name)}

    assert w.legacy not in tables
    assert w.current in tables


def test_semantic_fk_never_touches_a_legacy_table(world) -> None:
    w = world()
    current_attr = w.attribute("current-order", "XB_ORDERS.ORDER_ID")
    legacy_attr = w.attribute("legacy-order", "XA_ORDERS.ORDER_ID")
    w.term("current-orders", "XB_ORDERS")
    w.term("legacy-orders", "XA_ORDERS")
    w.mark_legacy()

    assert not a.merge_semantic_fk(w.columns["XA_ORDERS.CUSTOMER_ID"], current_attr)
    assert not a.merge_semantic_fk(w.columns["LINES.ORDER_ID"], legacy_attr)
    assert a.merge_semantic_fk(w.columns["LINES.ORDER_ID"], current_attr)


def test_sql_suggestions_skip_statements_on_legacy_tables(world) -> None:
    w = world()
    term_id = w.term("order", "XB_ORDERS")
    kept = w.query("current-only", "XB_ORDERS")
    dropped = w.query("joins-legacy", "XB_ORDERS", "XA_ORDERS")
    w.mark_legacy()

    [row] = [r for r in t.fetch_terms_with_sqls() if r["term_id"] == term_id]

    ids = {sql["sql_id"] for sql in row["sqls"]}
    assert kept in ids
    assert dropped not in ids


# ---------------------------------------------------------------------------
# Purge
# ---------------------------------------------------------------------------


def test_purge_removes_the_legacy_tables_entities_and_keeps_the_rest(
    world, semantic_vdb
) -> None:
    w = world()
    legacy_term = w.term("legacy-only", "XA_ORDERS")
    shared_term = w.term("shared", "XA_ORDERS", "XB_ORDERS")
    legacy_attr = w.attribute("legacy-order", "XA_ORDERS.ORDER_ID")
    current_attr = w.attribute("current-order", "XB_ORDERS.ORDER_ID")
    # A legacy column linked to an attribute another table owns.
    _link(
        s.column__has_attribute,
        column_id=w.columns["XA_ORDERS.CUSTOMER_ID"],
        attribute_id=current_attr,
    )
    # A current table referencing the legacy table's key.
    reference_attribute(w.columns["LINES.ORDER_ID"], legacy_attr)
    compiled_on_legacy = w.sql_attribute("compiled", "sql", "XA_ORDERS")
    manual_on_legacy = w.sql_attribute("manual", "manual", "XA_ORDERS")
    compiled_on_current = w.sql_attribute("compiled-current", "sql", "XB_ORDERS")

    r.purge_semantic_entities_for_tables([w.legacy])

    assert not _exists(s.term, legacy_term)
    assert not _exists(s.column_attribute, legacy_attr)
    assert not _exists(s.sql_attribute, compiled_on_legacy)
    assert set(semantic_vdb.deleted_ids) == {
        legacy_term,
        legacy_attr,
        compiled_on_legacy,
    }

    assert _exists(s.term, shared_term)
    assert _exists(s.column_attribute, current_attr)
    assert _exists(s.sql_attribute, manual_on_legacy)
    assert _exists(s.sql_attribute, compiled_on_current)
    assert _exists(s.catalog_table, w.legacy)

    links = store().query_read(
        select(s.table__term.c.table_id).where(s.table__term.c.term_id == shared_term)
    )
    assert [row["table_id"] for row in links] == [w.current]
    legacy_columns = [
        w.columns["XA_ORDERS.ORDER_ID"],
        w.columns["XA_ORDERS.CUSTOMER_ID"],
    ]
    assert not store().query_read(
        select(s.column__has_attribute.c.column_id).where(
            s.column__has_attribute.c.column_id.in_(legacy_columns)
        )
    )
    assert not store().query_read(
        select(s.concept__pk.c.column_id).where(
            s.concept__pk.c.column_id.in_(legacy_columns)
        )
    )
    assert references_of(legacy_attr) == []
    assert store().query_read(
        select(s.column__has_attribute.c.column_id).where(
            s.column__has_attribute.c.attribute_id == current_attr
        )
    ) == [{"column_id": w.columns["XB_ORDERS.ORDER_ID"]}]


def test_purge_of_nothing_is_a_no_op(semantic_vdb) -> None:
    assert r.purge_semantic_entities_for_tables([]) == 0
    assert semantic_vdb.deleted_ids == []
