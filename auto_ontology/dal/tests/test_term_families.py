# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The term family reads and the replace write, against a real store."""

from __future__ import annotations

import os
import uuid

import pytest

pytest.importorskip("sqlalchemy")

from sqlalchemy import select  # noqa: E402

from auto_ontology.dal import schema as s  # noqa: E402
from auto_ontology.dal import term_families as tf  # noqa: E402
from auto_ontology.dal.session import store  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def require_schema():
    if not os.environ.get("POSTGRES_USER"):
        pytest.skip("POSTGRES_* not set")
    try:
        store().query_read("SELECT 1 FROM term_family LIMIT 1")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"auto_ontology schema unavailable (alembic upgrade head): {exc}")


def _add(table, **values) -> str:
    return store().query_write(table.insert().values(**values).returning(table.c.id))[
        0
    ]["id"]


def _link(table, **values) -> None:
    store().query_write(table.insert().values(**values))


class Database:
    """One database with one schema; tables get a Term and named columns."""

    def __init__(self, prefix: str, label: str) -> None:
        self.prefix = prefix
        self.name = f"{prefix}-{label}"
        self.id = _add(s.catalog_database, name=self.name)
        self.schema = _add(s.catalog_schema, database_id=self.id, name="dw")
        self.columns: dict[str, str] = {}

    def table(self, name: str, columns: list[str], *, term: bool = True) -> str:
        table_id = _add(s.catalog_table, schema_id=self.schema, name=name)
        # Inserted out of order, so the ordinal sort is what the read relies on.
        for position, column in reversed(list(enumerate(columns, start=1))):
            self.columns[f"{name}.{column}"] = _add(
                s.catalog_column,
                table_id=table_id,
                name=column,
                ordinal_position=position,
            )
        if term:
            term_id = _add(s.term, name=f"{self.prefix}-{self.name}-{name}")
            _link(s.table__term, table_id=table_id, term_id=term_id)
        return table_id

    def term_of(self, table_id: str) -> str:
        return store().query_read(
            select(s.table__term.c.term_id).where(s.table__term.c.table_id == table_id)
        )[0]["term_id"]

    def statement(self, label: str, owner=None) -> str:
        query = _add(s.sql_query, sql_full_query=f"SELECT 1 -- {self.prefix}-{label}")
        if owner is not None:
            owner_table, link_table, owner_column = owner
            owner_id = _add(owner_table, name=f"{self.prefix}-{label}")
            _link(link_table, **{owner_column: owner_id, "sql_query_id": query})
        return query

    def join(self, source: str, target: str, *statements: str) -> None:
        _link(
            s.column__join,
            source_column_id=self.columns[source],
            target_column_id=self.columns[target],
            refs=[f"{sql_id}|{source} = {target}" for sql_id in statements],
        )

    def fk(self, source: str, target: str) -> None:
        _link(
            s.column__foreign_key,
            source_column_id=self.columns[source],
            target_column_id=self.columns[target],
        )


@pytest.fixture
def world():
    prefix = f"tf-{uuid.uuid4().hex[:8]}"
    created: list[Database] = []

    def make(label: str) -> Database:
        database = Database(prefix, label)
        created.append(database)
        return database

    yield make

    for database in created:
        store().query_write(
            s.catalog_database.delete().where(s.catalog_database.c.id == database.id)
        )
    for table in (s.term, s.sql_attribute, s.custom_analysis):
        store().query_write(table.delete().where(table.c.name.like(f"{prefix}%")))
    store().query_write(
        s.sql_query.delete().where(s.sql_query.c.sql_full_query.like(f"%{prefix}%"))
    )


_SQL_ATTRIBUTE = (s.sql_attribute, s.sql_attribute__sql, "attribute_id")
_CUSTOM_ANALYSIS = (s.custom_analysis, s.custom_analysis__sql, "analysis_id")


def _pairs(rows: list[dict]) -> set[tuple[str, str, str]]:
    return {(r["source_column"], r["target_column"], r["origin"]) for r in rows}


def test_candidates_carry_columns_in_ordinal_order(world) -> None:
    db = world("shop")
    rooms = db.table("OLD_ROOMS", ["ROOM_ID", "BUILDING_ID", "FLOOR"])
    db.table("NO_TERM", ["ID"], term=False)

    rows = tf.fetch_family_candidates(db.name)

    assert [(r["table_id"], r["columns"]) for r in rows] == [
        (rooms, ["ROOM_ID", "BUILDING_ID", "FLOOR"])
    ]
    assert rows[0]["schema_id"] == db.schema
    assert rows[0]["term_id"] == db.term_of(rooms)


def test_edges_keep_declared_keys_and_customer_joins_only(world) -> None:
    db = world("shop")
    db.table("ROOMS", ["ROOM_ID", "BUILDING_ID", "STAFF_ID", "VISIT_ID"])
    db.table("BUILDING", ["BUILDING_ID"])
    db.table("STAFF", ["STAFF_ID"])
    db.table("VISITS", ["VISIT_ID"])
    db.table("AUDIT", ["ROOM_ID"])

    db.fk("ROOMS.BUILDING_ID", "BUILDING.BUILDING_ID")
    customer = db.statement("customer")
    generated = db.statement("generated", owner=_SQL_ATTRIBUTE)
    analysis = db.statement("analysis", owner=_CUSTOM_ANALYSIS)
    db.join("ROOMS.STAFF_ID", "STAFF.STAFF_ID", customer)
    db.join("ROOMS.VISIT_ID", "VISITS.VISIT_ID", generated, analysis)
    db.join("AUDIT.ROOM_ID", "ROOMS.ROOM_ID", generated, customer)

    assert _pairs(tf.fetch_family_edges(db.name)) == {
        ("BUILDING_ID", "BUILDING_ID", "fk"),
        ("STAFF_ID", "STAFF_ID", "sql"),
        ("ROOM_ID", "ROOM_ID", "sql"),
    }


def test_edges_stay_inside_the_database(world) -> None:
    shop, warehouse = world("shop"), world("warehouse")
    shop.table("ROOMS", ["BUILDING_ID"])
    warehouse.table("BUILDING", ["BUILDING_ID"])
    shop.columns.update(warehouse.columns)
    shop.fk("ROOMS.BUILDING_ID", "BUILDING.BUILDING_ID")

    assert tf.fetch_family_edges(shop.name) == []
    assert tf.fetch_family_edges(warehouse.name) == []


def _families(database_id: str) -> dict[str, set[tuple[str, str]]]:
    rows = store().query_read(
        select(
            s.term_family.c.prefix,
            s.term__term_family.c.term_id,
            s.term__term_family.c.base_name,
        )
        .join(
            s.term__term_family,
            s.term__term_family.c.family_id == s.term_family.c.id,
        )
        .join(s.catalog_schema, s.catalog_schema.c.id == s.term_family.c.schema_id)
        .where(s.catalog_schema.c.database_id == database_id)
    )
    out: dict[str, set[tuple[str, str]]] = {}
    for row in rows:
        out.setdefault(row["prefix"], set()).add((row["term_id"], row["base_name"]))
    return out


def test_replace_swaps_out_the_previous_families(world) -> None:
    shop, warehouse = world("shop"), world("warehouse")
    old = shop.term_of(shop.table("OLD_ROOMS", ["ID"]))
    new = shop.term_of(shop.table("NEW_ROOMS", ["ID"]))
    theirs = warehouse.term_of(warehouse.table("V1_STOCK", ["ID"]))
    tf.replace_term_families(
        warehouse.name, [(warehouse.schema, "V1", {theirs: "STOCK"})]
    )

    assert (
        tf.replace_term_families(shop.name, [(shop.schema, "OLD", {old: "ROOMS"})]) == 1
    )
    linked = tf.replace_term_families(
        shop.name,
        [(shop.schema, "NEW", {new: "ROOMS"}), (shop.schema, "OLD", {})],
    )

    assert linked == 1
    assert _families(shop.id) == {"NEW": {(new, "ROOMS")}}
    assert _families(warehouse.id) == {"V1": {(theirs, "STOCK")}}
