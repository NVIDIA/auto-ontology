# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reads and writes behind description enrichment, against a real store."""

from __future__ import annotations

import os
import uuid

import pytest

pytest.importorskip("sqlalchemy")

from sqlalchemy import select  # noqa: E402

from auto_ontology.dal import descriptions as d  # noqa: E402
from auto_ontology.dal import schema as s  # noqa: E402
from auto_ontology.dal.session import store  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def require_schema():
    if not os.environ.get("POSTGRES_USER"):
        pytest.skip("POSTGRES_* not set")
    try:
        store().query_read("SELECT source_description FROM catalog_column LIMIT 1")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"auto_ontology schema unavailable (alembic upgrade head): {exc}")


def _add(table, **values) -> str:
    return store().query_write(table.insert().values(**values).returning(table.c.id))[
        0
    ]["id"]


class World:
    def __init__(self, prefix: str) -> None:
        self.db = _add(s.catalog_database, name=prefix)
        self.schema = _add(s.catalog_schema, database_id=self.db, name="main")
        self.term = _add(s.term, name=f"{prefix}-term")
        self.tables: dict[str, str] = {}
        self.columns: dict[str, str] = {}

    def table(self, name: str, **values) -> str:
        tid = _add(s.catalog_table, schema_id=self.schema, name=name, **values)
        self.tables[name] = tid
        return tid

    def column(self, table: str, name: str, position: int = 1, **values) -> str:
        cid = _add(
            s.catalog_column,
            table_id=self.tables[table],
            name=name,
            ordinal_position=position,
            **values,
        )
        self.columns[f"{table}.{name}"] = cid
        return cid

    def attribute(self, table: str, column: str, **values) -> str:
        aid = _add(
            s.column_attribute,
            name=f"{table}-{column}",
            source_column=column,
            term_name="t",
            table_id=self.tables[table],
            **values,
        )
        store().query_write(
            s.column__has_attribute.insert().values(
                column_id=self.columns[f"{table}.{column}"], attribute_id=aid
            )
        )
        store().query_write(
            s.column_attribute__term.insert().values(
                attribute_id=aid, term_id=self.term
            )
        )
        return aid


@pytest.fixture
def world():
    w = World(f"de-{uuid.uuid4().hex[:8]}")
    yield w
    store().query_write(
        s.catalog_database.delete().where(s.catalog_database.c.id == w.db)
    )
    store().query_write(s.term.delete().where(s.term.c.id == w.term))


def _one(table, row_id: str) -> dict:
    return dict(store().query_read(select(table).where(table.c.id == row_id))[0])


def test_rewrite_keeps_the_original_once(world) -> None:
    tid = world.table("orders", description="Sales orders.")
    cid = world.column("orders", "Store Code", description="Store Code,, 7 digits")
    blank = world.column("orders", "Express", position=2)
    aid = world.attribute("orders", "Store Code", description="old")

    d.write_enriched_descriptions(
        tid, "Row per order.", {cid: "First.", blank: "Flag."}, {aid: "First."}
    )
    d.write_enriched_descriptions(tid, "Again.", {cid: "Second."}, {aid: "Second."})

    table = _one(s.catalog_table, tid)
    assert (table["description"], table["source_description"]) == (
        "Again.",
        "Sales orders.",
    )
    column = _one(s.catalog_column, cid)
    assert (column["description"], column["source_description"]) == (
        "Second.",
        "Store Code,, 7 digits",
    )
    assert _one(s.catalog_column, blank)["source_description"] == ""
    assert _one(s.column_attribute, aid)["description"] == "Second."

    read = d.fetch_table_for_enrichment(tid)
    assert read["source_description"] == "Sales orders."
    by_name = {c["name"]: c for c in read["columns"]}
    assert by_name["Store Code"]["source_description"] == "Store Code,, 7 digits"
    assert by_name["Store Code"]["attribute_id"] == aid
    assert by_name["Express"]["source_description"] == ""
    assert by_name["Express"]["attribute_id"] is None


def test_certified_text_is_never_replaced(world) -> None:
    tid = world.table("orders", description="Human.", description_certified=True)
    cid = world.column(
        "orders", "Store Code", description="Human.", description_certified=True
    )
    aid = world.attribute("orders", "Store Code", description="Human.", certified=True)

    written = d.write_enriched_descriptions(tid, "LLM.", {cid: "LLM."}, {aid: "LLM."})

    assert written == {"table": False, "columns": [], "attributes": []}
    assert _one(s.catalog_table, tid)["description"] == "Human."
    assert _one(s.catalog_table, tid)["source_description"] is None
    assert _one(s.catalog_column, cid)["description"] == "Human."
    attribute = _one(s.column_attribute, aid)
    assert attribute["description"] == "Human."
    assert attribute["description_suggestion"] == "LLM."


def test_links_cover_declared_and_semantic_fks_both_ways(world) -> None:
    world.table("orders")
    world.table("stores")
    world.table("returns")
    orders_key = world.column("orders", "StoreKey")
    stores_key = world.column("stores", "StoreKey")
    returns_key = world.column("returns", "store_key")
    world.column("orders", "Self")
    store().query_write(
        s.column__foreign_key.insert().values(
            source_column_id=orders_key, target_column_id=stores_key
        )
    )
    orders_attr = world.attribute("orders", "StoreKey")
    store().query_write(
        s.column__semantic_fk.insert().values(
            column_id=returns_key, attribute_id=orders_attr
        )
    )

    links = {
        (link["direction"], link["local_column"], link["other_table_name"])
        for link in d.fetch_table_links(world.tables["orders"])
    }
    assert links == {("out", "StoreKey", "stores"), ("in", "StoreKey", "returns")}

    assert d.fetch_tables_with_term(list(world.tables.values())) == set()
    store().query_write(
        s.table__term.insert().values(
            table_id=world.tables["orders"], term_id=world.term
        )
    )
    assert d.fetch_tables_with_term(list(world.tables.values())) == {
        world.tables["orders"]
    }

    neighbours = d.fetch_neighbour_tables([world.tables["stores"]])
    assert [c["name"] for c in neighbours[world.tables["stores"]]["columns"]] == [
        "StoreKey"
    ]
