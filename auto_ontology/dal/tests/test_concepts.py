# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Concept rows: one per keyed Term, keys on unique columns, merges that move rows.

A merge that recreated rows instead of moving them would still leave a
plausible Concept behind, so each merge test checks *ids*: the survivor's id,
the FK row that pointed at the absorbed key, and the key a recompile reuses.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
import time
import uuid

import pytest

pytest.importorskip("sqlalchemy")

from sqlalchemy import select  # noqa: E402

from auto_ontology.dal import concepts as c  # noqa: E402
from auto_ontology.dal import schema as s  # noqa: E402
from auto_ontology.dal.session import store, write_transaction  # noqa: E402
from auto_ontology.semantic.constants import SEMANTIC_SOURCE  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def require_schema():
    if not os.environ.get("POSTGRES_USER"):
        pytest.skip("POSTGRES_* not set")
    try:
        store().query_read("SELECT 1 FROM concept LIMIT 1")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"auto_ontology schema unavailable (alembic upgrade head): {exc}")


def _add(table, **values) -> str:
    return store().query_write(table.insert().values(**values).returning(table.c.id))[
        0
    ]["id"]


class World:
    def __init__(self, prefix: str) -> None:
        self.prefix = prefix
        self.database = _add(s.catalog_database, name=prefix)
        self.schema = _add(s.catalog_schema, database_id=self.database, name="sales")
        self.tables: dict[str, str] = {}
        self.columns: dict[str, str] = {}
        self.terms: dict[str, str] = {}

    def table(
        self,
        name: str,
        columns: tuple[str, ...],
        *,
        pk: str | None = None,
        unique: tuple[str, ...] = (),
        repeating: tuple[str, ...] = (),
    ) -> str:
        tid = _add(
            s.catalog_table,
            schema_id=self.schema,
            name=name,
            pk=[pk] if pk else None,
        )
        self.tables[name] = tid
        for position, column in enumerate(columns, start=1):
            is_unique = (
                True if column in unique else False if column in repeating else None
            )
            self.columns[f"{name}.{column}"] = _add(
                s.catalog_column,
                table_id=tid,
                name=column,
                ordinal_position=position,
                is_unique=is_unique,
                distinct_values=json.dumps(["C00001", "C00002"]),
            )
        return tid

    def term(self, name: str, table: str) -> str:
        tid = _add(s.term, name=f"{self.prefix}-{name}", source=SEMANTIC_SOURCE)
        store().query_write(
            s.table__term.insert().values(table_id=self.tables[table], term_id=tid)
        )
        self.terms[name] = tid
        return tid

    def field(self, term: str, column: str) -> str:
        table, name = column.split(".")
        aid = _add(
            s.column_attribute,
            name=f"{self.prefix}-{term}-{name}",
            source_column=name,
            term_name=f"{self.prefix}-{term}",
            table_id=self.tables[table],
        )
        store().query_write(
            s.column_attribute__term.insert().values(
                attribute_id=aid, term_id=self.terms[term]
            )
        )
        store().query_write(
            s.column__has_attribute.insert().values(
                column_id=self.columns[column], attribute_id=aid
            )
        )
        return aid

    def fk(self, column: str, key_id: str) -> None:
        store().query_write(
            s.column__semantic_fk.insert().values(
                column_id=self.columns[column], key_id=key_id
            )
        )


@pytest.fixture()
def world():
    w = World(f"concept-test-{uuid.uuid4().hex[:8]}")
    yield w
    store().query_write(
        s.catalog_database.delete().where(s.catalog_database.c.id == w.database)
    )
    for table in (s.term, s.column_attribute):
        store().query_write(table.delete().where(table.c.name.like(f"{w.prefix}%")))


def _concept(concept_id: str) -> dict | None:
    rows = store().query_read(select(s.concept).where(s.concept.c.id == concept_id))
    return dict(rows[0]) if rows else None


def _fk_key(column_id: str) -> str | None:
    rows = store().query_read(
        select(s.column__semantic_fk.c.key_id).where(
            s.column__semantic_fk.c.column_id == column_id
        )
    )
    return rows[0]["key_id"] if rows else None


@pytest.fixture()
def split(world):
    """``customer`` and ``customer_ext`` split one entity; ``orders`` references it."""
    world.table("customer", ("customer_id", "name"), pk="customer_id")
    world.table("customer_ext", ("customer_id", "segment"), pk="customer_id")
    world.table("orders", ("order_id", "customer_id"), pk="order_id")
    for term, table in (
        ("Customer", "customer"),
        ("Customer Ext", "customer_ext"),
        ("Order", "orders"),
    ):
        world.term(term, table)
    world.field("Customer", "customer.name")
    world.field("Customer Ext", "customer_ext.segment")
    return world


def test_every_term_gets_its_own_concept_keyed_on_its_pk(split) -> None:
    concept_id = c.ensure_term_concept(split.terms["Customer"])

    assert concept_id is not None
    assert _concept(concept_id)["name"] == f"{split.prefix}-Customer"
    assert _concept(concept_id)["database_id"] == split.database
    key_id = c.fetch_column_key_id(split.columns["customer.customer_id"])
    assert key_id is not None
    assert c.ensure_term_concept(split.terms["Customer"]) == concept_id
    assert c.fetch_table_concept_id(split.tables["customer"]) == concept_id


def test_a_term_on_no_table_has_no_concept(world) -> None:
    term_id = _add(s.term, name=f"{world.prefix}-Loose", source=SEMANTIC_SOURCE)
    assert c.ensure_term_concept(term_id) is None


def test_a_second_caller_waits_for_the_terms_first_concept(world) -> None:
    """Without the Term lock both create a Concept and one fails on membership."""
    world.table("customer", ("customer_id",), pk="customer_id")
    term = world.term("Customer", "customer")

    with ThreadPoolExecutor(max_workers=1) as pool:
        with write_transaction():
            first = c.ensure_term_concept(term)
            second = pool.submit(c.ensure_term_concept, term)
            time.sleep(0.5)
            assert not second.done(), "the second caller must wait for the lock"
        assert second.result(timeout=10) == first


def test_members_and_key_writes_agree_on_a_tables_concept(world) -> None:
    """A table representing two Terms resolves to one Concept everywhere."""
    world.table("customer", ("customer_id",), pk="customer_id")
    world.table("client", ("client_id",), pk="client_id")
    for term, table in (("Customer", "customer"), ("Client", "client")):
        c.ensure_term_concept(world.term(term, table))
    table_id = world.tables["customer"]
    store().query_write(
        s.table__term.insert().values(table_id=table_id, term_id=world.terms["Client"])
    )

    members = c.fetch_concept_members([table_id])

    assert members[table_id]["concept_id"] == c.fetch_table_concept_id(table_id)


def _date_typed(world, column: str) -> None:
    store().query_write(
        s.catalog_column.update()
        .where(s.catalog_column.c.id == world.columns[column])
        .values(data_type="date")
    )


def test_an_inferred_key_is_never_a_date(world) -> None:
    world.table(
        "visits", ("visit_id", "visited_on"), pk="visit_id", unique=("visited_on",)
    )
    _date_typed(world, "visits.visited_on")
    concept = c.ensure_term_concept(world.term("Visit", "visits"))

    assert c.ensure_concept_key(concept, world.columns["visits.visited_on"]) is None


def test_a_column_named_as_a_date_is_never_an_inferred_key(world) -> None:
    world.table("terms", ("term_id", "TERM_START_DATE"), unique=("TERM_START_DATE",))
    concept = c.ensure_term_concept(world.term("Term", "terms"))

    assert c.ensure_concept_key(concept, world.columns["terms.TERM_START_DATE"]) is None


def test_a_declared_date_key_is_kept(world) -> None:
    """A calendar table keyed on its date, and a declared FK's date target."""
    world.table("calendar", ("day", "label"), pk="day")
    _date_typed(world, "calendar.day")
    world.table("rates", ("rate_id", "as_of"), pk="rate_id", unique=("as_of",))
    _date_typed(world, "rates.as_of")
    world.table("orders", ("order_id", "ordered_on"), pk="order_id")
    store().query_write(
        s.column__foreign_key.insert().values(
            source_column_id=world.columns["orders.ordered_on"],
            target_column_id=world.columns["rates.as_of"],
        )
    )

    calendar = c.ensure_term_concept(world.term("Calendar", "calendar"))
    rates = c.ensure_term_concept(world.term("Rate", "rates"))

    assert c.fetch_column_key_id(world.columns["calendar.day"]) is not None
    assert c.ensure_concept_key(calendar, world.columns["calendar.day"])
    assert c.ensure_concept_key(rates, world.columns["rates.as_of"])


def test_a_repeating_column_cannot_be_a_key(world) -> None:
    world.table("orders", ("order_id", "status"), pk="order_id", repeating=("status",))
    concept_id = c.ensure_term_concept(world.term("Order", "orders"))

    assert c.ensure_concept_key(concept_id, world.columns["orders.status"]) is None


def test_a_key_column_must_be_on_a_member_table(split) -> None:
    customer = c.ensure_term_concept(split.terms["Customer"])
    c.ensure_term_concept(split.terms["Order"])

    assert c.ensure_concept_key(customer, split.columns["orders.order_id"]) is None


def test_a_merge_moves_members_keys_and_fks(split) -> None:
    customer = c.ensure_term_concept(split.terms["Customer"])
    ext = c.ensure_term_concept(split.terms["Customer Ext"])
    c.ensure_term_concept(split.terms["Order"])
    customer_key = c.fetch_column_key_id(split.columns["customer.customer_id"])
    ext_key = c.fetch_column_key_id(split.columns["customer_ext.customer_id"])
    split.fk("orders.customer_id", ext_key)

    assert c.merge_concepts(
        customer, ext, key_pairs={ext_key: customer_key}, name="Customer"
    )

    assert _concept(ext) is None
    assert _concept(customer)["name"] == "Customer"
    assert c.fetch_table_concept_id(split.tables["customer_ext"]) == customer
    assert c.fetch_column_key_id(split.columns["customer_ext.customer_id"]) == (
        customer_key
    )
    assert _fk_key(split.columns["orders.customer_id"]) == customer_key

    members = c.fetch_concept_members([split.tables["customer"]])
    concept = members[split.tables["customer"]]
    assert {m["table_name"] for m in concept["members"]} == {
        "customer",
        "customer_ext",
    }
    (key,) = concept["keys"]
    assert {col["table_name"] for col in key["columns"]} == {
        "customer",
        "customer_ext",
    }


def test_a_satellites_fk_to_its_own_concept_is_dropped(split) -> None:
    """``customer_ext.customer_id`` was an FK to ``customer``; now it is identity."""
    customer = c.ensure_term_concept(split.terms["Customer"])
    ext = c.ensure_term_concept(split.terms["Customer Ext"])
    customer_key = c.fetch_column_key_id(split.columns["customer.customer_id"])
    ext_key = c.fetch_column_key_id(split.columns["customer_ext.customer_id"])
    split.fk("customer_ext.customer_id", customer_key)

    assert c.merge_concepts(customer, ext, key_pairs={ext_key: customer_key})

    assert _fk_key(split.columns["customer_ext.customer_id"]) is None


def test_a_recompile_reuses_the_merged_ids(split) -> None:
    customer = c.ensure_term_concept(split.terms["Customer"])
    ext = c.ensure_term_concept(split.terms["Customer Ext"])
    customer_key = c.fetch_column_key_id(split.columns["customer.customer_id"])
    ext_key = c.fetch_column_key_id(split.columns["customer_ext.customer_id"])
    c.merge_concepts(customer, ext, key_pairs={ext_key: customer_key})

    assert c.ensure_term_concept(split.terms["Customer Ext"]) == customer
    assert c.ensure_term_concept(split.terms["Customer"]) == customer
    assert (
        c.ensure_concept_key(customer, split.columns["customer_ext.customer_id"])
        == customer_key
    )


def test_a_certified_name_survives_a_merge(split) -> None:
    customer = c.ensure_term_concept(split.terms["Customer"])
    ext = c.ensure_term_concept(split.terms["Customer Ext"])
    store().query_write(
        s.concept.update().where(s.concept.c.id == customer).values(certified=True)
    )

    assert c.merge_concepts(customer, ext, name="Renamed")
    assert _concept(customer)["name"] == f"{split.prefix}-Customer"


def test_concepts_of_different_databases_never_merge(split) -> None:
    other = World(f"{split.prefix}-other")
    try:
        other.table("customer", ("customer_id",), pk="customer_id")
        theirs = c.ensure_term_concept(other.term("Customer", "customer"))
        ours = c.ensure_term_concept(split.terms["Customer"])

        assert not c.merge_concepts(ours, theirs)
        assert _concept(theirs) is not None
    finally:
        store().query_write(
            s.catalog_database.delete().where(s.catalog_database.c.id == other.database)
        )


def test_each_field_names_the_member_it_comes_from(split) -> None:
    customer = c.ensure_term_concept(split.terms["Customer"])
    ext = c.ensure_term_concept(split.terms["Customer Ext"])
    c.merge_concepts(customer, ext)

    fields = {
        row["attribute_name"]: (
            row["term_id"],
            f"{row['schema_name']}.{row['table_name']}.{row['column_name']}",
            row["database_name"],
        )
        for row in c.fetch_concept_fields([customer])
    }
    assert fields == {
        f"{split.prefix}-Customer-name": (
            split.terms["Customer"],
            "sales.customer.name",
            split.prefix,
        ),
        f"{split.prefix}-Customer Ext-segment": (
            split.terms["Customer Ext"],
            "sales.customer_ext.segment",
            split.prefix,
        ),
    }


def test_merge_candidates_list_key_columns_with_their_concept(split) -> None:
    for term in split.terms.values():
        c.ensure_term_concept(term)

    rows = c.fetch_concept_merge_candidates(split.prefix)

    assert {(r["table_name"], r["column_name"]) for r in rows} == {
        ("customer", "customer_id"),
        ("customer_ext", "customer_id"),
        ("orders", "order_id"),
    }
    assert all(r["concept_id"] and r["is_pk"] for r in rows)


def test_a_term_without_a_key_gets_no_concept_when_one_is_required(world) -> None:
    world.table("events", ("event_id", "kind"), unique=("event_id",))
    term = world.term("Event", "events")

    assert c.ensure_term_concept(term, require_key=True) is None
    assert c.fetch_table_concept_id(world.tables["events"]) is None


def test_a_term_with_a_pk_gets_its_concept_when_a_key_is_required(world) -> None:
    world.table("customer", ("customer_id",), pk="customer_id")
    concept = c.ensure_term_concept(
        world.term("Customer", "customer"), require_key=True
    )

    assert concept is not None
    assert c.fetch_column_key_id(world.columns["customer.customer_id"]) is not None


def test_a_column_key_creates_its_concept_on_demand(world) -> None:
    world.table("events", ("event_id", "kind"), unique=("event_id",))
    world.term("Event", "events")

    key = c.ensure_column_key(world.columns["events.event_id"])

    assert key is not None
    assert c.fetch_table_concept_id(world.tables["events"]) is not None


def test_keyless_concepts_are_deleted_and_keyed_ones_kept(world) -> None:
    world.table("customer", ("customer_id",), pk="customer_id")
    world.table("events", ("event_id",))
    keyed = c.ensure_term_concept(world.term("Customer", "customer"))
    keyless = c.ensure_term_concept(world.term("Event", "events"))

    assert c.delete_keyless_concepts([world.prefix]) == 1
    assert _concept(keyless) is None
    assert _concept(keyed) is not None
    assert c.delete_keyless_concepts([]) == 0


def test_merge_candidates_include_terms_without_a_concept(split) -> None:
    rows = c.fetch_concept_merge_candidates(split.prefix)

    assert {(r["table_name"], r["column_name"]) for r in rows} == {
        ("customer", "customer_id"),
        ("customer_ext", "customer_id"),
        ("orders", "order_id"),
    }
    assert all(r["concept_id"] is None and r["term_id"] for r in rows)
