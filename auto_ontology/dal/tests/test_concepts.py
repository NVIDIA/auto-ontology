# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Concepts: their keys, merging them, references to them and the joins they define."""

from __future__ import annotations

import os
import uuid

import pytest

pytest.importorskip("sqlalchemy")

from sqlalchemy import select  # noqa: E402

from auto_ontology.dal import attributes as a  # noqa: E402
from auto_ontology.dal import concepts as c  # noqa: E402
from auto_ontology.dal import schema as s  # noqa: E402
from auto_ontology.dal.session import store  # noqa: E402
from auto_ontology.dal.tests.test_attributes import World  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def require_schema():
    if not os.environ.get("POSTGRES_USER"):
        pytest.skip("POSTGRES_* not set")
    try:
        store().query_read("SELECT 1 FROM concept LIMIT 1")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"auto_ontology schema unavailable (alembic upgrade head): {exc}")


@pytest.fixture
def world():
    w = World(f"c-{uuid.uuid4().hex[:8]}")
    yield w
    for database_id in w.databases.values():
        store().query_write(
            s.catalog_database.delete().where(s.catalog_database.c.id == database_id)
        )
    for table, row_id in w.created:
        store().query_write(table.delete().where(table.c.id == row_id))
    store().query_write(s.concept.delete().where(s.concept.c.name.like(f"{w.prefix}%")))


def _concept(world: World, table: str, *key: str) -> str:
    """A concept for *table*'s new term, keyed on *key*."""
    concept_id = c.ensure_term_concept(world.describe(table, table))
    assert concept_id
    assert c.set_concept_key(
        concept_id, world.tables[table], [world.columns[f"{table}.{k}"] for k in key]
    )
    return concept_id


@pytest.fixture
def orders(world):
    """Order: ``orders`` and ``order_details`` share its key; two tables refer to it."""
    world.table("orders", pk=["id"])
    world.column("orders", "id")
    world.table("order_details", pk=["order_id"])
    world.column("order_details", "order_id")
    world.table("shipments")
    world.column("shipments", "order_id")
    world.table("invoices")
    world.column("invoices", "order_id")
    order = _concept(world, "orders", "id")
    details = _concept(world, "order_details", "order_id")
    assert c.merge_concepts(order, details)
    for table in ("shipments", "invoices"):
        assert c.merge_concept_fk(order, [world.columns[f"{table}.order_id"]])
    return world, order


@pytest.fixture
def lines(world):
    """Order line: keyed on ``(order_id, line_no)``; ``line_notes`` refers to it."""
    world.table("lines", pk=["order_id", "line_no"])
    world.column("lines", "order_id", 1)
    world.column("lines", "line_no", 2)
    world.table("line_notes")
    world.column("line_notes", "order_id", 1)
    world.column("line_notes", "line_no", 2)
    concept_id = _concept(world, "lines", "order_id", "line_no")
    return world, concept_id


# --------------------------------------------------------------------------
# Keys and merging
# --------------------------------------------------------------------------


def test_a_term_gets_exactly_one_concept(world) -> None:
    world.table("orders")
    term_id = world.describe("orders", "Order")

    concept_id = c.ensure_term_concept(term_id)

    assert concept_id
    assert c.ensure_term_concept(term_id) == concept_id
    assert c.concept_of_term(term_id) == concept_id


def test_a_key_must_be_on_its_table_and_match_the_other_members(world) -> None:
    world.table("a")
    world.column("a", "x", 1)
    world.column("a", "y", 2)
    world.table("b")
    world.column("b", "x")
    concept_id = _concept(world, "a", "x", "y")

    assert not c.set_concept_key(concept_id, world.tables["b"], [world.columns["a.x"]])
    assert not c.set_concept_key(concept_id, world.tables["b"], [world.columns["b.x"]])
    assert c.fetch_concept_keys(concept_id) == {
        world.tables["a"]: [world.columns["a.x"], world.columns["a.y"]]
    }


def test_merging_moves_terms_keys_and_references(orders) -> None:
    world, order = orders

    terms = {
        r["term_id"]
        for r in store().query_read(
            select(s.term__concept.c.term_id).where(
                s.term__concept.c.concept_id == order
            )
        )
    }
    assert len(terms) == 2
    assert set(c.fetch_concept_keys(order)) == {
        world.tables["orders"],
        world.tables["order_details"],
    }
    assert {
        r["column_id"]
        for r in store().query_read(
            select(c.SEMANTIC_FK_LINKS.c.column_id).where(
                c.SEMANTIC_FK_LINKS.c.concept_id == order
            )
        )
    } == {world.columns["shipments.order_id"], world.columns["invoices.order_id"]}


def test_merging_refuses_keys_of_different_length(orders, lines) -> None:
    _, order = orders
    _, line = lines

    assert not c.merge_concepts(order, line)
    assert c.fetch_concept_keys(line)


def test_merging_refuses_two_keys_on_one_table(world) -> None:
    world.table("t")
    world.column("t", "x", 1)
    world.column("t", "y", 2)
    first = _concept(world, "t", "x")
    second = c.ensure_term_concept(world.describe("t", "other"))
    assert second
    assert c.set_concept_key(second, world.tables["t"], [world.columns["t.y"]])

    assert not c.merge_concepts(first, second)


# --------------------------------------------------------------------------
# References
# --------------------------------------------------------------------------


def test_a_reference_must_match_the_key_length(lines) -> None:
    world, concept_id = lines

    assert c.merge_concept_fk(concept_id, [world.columns["line_notes.line_no"]]) is None


def test_a_column_references_one_concept(orders, world) -> None:
    _, order = orders
    world.table("customers")
    world.column("customers", "id")
    customer = _concept(world, "customers", "id")

    assert c.merge_concept_fk(customer, [world.columns["shipments.order_id"]]) is None


def test_a_composite_reference_takes_its_other_columns_by_name(lines) -> None:
    world, concept_id = lines

    fk_id = c.link_column_to_key_column(
        world.columns["line_notes.line_no"], world.columns["lines.line_no"]
    )

    assert fk_id
    rows = store().query_read(
        select(s.concept__fk_column.c.column_id)
        .where(s.concept__fk_column.c.fk_id == fk_id)
        .order_by(s.concept__fk_column.c.ordinal)
    )
    assert [r["column_id"] for r in rows] == [
        world.columns["line_notes.order_id"],
        world.columns["line_notes.line_no"],
    ]


def test_a_composite_reference_needs_every_key_column(world) -> None:
    world.table("lines", pk=["order_id", "line_no"])
    world.column("lines", "order_id", 1)
    world.column("lines", "line_no", 2)
    world.table("line_notes")
    world.column("line_notes", "line_no")
    _concept(world, "lines", "order_id", "line_no")

    assert (
        c.link_column_to_key_column(
            world.columns["line_notes.line_no"], world.columns["lines.line_no"]
        )
        is None
    )


def test_a_declared_key_table_is_not_rekeyed_onto_another_column(world) -> None:
    world.table("orders", pk=["id"])
    world.column("orders", "id", 1)
    world.column("orders", "status", 2)
    world.table("shipments")
    world.column("shipments", "status")
    order = _concept(world, "orders", "id")

    assert (
        c.link_column_to_key_column(
            world.columns["shipments.status"], world.columns["orders.status"]
        )
        is None
    )
    assert c.fetch_concept_keys(order) == {
        world.tables["orders"]: [world.columns["orders.id"]]
    }


def test_a_table_without_a_declared_key_is_keyed_by_what_refers_to_it(world) -> None:
    world.table("statuses")
    world.column("statuses", "code")
    world.table("orders")
    world.column("orders", "status_code")
    world.describe("statuses", "Status")

    fk_id = c.link_column_to_key_column(
        world.columns["orders.status_code"], world.columns["statuses.code"]
    )

    assert fk_id
    assert [
        k["key"] for k in c.concept_keys_containing(world.columns["statuses.code"])
    ] == [[world.columns["statuses.code"]]]


# --------------------------------------------------------------------------
# Join edges
# --------------------------------------------------------------------------


def _edges(world: World, **kwargs) -> set[tuple[str, str, str, tuple]]:
    names = {v: k for k, v in world.tables.items()}
    columns = {v: k for k, v in world.columns.items()}
    return {
        (
            e["kind"],
            names[e["src_table_id"]],
            names[e["dst_table_id"]],
            tuple((columns[x], columns[y]) for x, y in e["pairs"]),
        )
        for e in c.semantic_table_edges(world.databases["shopdb"], **kwargs)
    }


def test_references_join_every_member_and_members_join_each_other(orders) -> None:
    world, _ = orders

    edges = _edges(world)

    assert (
        "member",
        "orders",
        "order_details",
        (("orders.id", "order_details.order_id"),),
    ) in edges
    assert (
        "member",
        "order_details",
        "orders",
        (("order_details.order_id", "orders.id"),),
    ) in edges
    for table in ("shipments", "invoices"):
        column = f"{table}.order_id"
        assert ("fk", table, "orders", ((column, "orders.id"),)) in edges
        assert ("fk", "orders", table, (("orders.id", column),)) in edges
        assert (
            "fk",
            table,
            "order_details",
            ((column, "order_details.order_id"),),
        ) in edges


def test_two_references_to_one_concept_are_not_joined(orders) -> None:
    world, _ = orders

    pairs = {(src, dst) for _, src, dst, _ in _edges(world)}

    assert ("shipments", "invoices") not in pairs
    assert ("invoices", "shipments") not in pairs


def test_a_disabled_reference_is_left_out_unless_asked_for(orders) -> None:
    world, order = orders
    store().query_write(
        s.concept__fk.update()
        .where(s.concept__fk.c.source_table_id == world.tables["shipments"])
        .values(use_in_join_paths=False)
    )

    assert not any(src == "shipments" for _, src, _, _ in _edges(world))
    assert any(
        src == "shipments" for _, src, _, _ in _edges(world, include_disabled=True)
    )


def test_a_composite_reference_is_one_edge_with_every_pair(lines) -> None:
    world, _ = lines
    assert c.link_column_to_key_column(
        world.columns["line_notes.line_no"], world.columns["lines.line_no"]
    )

    assert (
        "fk",
        "line_notes",
        "lines",
        (
            ("line_notes.order_id", "lines.order_id"),
            ("line_notes.line_no", "lines.line_no"),
        ),
    ) in _edges(world)


def test_a_composite_join_path_lists_every_column_pair(lines) -> None:
    world, _ = lines
    assert c.link_column_to_key_column(
        world.columns["line_notes.line_no"], world.columns["lines.line_no"]
    )

    hops = a.find_join_path(
        world.columns["line_notes.line_no"], world.columns["lines.order_id"]
    )

    assert len(hops) == 1
    assert hops[0]["source_table"] == "line_notes"
    assert hops[0]["target_table"] == "lines"
    assert hops[0]["column_pairs"] == [
        {"source_column": "order_id", "target_column": "order_id"},
        {"source_column": "line_no", "target_column": "line_no"},
    ]


def test_concepts_for_tables_lists_only_the_tables_in_scope(orders) -> None:
    world, order = orders
    scope = [world.tables[t] for t in ("orders", "shipments", "invoices")]

    (concept,) = c.fetch_concepts_for_tables(scope)

    assert concept["id"] == order
    assert [(k["table_name"], k["columns"]) for k in concept["keys"]] == [
        ("orders", ["id"])
    ]
    assert [(r["table_name"], r["columns"]) for r in concept["references"]] == [
        ("invoices", ["order_id"]),
        ("shipments", ["order_id"]),
    ]


def test_concepts_for_tables_needs_a_key_table_in_scope(orders) -> None:
    world, _ = orders

    assert not c.fetch_concepts_for_tables(
        [world.tables["shipments"], world.tables["invoices"]]
    )
    assert not c.fetch_concepts_for_tables([world.tables["orders"]])


def test_concepts_for_tables_keeps_composite_columns_in_key_order(lines) -> None:
    world, _ = lines
    assert c.link_column_to_key_column(
        world.columns["line_notes.line_no"], world.columns["lines.line_no"]
    )

    (concept,) = c.fetch_concepts_for_tables(
        [world.tables["lines"], world.tables["line_notes"]]
    )

    assert concept["keys"][0]["columns"] == ["order_id", "line_no"]
    assert concept["references"][0]["columns"] == ["order_id", "line_no"]
