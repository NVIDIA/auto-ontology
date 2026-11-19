# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ColumnAttribute, SEMANTIC_FK, and the join-path traversal.

The traversal gets most of the attention here, and one case in it gets more than
the rest: **two columns that both point at the same key must not be joined
to each other.** It is a wrong answer that looks completely reasonable — a join between two
``customer_id`` columns is exactly the sort of thing a reviewer nods at.
"""

from __future__ import annotations

import os
import uuid

import pytest

pytest.importorskip("sqlalchemy")

from sqlalchemy import select  # noqa: E402

from auto_ontology.dal import attributes as a  # noqa: E402
from auto_ontology.dal import schema as s  # noqa: E402
from auto_ontology.dal.concepts import (  # noqa: E402
    ensure_column_key,
    ensure_term_concept,
    fetch_column_key_id,
    fetch_table_concept_id,
    merge_concepts,
)
from auto_ontology.dal.session import store  # noqa: E402
from auto_ontology.dal.tags import PII_TAG_NAME, get_or_create_tag  # noqa: E402
from auto_ontology.semantic.constants import SEMANTIC_SOURCE  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def require_schema():
    if not os.environ.get("POSTGRES_USER"):
        pytest.skip("POSTGRES_* not set")
    try:
        store().query_read("SELECT 1 FROM catalog_table LIMIT 1")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"auto_ontology schema unavailable (alembic upgrade head): {exc}")


def _add(table, **values) -> str:
    return store().query_write(table.insert().values(**values).returning(table.c.id))[
        0
    ]["id"]


def _link(table, **values) -> None:
    store().query_write(table.insert().values(**values))


class World:
    def __init__(self, prefix: str, database: str = "shopdb") -> None:
        self.prefix = prefix
        self.databases: dict[str, str] = {}
        self.schemas: dict[str, str] = {}
        self.tables: dict[str, str] = {}
        self.columns: dict[str, str] = {}
        self.attributes: dict[str, str] = {}
        self.owners: dict[str, str] = {}
        self.table_terms: dict[str, str] = {}
        # Cleaned up by id, not by name pattern: `update_column_attribute`
        # renames rows, and a renamed row no longer matches the prefix -- which
        # leaks it into the next test and collides on the 5-part merge key.
        self.created: list[tuple] = []
        self.database(database)

    def database(self, name: str) -> str:
        did = _add(s.catalog_database, name=f"{self.prefix}-{name}")
        self.databases[name] = did
        sid = _add(s.catalog_schema, database_id=did, name="public")
        self.schemas[name] = sid
        return did

    def table(
        self, name: str, database: str = "shopdb", pk: list[str] | None = None
    ) -> str:
        tid = _add(s.catalog_table, schema_id=self.schemas[database], name=name, pk=pk)
        self.tables[name] = tid
        return tid

    def column(self, table: str, name: str, position: int = 1) -> str:
        cid = _add(
            s.catalog_column,
            table_id=self.tables[table],
            name=name,
            ordinal_position=position,
        )
        self.columns[f"{table}.{name}"] = cid
        return cid

    def term(self, name: str, description: str | None = None) -> str:
        tid = _add(
            s.term,
            name=f"{self.prefix}-{name}",
            description=description,
            source=SEMANTIC_SOURCE,
        )
        self.created.append((s.term, tid))
        return tid

    def attribute(self, key: str, owner: str | None = None) -> str:
        """An attribute, optionally owned (HAS_ATTRIBUTE) by a column."""
        aid = _add(
            s.column_attribute,
            name=f"{self.prefix}-{key}",
            source_column=key,
            term_name=key,
            table_id="",
        )
        self.attributes[key] = aid
        self.created.append((s.column_attribute, aid))
        if owner:
            self.owners[key] = owner
            _link(
                s.column__has_attribute, column_id=self.columns[owner], attribute_id=aid
            )
        return aid

    def represent(self, table: str) -> str:
        """The Term representing *table*, and so its one-member Concept."""
        if table not in self.table_terms:
            term_id = self.term(f"{table}-term")
            _link(s.table__term, table_id=self.tables[table], term_id=term_id)
            self.table_terms[table] = term_id
            ensure_term_concept(term_id)
        return self.table_terms[table]

    def concept(self, table: str) -> str:
        self.represent(table)
        concept_id = fetch_table_concept_id(self.tables[table])
        assert concept_id
        return concept_id

    def key(self, column: str) -> str:
        """The Concept key *column* stands for, on its own table's Concept."""
        self.represent(column.split(".")[0])
        key_id = ensure_column_key(self.columns[column])
        assert key_id
        return key_id

    def references(
        self,
        column: str,
        attribute: str,
        *,
        use_in_join_paths: bool = True,
    ) -> None:
        """*column* holds a semantic FK to the key of *attribute*'s owner column."""
        _link(
            s.column__semantic_fk,
            column_id=self.columns[column],
            key_id=self.key(self.owners[attribute]),
            use_in_join_paths=use_in_join_paths,
        )


@pytest.fixture
def world():
    w = World(f"a-{uuid.uuid4().hex[:8]}")
    yield w
    for database_id in w.databases.values():
        store().query_write(
            s.catalog_database.delete().where(s.catalog_database.c.id == database_id)
        )
    for table, row_id in w.created:
        store().query_write(table.delete().where(table.c.id == row_id))
    # Anything merge_column_attribute created, which World never saw.
    store().query_write(
        s.column_attribute.delete().where(
            s.column_attribute.c.term_name.like(f"{w.prefix}%")
        )
    )


# --------------------------------------------------------------------------
# find_join_path
# --------------------------------------------------------------------------


@pytest.fixture
def joined(world):
    """`orders.customer_id` references the attribute `customers.id` owns."""
    world.table("customers", pk=["id"])
    world.column("customers", "id")
    world.table("orders")
    world.column("orders", "customer_id")
    world.attribute("customer-id", owner="customers.id")
    world.references("orders.customer_id", "customer-id")
    return world


def test_a_two_column_join_is_one_hop(joined) -> None:
    hops = a.find_join_path(
        joined.columns["orders.customer_id"], joined.columns["customers.id"]
    )
    assert hops == [
        {
            "source_database": f"{joined.prefix}-shopdb",
            "source_schema": "public",
            "source_table": "orders",
            "source_column": "customer_id",
            "target_database": f"{joined.prefix}-shopdb",
            "target_schema": "public",
            "target_table": "customers",
            "target_column": "id",
            "key": f"{joined.prefix}-customers-term.id",
        }
    ]


def test_join_path_uses_only_the_selected_parallel_semantic_fk(world) -> None:
    world.table("customers", pk=["id"])
    world.column("customers", "id")
    world.attribute("customer-id", owner="customers.id")
    world.table("orders")
    world.column("orders", "billing_customer_id", 1)
    world.column("orders", "customer_id", 2)
    world.references(
        "orders.billing_customer_id",
        "customer-id",
        use_in_join_paths=False,
    )
    world.references("orders.customer_id", "customer-id")

    hops = a.find_join_path(
        world.columns["orders.billing_customer_id"],
        world.columns["customers.id"],
    )

    assert len(hops) == 1
    assert hops[0]["source_column"] == "customer_id"
    assert hops[0]["target_column"] == "id"


def test_adjacent_tables_get_every_parallel_fk_default_first(world) -> None:
    world.table("customers", pk=["id"])
    world.column("customers", "id")
    world.attribute("customer-id", owner="customers.id")
    world.table("orders")
    world.column("orders", "billing_customer_id", 1)
    world.column("orders", "customer_id", 2)
    world.references(
        "orders.billing_customer_id", "customer-id", use_in_join_paths=False
    )
    world.references("orders.customer_id", "customer-id")

    routes = a.find_join_routes(
        world.columns["orders.billing_customer_id"], world.columns["customers.id"]
    )

    assert [
        (
            route["default"],
            [(h["source_column"], h["target_column"], h["key"]) for h in route["hops"]],
        )
        for route in routes
    ] == [
        (True, [("customer_id", "id", f"{world.prefix}-customers-term.id")]),
        (False, [("billing_customer_id", "id", f"{world.prefix}-customers-term.id")]),
    ]


def test_a_failed_key_label_lookup_keeps_the_routes(world, monkeypatch) -> None:
    world.table("customers", pk=["id"])
    world.column("customers", "id")
    world.attribute("customer-id", owner="customers.id")
    world.table("orders")
    world.column("orders", "customer_id")
    world.references("orders.customer_id", "customer-id")

    def fail(key_ids: set[str]) -> dict[str, str]:
        raise RuntimeError("lookup failed")

    monkeypatch.setattr(a, "_key_labels", fail)
    routes = a.find_join_routes(
        world.columns["orders.customer_id"], world.columns["customers.id"]
    )

    assert [
        [(h["source_column"], h["target_column"], h["key"]) for h in route["hops"]]
        for route in routes
    ] == [[("customer_id", "id", "")]]


def test_a_turned_off_fk_is_offered_directly_but_never_walked(world) -> None:
    """A longer route only crosses default edges; one hop offers every edge."""
    world.table("addresses", pk=["id"])
    world.column("addresses", "id")
    world.attribute("address-id", owner="addresses.id")
    world.table("customers", pk=["id"])
    world.column("customers", "id", 1)
    world.column("customers", "address_id", 2)
    world.attribute("customer-id", owner="customers.id")
    world.references("customers.address_id", "address-id")
    world.table("orders")
    world.column("orders", "billing_customer_id", 1)
    world.references(
        "orders.billing_customer_id", "customer-id", use_in_join_paths=False
    )

    direct = a.find_join_routes(
        world.columns["orders.billing_customer_id"], world.columns["customers.id"]
    )
    assert [route["default"] for route in direct] == [False]

    assert (
        a.find_join_routes(
            world.columns["orders.billing_customer_id"], world.columns["addresses.id"]
        )
        == []
    )


def test_setting_join_default_disables_every_other_group_member(world) -> None:
    world.table("customers", pk=["id"])
    world.column("customers", "id")
    world.attribute("customer-id", owner="customers.id")
    world.table("orders")
    world.column("orders", "billing_customer_id", 1)
    world.column("orders", "customer_id", 2)
    world.references("orders.billing_customer_id", "customer-id")
    world.references("orders.customer_id", "customer-id")

    group = a.fetch_semantic_fk_join_groups(f"{world.prefix}-shopdb")
    assert {
        (row["source_column_name"], row["target_column_name"]) for row in group
    } == {
        ("billing_customer_id", "id"),
        ("customer_id", "id"),
    }

    assert {row["target_concept_id"] for row in group} == {world.concept("customers")}

    a.set_semantic_fk_join_default(
        world.tables["orders"],
        world.concept("customers"),
        world.columns["orders.customer_id"],
    )

    rows = store().query_read(
        select(
            s.column__semantic_fk.c.column_id,
            s.column__semantic_fk.c.use_in_join_paths,
        ).where(s.column__semantic_fk.c.key_id == world.key("customers.id"))
    )
    assert {row["column_id"]: row["use_in_join_paths"] for row in rows} == {
        world.columns["orders.billing_customer_id"]: False,
        world.columns["orders.customer_id"]: True,
    }


def test_two_referencing_tables_join_through_the_attribute_owner(joined) -> None:
    """Sibling FKs route through their owner table, never directly to each other."""
    joined.table("invoices")
    joined.column("invoices", "customer_id")
    joined.references("invoices.customer_id", "customer-id")

    hops = a.find_join_path(
        joined.columns["orders.customer_id"],
        joined.columns["invoices.customer_id"],
    )
    assert [(hop["source_table"], hop["source_column"]) for hop in hops] == [
        ("orders", "customer_id"),
        ("customers", "id"),
    ]
    assert [(hop["target_table"], hop["target_column"]) for hop in hops] == [
        ("customers", "id"),
        ("invoices", "customer_id"),
    ]


def test_an_fk_to_a_repeating_column_is_refused(world) -> None:
    """A key is an identity, so a column profiled non-unique cannot be one.

    `buyers.buyer_status_id` and `sellers.seller_status_id` both name values of
    `lookup.status_code`, which repeats. Joining the two through `lookup` would
    be fabricated: nothing says a buyer and a seller with the same status are
    related. So no FK lands there, and no path routes through it.
    """
    world.table("lookup")
    world.column("lookup", "id", 1)
    world.column("lookup", "status_code", 2)
    store().query_write(
        s.catalog_column.update()
        .where(s.catalog_column.c.id == world.columns["lookup.status_code"])
        .values(is_unique=False)
    )
    world.attribute("status-code", owner="lookup.status_code")
    world.represent("lookup")
    world.table("buyers")
    world.column("buyers", "buyer_status_id")
    world.table("sellers")
    world.column("sellers", "seller_status_id")

    for column in ("buyers.buyer_status_id", "sellers.seller_status_id"):
        assert (
            a.merge_semantic_fk(world.columns[column], world.attributes["status-code"])
            is False
        )
    assert (
        a.find_join_path(
            world.columns["buyers.buyer_status_id"],
            world.columns["sellers.seller_status_id"],
        )
        == []
    )


@pytest.fixture
def split(world):
    """`customers` and `customer_ext` are one Concept; `orders` references it.

    `orders.customer_id` was resolved against `customers.id`; `customer_ext`
    lives in another schema, and keys the same customers by `customer_id`.
    """
    world.table("customers", pk=["id"])
    world.column("customers", "id", 1)
    world.column("customers", "name", 2)
    world.schemas["crm"] = _add(
        s.catalog_schema, database_id=world.databases["shopdb"], name="crm"
    )
    tid = _add(
        s.catalog_table,
        schema_id=world.schemas["crm"],
        name="customer_ext",
        pk=["customer_id"],
    )
    world.tables["customer_ext"] = tid
    world.column("customer_ext", "customer_id", 1)
    world.column("customer_ext", "segment", 2)
    world.table("orders")
    world.column("orders", "customer_id", 1)
    world.column("orders", "total", 2)
    world.attribute("customer-id", owner="customers.id")
    world.references("orders.customer_id", "customer-id")
    assert merge_concepts(
        world.concept("customers"),
        world.concept("customer_ext"),
        key_pairs={
            world.key("customer_ext.customer_id"): world.key("customers.id"),
        },
    )
    return world


def test_a_path_enters_the_member_holding_the_field(split) -> None:
    """The FK reaches `customer_ext` on its own key column, fully qualified."""
    hops = a.find_join_path(
        split.columns["orders.total"], split.columns["customer_ext.segment"]
    )
    assert hops == [
        {
            "source_database": f"{split.prefix}-shopdb",
            "source_schema": "public",
            "source_table": "orders",
            "source_column": "customer_id",
            "target_database": f"{split.prefix}-shopdb",
            "target_schema": "crm",
            "target_table": "customer_ext",
            "target_column": "customer_id",
            "key": f"{split.prefix}-customers-term.id",
        }
    ]


def test_the_members_of_a_concept_join_on_their_key_columns(split) -> None:
    hops = a.find_join_path(
        split.columns["customers.name"], split.columns["customer_ext.segment"]
    )
    assert [
        (h["source_schema"], h["source_table"], h["source_column"]) for h in hops
    ] == [("public", "customers", "id")]
    assert [
        (h["target_schema"], h["target_table"], h["target_column"]) for h in hops
    ] == [("crm", "customer_ext", "customer_id")]


def test_the_fk_still_reaches_the_member_it_was_resolved_against(split) -> None:
    hops = a.find_join_path(
        split.columns["orders.total"], split.columns["customers.name"]
    )
    assert [(h["target_table"], h["target_column"]) for h in hops] == [
        ("customers", "id")
    ]


def test_join_path_is_oriented_from_anchor_table_in_both_directions(joined) -> None:
    forward = a.find_join_path(
        joined.columns["orders.customer_id"], joined.columns["customers.id"]
    )
    backward = a.find_join_path(
        joined.columns["customers.id"], joined.columns["orders.customer_id"]
    )
    assert (forward[0]["source_table"], forward[0]["target_table"]) == (
        "orders",
        "customers",
    )
    assert (backward[0]["source_table"], backward[0]["target_table"]) == (
        "customers",
        "orders",
    )


def test_a_column_has_no_path_to_itself(joined) -> None:
    column = joined.columns["orders.customer_id"]
    assert a.find_join_path(column, column) == []


def test_unconnected_columns_yield_no_path(world) -> None:
    world.table("alpha")
    world.column("alpha", "x")
    world.table("beta")
    world.column("beta", "y")
    assert a.find_join_path(world.columns["alpha.x"], world.columns["beta.y"]) == []


def test_a_missing_column_yields_no_path(joined) -> None:
    assert a.find_join_path("no-such-column", joined.columns["customers.id"]) == []


def test_two_schemas_are_not_connected_through_their_schema(world) -> None:
    """Schemas are not nodes in the view, so no path can route through one.

    `CONTAINS` is traversed undirected, so a Schema node in the graph would let
    a path walk up from one table and out to every other table in the database.
    Two tables in the same schema, unconnected by anything else, must not find
    each other.
    """
    world.table("alpha")
    world.column("alpha", "x")
    world.table("beta")
    world.column("beta", "y")
    # Same schema, so a Schema hop would connect them.
    assert (
        store().query_read(
            select(s.catalog_table.c.schema_id).where(
                s.catalog_table.c.id.in_([world.tables["alpha"], world.tables["beta"]])
            )
        )[0]["schema_id"]
        == world.schemas["shopdb"]
    )
    assert a.find_join_path(world.columns["alpha.x"], world.columns["beta.y"]) == []


def test_two_columns_in_the_same_table_need_no_join(world) -> None:
    """Two columns in one table resolve to the same table_id, so find_join_path
    returns early before any traversal -- no path to mistake for a join.

    A traversal over `column <-> table` containment edges would find the
    2-edge path `column -> table -> column` and mistake co-location for a real
    crossing (confirmed live case: `therapy_details`/`medchg`, no FK between
    them, reported as a false "1 hop" join). `_find_table_join_hops` walks
    only semantic FK and Concept identity edges, so it cannot.
    """
    world.table("customers")
    world.column("customers", "id", 1)
    world.column("customers", "name", 2)

    assert (
        a.find_join_path(world.columns["customers.id"], world.columns["customers.name"])
        == []
    )


def test_endpoints_are_tables_to_connect_not_required_join_columns(world) -> None:
    """Return real join keys even when neither input column is a join key."""
    world.table("customers")
    world.column("customers", "name", 1)
    world.column("customers", "address_id", 2)
    world.table("addresses")
    world.column("addresses", "id", 1)
    world.column("addresses", "city", 2)

    world.attribute("address-id", owner="addresses.id")
    world.references("customers.address_id", "address-id")

    assert a.find_join_path(
        world.columns["customers.name"], world.columns["addresses.city"]
    ) == [
        {
            "source_database": f"{world.prefix}-shopdb",
            "source_schema": "public",
            "source_table": "customers",
            "source_column": "address_id",
            "target_database": f"{world.prefix}-shopdb",
            "target_schema": "public",
            "target_table": "addresses",
            "target_column": "id",
            "key": f"{world.prefix}-addresses-term.id",
        }
    ]


def test_path_can_enter_and_leave_an_intermediate_fk_table(world) -> None:
    """Cities -> suppliers -> categories requires opposite FK directions."""
    world.table("cities", pk=["id"])
    world.column("cities", "id", 1)
    world.column("cities", "name", 2)
    world.table("suppliers")
    world.column("suppliers", "city_id", 1)
    world.column("suppliers", "category_id", 2)
    world.table("categories", pk=["id"])
    world.column("categories", "id", 1)
    world.column("categories", "name", 2)

    world.attribute("city-id", owner="cities.id")
    world.attribute("category-id", owner="categories.id")
    world.references("suppliers.city_id", "city-id")
    world.references("suppliers.category_id", "category-id")

    hops = a.find_join_path(
        world.columns["cities.name"], world.columns["categories.name"]
    )
    assert [(hop["source_table"], hop["source_column"]) for hop in hops] == [
        ("cities", "id"),
        ("suppliers", "category_id"),
    ]
    assert [(hop["target_table"], hop["target_column"]) for hop in hops] == [
        ("suppliers", "city_id"),
        ("categories", "id"),
    ]


def test_a_multi_hop_path_pairs_its_columns_correctly(world) -> None:
    """`orders -> customers -> addresses`, with only semantic crossings emitted."""
    world.table("addresses")
    world.column("addresses", "id")
    world.table("customers")
    world.column("customers", "id", 1)
    world.column("customers", "address_id", 2)
    world.table("orders")
    world.column("orders", "customer_id")

    world.attribute("address-id", owner="addresses.id")
    world.attribute("customer-id", owner="customers.id")
    world.references("customers.address_id", "address-id")
    world.references("orders.customer_id", "customer-id")

    hops = a.find_join_path(
        world.columns["orders.customer_id"], world.columns["addresses.id"]
    )
    assert [(h["source_table"], h["source_column"]) for h in hops] == [
        ("orders", "customer_id"),
        ("customers", "address_id"),
    ]
    assert [(h["target_table"], h["target_column"]) for h in hops] == [
        ("customers", "id"),
        ("addresses", "id"),
    ]


def test_a_cycle_terminates(world) -> None:
    """Two tables referencing each other's attributes, plus an unreachable target.

    Without a shared visited set this walks the loop until the depth bound; with
    one it exhausts the component and stops. Either way it must return, and the
    assertion is that it does so with the right answer.
    """
    world.table("alpha")
    world.column("alpha", "id", 1)
    world.column("alpha", "beta_id", 2)
    world.table("beta")
    world.column("beta", "id", 1)
    world.column("beta", "alpha_id", 2)
    world.table("island")
    world.column("island", "x")

    world.attribute("alpha-id", owner="alpha.id")
    world.attribute("beta-id", owner="beta.id")
    world.references("beta.alpha_id", "alpha-id")
    world.references("alpha.beta_id", "beta-id")

    assert a.find_join_path(world.columns["alpha.id"], world.columns["island.x"]) == []


def test_the_depth_bound_is_enforced(world, monkeypatch) -> None:
    """A chain longer than the bound returns nothing rather than searching on.

    This test is why `MAX_PATH_DEPTH` is 30 and not 10: a *hop* is four edges,
    so even a short chain outruns a bound picked by counting hops.
    """
    length = 3
    for i in range(length):
        world.table(f"t{i}")
        world.column(f"t{i}", "id", 1)
        world.column(f"t{i}", "next_id", 2)
    for i in range(length):
        world.attribute(f"id{i}", owner=f"t{i}.id")
    for i in range(length - 1):
        world.references(f"t{i}.next_id", f"id{i + 1}")

    anchor = world.columns["t0.next_id"]
    dest = world.columns[f"t{length - 1}.id"]
    assert a.find_join_path(anchor, dest) != []

    monkeypatch.setattr(a, "MAX_PATH_DEPTH", 2)
    assert a.find_join_path(anchor, dest) == []


def test_find_table_join_hops_queries_once_regardless_of_fan_out(world) -> None:
    """The property that separates this from a recursive CTE.

    A hub attribute referenced by many columns creates many distinct routes to
    the same table. The whole database's semantic edges are fetched in one
    query and walked in memory from there, so the query count never grows
    with the number of routes through the hub, of which there are 8 here --
    unlike a per-path traversal, which would re-expand the hub once per route.
    """
    world.table("hub", pk=["id"])
    world.column("hub", "id")
    world.attribute("hub-id", owner="hub.id")
    for i in range(8):
        world.table(f"spoke{i}")
        world.column(f"spoke{i}", "hub_id")
        world.references(f"spoke{i}.hub_id", "hub-id")
    world.table("island")
    world.column("island", "x")

    calls = 0
    original = a.store

    def counting_store():
        nonlocal calls
        calls += 1
        return original()

    a.store = counting_store
    try:
        a._find_table_join_hops(
            world.tables["spoke0"], world.tables["island"], world.databases["shopdb"]
        )
    finally:
        a.store = original

    assert calls == 1


# --------------------------------------------------------------------------
# merge_column_attribute
# --------------------------------------------------------------------------


def test_merge_creates_and_links(world) -> None:
    world.table("customers")
    world.column("customers", "id")
    world.term("Customer")

    attr_id = a.merge_column_attribute(
        term_name=f"{world.prefix}-Customer",
        table_id=world.tables["customers"],
        source_column="id",
        attr_name="customer id",
        datatype="integer",
        description="the identifier",
    )
    assert attr_id

    assert (
        a.find_column_attribute_by_column_id(world.columns["customers.id"]) == attr_id
    )
    linked = store().query_read(
        select(s.column_attribute__term.c.term_id).where(
            s.column_attribute__term.c.attribute_id == attr_id
        )
    )
    assert len(linked) == 1


def test_merge_is_idempotent_and_preserves_a_curated_description(world) -> None:
    """A second merge with nothing to say must not erase what a human wrote."""
    world.table("customers")
    world.column("customers", "id")
    world.term("Customer")
    common = {
        "term_name": f"{world.prefix}-Customer",
        "table_id": world.tables["customers"],
        "source_column": "id",
        "attr_name": "customer id",
    }

    first = a.merge_column_attribute(**common, datatype="integer", description="mine")
    second = a.merge_column_attribute(**common, datatype="bigint", description=None)

    assert first == second
    row = store().query_read(
        select(s.column_attribute.c.description, s.column_attribute.c.datatype).where(
            s.column_attribute.c.id == first
        )
    )[0]
    assert row["description"] == "mine"
    # datatype describes the column, not an opinion about it, so it is assigned.
    assert row["datatype"] == "bigint"


def test_merge_writes_nothing_when_the_column_is_missing(world) -> None:
    world.table("customers")
    world.term("Customer")
    assert (
        a.merge_column_attribute(
            term_name=f"{world.prefix}-Customer",
            table_id=world.tables["customers"],
            source_column="no_such_column",
            attr_name="x",
            datatype="integer",
            description=None,
        )
        is None
    )
    assert (
        store().query_read(
            select(s.column_attribute.c.id).where(
                s.column_attribute.c.name == f"{world.prefix}-x"
            )
        )
        == []
    )


def test_merge_writes_nothing_when_the_term_is_missing(world) -> None:
    world.table("customers")
    world.column("customers", "id")
    assert (
        a.merge_column_attribute(
            term_name=f"{world.prefix}-NoSuchTerm",
            table_id=world.tables["customers"],
            source_column="id",
            attr_name="x",
            datatype="integer",
            description=None,
        )
        is None
    )


# --------------------------------------------------------------------------
# update_column_attribute
# --------------------------------------------------------------------------


@pytest.fixture
def attributed(world):
    world.table("customers")
    world.column("customers", "id")
    term = world.term("Customer")
    attr = world.attribute("customer-id", owner="customers.id")
    _link(s.column_attribute__term, attribute_id=attr, term_id=term)
    world.term_id = term
    world.attr_id = attr
    return world


def test_update_returns_the_embedding_context(attributed) -> None:
    result = a.update_column_attribute(
        attributed.attr_id,
        attributed.term_id,
        name="customer identifier",
        description="the id",
        certified=True,
    )
    assert result["name"] == "customer identifier"
    assert result["description"] == "the id"
    assert result["certified"] is True
    assert result["column_id"] == attributed.columns["customers.id"]
    assert result["term_id"] == attributed.term_id
    assert result["database_name"] == f"{attributed.prefix}-shopdb"


def test_update_leaves_omitted_fields_alone(attributed) -> None:
    """A PATCH, not a PUT — every field is coalesced."""
    a.update_column_attribute(
        attributed.attr_id, attributed.term_id, description="first", certified=True
    )
    result = a.update_column_attribute(
        attributed.attr_id, attributed.term_id, name="renamed"
    )
    assert result["name"] == "renamed"
    assert result["description"] == "first"
    assert result["certified"] is True


def test_update_rejects_a_term_that_does_not_own_the_attribute(attributed) -> None:
    """Both ids come from a URL; a mismatched pair must not edit anything."""
    other_term = attributed.term("Unrelated")
    assert (
        a.update_column_attribute(attributed.attr_id, other_term, name="hijacked")
        is None
    )
    row = store().query_read(
        select(s.column_attribute.c.name).where(
            s.column_attribute.c.id == attributed.attr_id
        )
    )[0]
    assert row["name"] != "hijacked"


def test_update_of_a_missing_attribute_is_none(attributed) -> None:
    assert a.update_column_attribute("no-such-attr", attributed.term_id) is None


# --------------------------------------------------------------------------
# fetch_attr_column_contexts
# --------------------------------------------------------------------------


def test_contexts_read_semantic_fk_backwards(joined) -> None:
    """Binds the attribute and finds the column pointing *at* it.

    The referencing column reaches the attribute through the Concept key its
    FK targets, whose member column owns the attribute.
    """
    contexts = a.fetch_attr_column_contexts(
        [joined.attributes["customer-id"]], database_name=None
    )
    context = contexts[joined.attributes["customer-id"]]
    assert context["table_name"] in {"customers", "orders"}
    assert context["schema_name"] == "public"
    assert context["database_name"] == f"{joined.prefix}-shopdb"


def test_contexts_are_empty_for_an_empty_list() -> None:
    assert a.fetch_attr_column_contexts([], database_name=None) == {}


def test_an_attribute_with_no_columns_still_gets_an_entry(world) -> None:
    """Nulls become empty strings; the attribute does not vanish."""
    orphan = world.attribute("orphan")
    context = a.fetch_attr_column_contexts([orphan], database_name=None)[orphan]
    assert context["col_id"] is None
    assert context["table_name"] == ""
    assert context["attr_name"] == f"{world.prefix}-orphan"


def test_a_database_filter_does_not_drop_the_attribute(joined) -> None:
    """The attribute survives with blank context rather than disappearing.

    Filtering in a plain WHERE over an outer join would delete the row entirely
    instead of blanking its context — a silent difference, since the caller's
    `.get` would return the same empty strings either way until something
    downstream noticed a missing key.
    """
    contexts = a.fetch_attr_column_contexts(
        [joined.attributes["customer-id"]], database_name="some-other-database"
    )
    assert joined.attributes["customer-id"] in contexts


# --------------------------------------------------------------------------
# find_unlinked_fk_columns
# --------------------------------------------------------------------------


def test_unlinked_excludes_columns_with_either_link(world) -> None:
    world.table("customers")
    world.column("customers", "id", 1)
    world.column("customers", "name", 2)
    world.column("customers", "plain", 3)
    world.table("orders")
    world.column("orders", "customer_id")

    world.attribute("customer-id", owner="customers.id")
    world.references("orders.customer_id", "customer-id")

    names = {r["name"] for r in a.find_unlinked_fk_columns(f"{world.prefix}-shopdb")}
    assert names == {"name", "plain"}


def test_unlinked_reports_a_declared_fk_target_when_there_is_one(world) -> None:
    world.table("customers")
    world.column("customers", "id")
    world.table("orders")
    world.column("orders", "customer_id")
    _link(
        s.column__foreign_key,
        source_column_id=world.columns["orders.customer_id"],
        target_column_id=world.columns["customers.id"],
    )

    rows = {r["name"]: r for r in a.find_unlinked_fk_columns(f"{world.prefix}-shopdb")}
    assert rows["customer_id"]["fk_target_col_id"] == world.columns["customers.id"]
    # A column with no declared key still appears -- those are the ones the
    # semantic resolver exists for.
    assert rows["id"]["fk_target_col_id"] is None


def test_unlinked_keeps_a_key_column_only_with_a_declared_fk(world) -> None:
    """A key that also references another entity's key is a declared FK."""
    world.table("customers", pk=["id"])
    world.column("customers", "id")
    world.table("vips", pk=["customer_id"])
    world.column("vips", "customer_id")
    world.represent("customers")
    world.represent("vips")
    _link(
        s.column__foreign_key,
        source_column_id=world.columns["vips.customer_id"],
        target_column_id=world.columns["customers.id"],
    )

    names = {r["name"] for r in a.find_unlinked_fk_columns(f"{world.prefix}-shopdb")}
    assert names == {"customer_id"}


def test_unlinked_scopes_to_one_database(world) -> None:
    world.database("otherdb")
    world.table("here")
    world.column("here", "x")
    world.table("there", database="otherdb")
    world.column("there", "y")

    scoped = {r["name"] for r in a.find_unlinked_fk_columns(f"{world.prefix}-shopdb")}
    assert scoped == {"x"}
    assert "y" in {r["name"] for r in a.find_unlinked_fk_columns()}


def test_unlinked_carries_the_columns_stored_distinct_values(world) -> None:
    world.table("orders")
    world.column("orders", "customer_ref")
    store().query_write(
        s.catalog_column.update()
        .where(s.catalog_column.c.id == world.columns["orders.customer_ref"])
        .values(distinct_values="[1001, 1002]", data_type="integer")
    )

    (row,) = a.find_unlinked_fk_columns(f"{world.prefix}-shopdb")
    assert row["distinct_values"] == "[1001, 1002]"
    assert row["data_type"] == "integer"
    assert row["schema_name"] == "public"


# --------------------------------------------------------------------------
# fetch_value_key_candidates
# --------------------------------------------------------------------------


def _mark_unique(column: str, world: World, distinct_values: str | None) -> None:
    store().query_write(
        s.catalog_column.update()
        .where(s.catalog_column.c.id == world.columns[column])
        .values(is_unique=True, distinct_values=distinct_values)
    )


def test_value_key_candidates_are_unique_attribute_owners(world) -> None:
    world.table("customers")
    world.column("customers", "id", 1)
    world.column("customers", "email", 2)
    world.column("customers", "status", 3)
    world.attribute("customer-id", owner="customers.id")
    world.attribute("customer-email", owner="customers.email")
    _mark_unique("customers.id", world, "[1, 2]")
    _mark_unique("customers.status", world, '["a"]')

    (row,) = a.fetch_value_key_candidates(f"{world.prefix}-shopdb")

    assert row == {
        "attribute_id": world.attributes["customer-id"],
        "column_id": world.columns["customers.id"],
        "column_name": "id",
        "data_type": None,
        "distinct_values": "[1, 2]",
        "table_id": world.tables["customers"],
        "table_name": "customers",
        "schema_name": "public",
    }


def test_value_key_candidates_exclude_pii_tagged_columns(world) -> None:
    world.table("customers")
    world.column("customers", "id", 1)
    world.column("customers", "email", 2)
    world.attribute("customer-id", owner="customers.id")
    world.attribute("customer-email", owner="customers.email")
    _mark_unique("customers.id", world, None)
    _mark_unique("customers.email", world, None)
    pii_tag = get_or_create_tag(name=PII_TAG_NAME)
    _link(
        s.tag_target, tag_id=pii_tag["id"], column_id=world.columns["customers.email"]
    )

    rows = a.fetch_value_key_candidates(f"{world.prefix}-shopdb")

    assert [row["column_name"] for row in rows] == ["id"]


def test_value_key_candidates_scope_to_one_database(world) -> None:
    world.database("otherdb")
    world.table("there", database="otherdb")
    world.column("there", "id")
    world.attribute("there-id", owner="there.id")
    _mark_unique("there.id", world, None)

    assert a.fetch_value_key_candidates(f"{world.prefix}-shopdb") == []
    assert len(a.fetch_value_key_candidates(f"{world.prefix}-otherdb")) == 1


# --------------------------------------------------------------------------
# Attribute descriptions
# --------------------------------------------------------------------------


@pytest.fixture
def describable(world):
    """``orders.customer_id`` -> ``customers.id``; the Customer Term in family NEW_."""
    world.table("customers")
    world.column("customers", "id", 1)
    world.column("customers", "name", 2)
    world.table("orders")
    world.column("orders", "customer_id")
    term = world.term("Customer", description="A buyer")
    for key, owner in (("customer-id", "customers.id"), ("name", "customers.name")):
        _link(
            s.column_attribute__term,
            attribute_id=world.attribute(key, owner=owner),
            term_id=term,
        )
    world.references("orders.customer_id", "customer-id")
    family = _add(s.term_family, schema_id=world.schemas["shopdb"], prefix="NEW_")
    _link(s.term__term_family, term_id=term, family_id=family, base_name="customers")
    return world


def _describe(world, key: str, text: str | None) -> None:
    store().query_write(
        s.column_attribute.update()
        .where(s.column_attribute.c.id == world.attributes[key])
        .values(description=text)
    )


def test_attributes_to_describe_skip_described_ones(describable) -> None:
    _describe(describable, "name", "Curated")
    _describe(describable, "customer-id", "  ")

    (row,) = a.fetch_attributes_to_describe(f"{describable.prefix}-shopdb")

    assert row["attribute_id"] == describable.attributes["customer-id"]
    assert row["column_id"] == describable.columns["customers.id"]
    assert row["term_description"] == "A buyer"
    assert row["table_name"] == "customers"
    assert row["family_prefix"] == "NEW_"


def test_semantic_fk_ends_name_the_owning_column(describable) -> None:
    (end,) = a.fetch_semantic_fk_ends(f"{describable.prefix}-shopdb")

    assert end == {
        "source_column_id": describable.columns["orders.customer_id"],
        "source_table_name": "orders",
        "source_column_name": "customer_id",
        "target_concept_id": describable.concept("customers"),
        "target_attribute_id": describable.attributes["customer-id"],
        "target_table_name": "customers",
        "target_column_name": "id",
    }


def test_storing_attribute_descriptions_keeps_a_curated_one(describable) -> None:
    _describe(describable, "name", "Curated")

    written = a.store_attribute_descriptions(
        {
            describable.attributes["customer-id"]: "Identifies the customer.",
            describable.attributes["name"]: "Generated",
        }
    )

    assert written == 1
    rows = store().query_read(
        select(s.column_attribute.c.id, s.column_attribute.c.description).where(
            s.column_attribute.c.id.in_(list(describable.attributes.values()))
        )
    )
    assert {r["id"]: r["description"] for r in rows} == {
        describable.attributes["customer-id"]: "Identifies the customer.",
        describable.attributes["name"]: "Curated",
    }


def test_update_context_carries_the_column_description_and_family(
    describable,
) -> None:
    store().query_write(
        s.catalog_column.update()
        .where(s.catalog_column.c.id == describable.columns["customers.name"])
        .values(description="Full name")
    )
    term_id = store().query_read(
        select(s.column_attribute__term.c.term_id).where(
            s.column_attribute__term.c.attribute_id == describable.attributes["name"]
        )
    )[0]["term_id"]

    row = a.update_column_attribute(describable.attributes["name"], term_id)

    assert row["column_description"] == "Full name"
    assert row["family_prefix"] == "NEW_"


# --------------------------------------------------------------------------
# fetch_column_attribute_columns_map
# --------------------------------------------------------------------------


def test_columns_map_separates_primary_from_referenced(joined) -> None:
    result = a.fetch_column_attribute_columns_map([joined.attributes["customer-id"]])
    entry = result[joined.attributes["customer-id"]]

    assert entry["primary_column"]["column_name"] == "id"
    assert entry["primary_column"]["table_name"] == "customers"
    assert entry["primary_column"]["db_id"] == joined.databases["shopdb"]
    assert [c["table_name"] for c in entry["referenced_columns"]] == ["orders"]


def test_columns_map_defaults_every_requested_id(world) -> None:
    """Callers read these keys directly rather than defaulting them."""
    result = a.fetch_column_attribute_columns_map(["no-such-attr"])
    assert result == {
        "no-such-attr": {"primary_column": None, "referenced_columns": []}
    }


def test_columns_map_is_empty_for_an_empty_list() -> None:
    assert a.fetch_column_attribute_columns_map([]) == {}


# --------------------------------------------------------------------------
# merge_semantic_fk
# --------------------------------------------------------------------------


def test_merge_semantic_fk_is_idempotent(world) -> None:
    world.table("customers")
    world.column("customers", "id")
    world.table("orders")
    world.column("orders", "customer_id")
    attr = world.attribute("customer-id", owner="customers.id")
    world.represent("customers")

    for _ in range(2):
        assert a.merge_semantic_fk(world.columns["orders.customer_id"], attr)

    rows = store().query_read(
        select(s.column__semantic_fk.c.key_id).where(
            s.column__semantic_fk.c.column_id == world.columns["orders.customer_id"]
        )
    )
    assert rows == [{"key_id": world.key("customers.id")}]


def test_merge_semantic_fk_refuses_a_key_column_as_its_source(world) -> None:
    """A Concept key column is the entity's identity, not a reference."""
    world.table("customers", pk=["id"])
    world.column("customers", "id")
    world.table("vips", pk=["customer_id"])
    world.column("vips", "customer_id")
    attr = world.attribute("customer-id", owner="customers.id")
    world.represent("customers")
    world.represent("vips")

    assert a.merge_semantic_fk(world.columns["vips.customer_id"], attr) is False


def test_merge_semantic_fk_takes_a_declared_fk_from_a_key_column(world) -> None:
    world.table("customers", pk=["id"])
    world.column("customers", "id")
    world.table("vips", pk=["customer_id"])
    world.column("vips", "customer_id")
    world.represent("customers")
    world.represent("vips")

    assert a.merge_semantic_fk_to_column(
        world.columns["vips.customer_id"], world.columns["customers.id"], declared=True
    )
    rows = store().query_read(
        select(s.column__semantic_fk.c.key_id).where(
            s.column__semantic_fk.c.column_id == world.columns["vips.customer_id"]
        )
    )
    assert rows == [{"key_id": world.key("customers.id")}]


def test_merge_semantic_fk_refuses_a_declared_fk_within_one_concept(world) -> None:
    world.table("customers", pk=["id"])
    world.column("customers", "id", 1)
    world.column("customers", "code", 2)
    world.represent("customers")

    assert (
        a.merge_semantic_fk_to_column(
            world.columns["customers.id"],
            world.columns["customers.code"],
            declared=True,
        )
        is False
    )
    assert fetch_column_key_id(world.columns["customers.code"]) is None


def test_merge_semantic_fk_needs_a_concept_on_the_target_table(world) -> None:
    world.table("customers")
    world.column("customers", "id")
    world.table("orders")
    world.column("orders", "customer_id")
    attr = world.attribute("customer-id", owner="customers.id")

    assert a.merge_semantic_fk(world.columns["orders.customer_id"], attr) is False


def test_merge_semantic_fk_refuses_a_cross_database_edge(world) -> None:
    """A column and attribute in different ingested databases must never link.

    This is the only place a SEMANTIC_FK edge is created, so it is the only
    place that has to enforce this — every other writer (compilation, test
    fixtures, dev tooling) relies on this guard instead of checking first.
    """
    world.database("otherdb")
    world.table("customers")
    world.column("customers", "id")
    world.table("remote_orders", database="otherdb")
    world.column("remote_orders", "customer_id")
    attr = world.attribute("customer-id", owner="customers.id")

    world.represent("customers")

    written = a.merge_semantic_fk(world.columns["remote_orders.customer_id"], attr)
    assert written is False

    rows = store().query_read(
        select(s.column__semantic_fk.c.column_id).where(
            s.column__semantic_fk.c.column_id
            == world.columns["remote_orders.customer_id"]
        )
    )
    assert rows == []
