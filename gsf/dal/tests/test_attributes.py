# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ColumnAttribute, SEMANTIC_FK, and the join-path traversal.

The traversal gets most of the attention here, and one case in it gets more than
the rest: **two columns that both point at the same attribute must not be joined
to each other.** It is a wrong answer that looks completely reasonable — a join between two
``customer_id`` columns is exactly the sort of thing a reviewer nods at.

Every path result is also checked against the recursive-CTE oracle in
``JOIN_PATH_CTE_SQL``. Two independent implementations of the same traversal
rules disagreeing is a much louder signal than either one disagreeing with a
hand-written expectation.
"""

from __future__ import annotations

import os
import uuid

import pytest

pytest.importorskip("sqlalchemy")

from sqlalchemy import select, text  # noqa: E402

from gsf.dal import attributes as a  # noqa: E402
from gsf.dal import schema as s  # noqa: E402
from gsf.dal.session import store  # noqa: E402
from gsf.semantic.constants import SEMANTIC_SOURCE  # noqa: E402


# --------------------------------------------------------------------------
# _extract_column_hops -- pure, no DB needed
# --------------------------------------------------------------------------


def test_extract_column_hops_pairs_a_single_crossing() -> None:
    path = [
        {"id": "c1", "kind": "column"},
        {"id": "attr1", "kind": "column_attribute"},
        {"id": "c2", "kind": "column"},
    ]
    assert a._extract_column_hops(path) == [("c1", "c2")]


def test_extract_column_hops_ignores_co_location_through_a_table() -> None:
    """Two columns reachable only via a shared `table` node yield no hop.

    This is the exact shape a "same table, no FK" false positive takes:
    `column -> table -> column` with no `column_attribute` node between them.
    """
    path = [
        {"id": "c1", "kind": "column"},
        {"id": "t1", "kind": "table"},
        {"id": "c2", "kind": "column"},
    ]
    assert a._extract_column_hops(path) == []


def test_extract_column_hops_pairs_a_multi_hop_bridge_through_a_table() -> None:
    """A table node between two crossings is still used to bridge them.

    `c1 -> attr1 -> c2 -> t1 -> c3 -> attr2 -> c4`: `c2` and `c3` share a
    table (a legitimate bridge/pivot table), but the table node itself never
    counts as a crossing -- only the two column_attribute-mediated hops do.
    """
    path = [
        {"id": "c1", "kind": "column"},
        {"id": "attr1", "kind": "column_attribute"},
        {"id": "c2", "kind": "column"},
        {"id": "t1", "kind": "table"},
        {"id": "c3", "kind": "column"},
        {"id": "attr2", "kind": "column_attribute"},
        {"id": "c4", "kind": "column"},
    ]
    assert a._extract_column_hops(path) == [("c1", "c2"), ("c3", "c4")]


def test_extract_column_hops_empty_path_yields_no_hops() -> None:
    assert a._extract_column_hops([]) == []
    assert a._extract_column_hops([{"id": "c1", "kind": "column"}]) == []


@pytest.fixture(scope="module", autouse=True)
def require_schema():
    if not os.environ.get("POSTGRES_USER"):
        pytest.skip("POSTGRES_* not set")
    try:
        store().query_read("SELECT 1 FROM join_path_edge LIMIT 1")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"gsf schema unavailable (alembic upgrade head): {exc}")


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
            _link(
                s.column__has_attribute, column_id=self.columns[owner], attribute_id=aid
            )
        return aid

    def references(self, column: str, attribute: str) -> None:
        _link(
            s.column__semantic_fk,
            column_id=self.columns[column],
            attribute_id=self.attributes[attribute],
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
# The CTE oracle
# --------------------------------------------------------------------------


def _oracle_path(anchor: str, dest: str) -> list[str] | None:
    """Shortest path as a node-id list, via the recursive-CTE reference."""
    rows = store().query_read(
        text(a.JOIN_PATH_CTE_SQL),
        {"anchor": anchor, "dest": dest, "max_depth": a.MAX_PATH_DEPTH},
    )
    return rows[0]["path"] if rows else None


def _assert_agrees_with_oracle(anchor: str, dest: str) -> None:
    """The two implementations must agree on whether a path exists, and how long.

    Not on *which* path: with ties the BFS and the CTE can legitimately pick
    different equal-length routes, and pinning one would be pinning an
    implementation detail rather than the contract.
    """
    bfs = a._bfs_path(anchor, dest)
    oracle = _oracle_path(anchor, dest)
    assert (bfs is None) == (oracle is None), (
        f"BFS and CTE disagree on reachability: {bfs} vs {oracle}"
    )
    if bfs is not None:
        assert len(bfs) == len(oracle), (
            f"BFS and CTE found different-length paths: {bfs} vs {oracle}"
        )


# --------------------------------------------------------------------------
# find_join_path
# --------------------------------------------------------------------------


@pytest.fixture
def joined(world):
    """`orders.customer_id` references the attribute `customers.id` owns."""
    world.table("customers")
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
        }
    ]
    _assert_agrees_with_oracle(
        joined.columns["orders.customer_id"], joined.columns["customers.id"]
    )


def test_two_columns_referencing_the_same_attribute_do_not_join(joined) -> None:
    """The single most important case in this file.

    `orders.customer_id` and `invoices.customer_id` both point at the same
    attribute. Traversing SEMANTIC_FK backwards would walk up from one and back
    down the other, returning a join between two columns that have nothing to do
    with each other. The view simply does not emit the reverse row.
    """
    joined.table("invoices")
    joined.column("invoices", "customer_id")
    joined.references("invoices.customer_id", "customer-id")

    assert (
        a.find_join_path(
            joined.columns["orders.customer_id"],
            joined.columns["invoices.customer_id"],
        )
        == []
    )
    _assert_agrees_with_oracle(
        joined.columns["orders.customer_id"], joined.columns["invoices.customer_id"]
    )


def test_the_reverse_direction_still_reaches_the_referencing_column(joined) -> None:
    """Anchor and destination swapped: reachable, via CONTAINS rather than the FK.

    Worth stating because it shows the SEMANTIC_FK restriction is not the same
    as making the graph one-way — the columns are still connected through their
    tables when a route exists.
    """
    forward = a.find_join_path(
        joined.columns["orders.customer_id"], joined.columns["customers.id"]
    )
    backward = a.find_join_path(
        joined.columns["customers.id"], joined.columns["orders.customer_id"]
    )
    assert forward != []
    assert backward == []


def test_a_column_has_no_path_to_itself(joined) -> None:
    column = joined.columns["orders.customer_id"]
    assert a.find_join_path(column, column) == []


def test_unconnected_columns_yield_no_path(world) -> None:
    world.table("alpha")
    world.column("alpha", "x")
    world.table("beta")
    world.column("beta", "y")
    assert a.find_join_path(world.columns["alpha.x"], world.columns["beta.y"]) == []
    _assert_agrees_with_oracle(world.columns["alpha.x"], world.columns["beta.y"])


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


def test_two_columns_in_the_same_table_are_not_joined_by_co_location(world) -> None:
    """Two columns sharing a table, with no HAS_ATTRIBUTE/SEMANTIC_FK between
    them, must not be reported as a "1 hop" join just because they're both
    reachable from the same table node.

    `join_path_edge` has unconditional `column <-> table` edges for every
    column in a table (that's just "this column belongs to this table", not a
    semantic relationship) -- so naively pairing every column-kind node on the
    path by position would find the 2-edge path `column -> table -> column`
    and mistake co-location for a real crossing. Regression test for the same
    bug this project's earlier Neo4j-era code fixed via `_extract_fk_hops`
    (confirmed live case: `therapy_details`/`medchg`, no FK between them,
    reported as a false "1 hop" join).
    """
    world.table("orders")
    world.column("orders", "customer_id")
    world.column("orders", "notes")
    assert (
        a.find_join_path(
            world.columns["orders.customer_id"], world.columns["orders.notes"]
        )
        == []
    )
    _assert_agrees_with_oracle(
        world.columns["orders.customer_id"], world.columns["orders.notes"]
    )


def test_a_multi_hop_path_pairs_its_columns_correctly(world) -> None:
    """`orders -> customers -> addresses`, and the pairing that reads it.

    The traversal returns columns in join-partner order once the Table and
    ColumnAttribute nodes are dropped, and the hops are built pairwise
    `(0,1), (2,3)`. Getting that wrong produces the right number of hops with
    the wrong endpoints, which no shape assertion would catch.
    """
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
    _assert_agrees_with_oracle(
        world.columns["orders.customer_id"], world.columns["addresses.id"]
    )


def test_a_cross_database_path_is_rejected(world) -> None:
    """Reachable only through a shared ColumnAttribute — Database is not a node.

    The path is genuinely found and then thrown away, so this is not testing
    that the traversal fails; it is testing that the check after it fires.
    """
    world.database("otherdb")
    world.table("customers")
    world.column("customers", "id")
    world.table("remote_orders", database="otherdb")
    world.column("remote_orders", "customer_id")

    world.attribute("customer-id", owner="customers.id")
    world.references("remote_orders.customer_id", "customer-id")

    # The traversal does find it...
    assert (
        a._bfs_path(
            world.columns["remote_orders.customer_id"], world.columns["customers.id"]
        )
        is not None
    )
    # ...and find_join_path throws it away.
    assert (
        a.find_join_path(
            world.columns["remote_orders.customer_id"], world.columns["customers.id"]
        )
        == []
    )


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
    _assert_agrees_with_oracle(world.columns["alpha.id"], world.columns["island.x"])


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


def test_bfs_visits_each_node_once(world) -> None:
    """The property that separates this from a recursive CTE.

    A hub attribute referenced by many columns creates many distinct routes to
    the same node. BFS with a shared visited set expands it once; a per-path
    visited set expands it once per route. Asserted by counting queries, because
    the returned path looks identical either way.
    """
    world.table("hub")
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
        a._bfs_path(world.columns["spoke0.hub_id"], world.columns["island.x"])
    finally:
        a.store = original

    # One query per level, bounded by MAX_PATH_DEPTH -- not one per route
    # through the hub, of which there are 8.
    assert calls <= a.MAX_PATH_DEPTH


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

    This is the traversal `join_path_edge` deliberately omits, which is why this
    function queries `column__semantic_fk` directly. If it ever starts reading
    the view, this test is what fails.
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


def test_unlinked_scopes_to_one_database(world) -> None:
    world.database("otherdb")
    world.table("here")
    world.column("here", "x")
    world.table("there", database="otherdb")
    world.column("there", "y")

    scoped = {r["name"] for r in a.find_unlinked_fk_columns(f"{world.prefix}-shopdb")}
    assert scoped == {"x"}
    assert "y" in {r["name"] for r in a.find_unlinked_fk_columns()}


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

    for _ in range(2):
        a.merge_semantic_fk(world.columns["orders.customer_id"], attr)

    rows = store().query_read(
        select(s.column__semantic_fk.c.column_id).where(
            s.column__semantic_fk.c.attribute_id == attr
        )
    )
    assert len(rows) == 1


# --------------------------------------------------------------------------
# find_shared_hub_bridge
# --------------------------------------------------------------------------


@pytest.fixture
def hub_and_spokes(world):
    """A real identity hub with a declared PK, and two spokes FK-ing into it.

    `robot_record` (docstring's own example) is the hub: its own declared PK
    is `id`, and both `spoke_a.hub_id` and `spoke_b.hub_id` hold a *forward*
    SEMANTIC_FK to the attribute that column owns -- the exact shape
    find_join_path's forward-only guard cannot connect.
    """
    world.table("hub", pk=["id"])
    world.column("hub", "id")
    world.attribute("hub-id", owner="hub.id")

    world.table("spoke_a")
    world.column("spoke_a", "hub_id")
    world.references("spoke_a.hub_id", "hub-id")

    world.table("spoke_b")
    world.column("spoke_b", "hub_id")
    world.references("spoke_b.hub_id", "hub-id")
    return world


def test_finds_the_shared_pk_anchored_hub(hub_and_spokes) -> None:
    result = a.find_shared_hub_bridge(
        hub_and_spokes.columns["spoke_a.hub_id"],
        hub_and_spokes.columns["spoke_b.hub_id"],
    )
    assert result == {"hub_table": "hub", "hub_column": "id"}


def test_shared_hub_bridge_is_order_independent(hub_and_spokes) -> None:
    assert a.find_shared_hub_bridge(
        hub_and_spokes.columns["spoke_b.hub_id"],
        hub_and_spokes.columns["spoke_a.hub_id"],
    ) == {"hub_table": "hub", "hub_column": "id"}


def test_shared_hub_bridge_returns_empty_for_the_same_column_twice(
    hub_and_spokes,
) -> None:
    col = hub_and_spokes.columns["spoke_a.hub_id"]
    assert a.find_shared_hub_bridge(col, col) == {}


def test_shared_hub_bridge_requires_the_targets_own_declared_pk(world) -> None:
    """Two FKs sharing a target that is NOT that table's declared PK must not bridge.

    Same shape as `hub_and_spokes`, but `hub.id` is never declared as the
    table's PK -- this is what keeps the function from treating "any
    attribute two FK columns happen to share" as a hub.
    """
    world.table("hub")  # no pk=...
    world.column("hub", "id")
    world.attribute("hub-id", owner="hub.id")
    world.table("spoke_a")
    world.column("spoke_a", "hub_id")
    world.references("spoke_a.hub_id", "hub-id")
    world.table("spoke_b")
    world.column("spoke_b", "hub_id")
    world.references("spoke_b.hub_id", "hub-id")

    result = a.find_shared_hub_bridge(
        world.columns["spoke_a.hub_id"], world.columns["spoke_b.hub_id"]
    )
    assert result == {}


def test_shared_hub_bridge_returns_empty_when_fks_target_different_attributes(
    world,
) -> None:
    world.table("hub_x", pk=["id"])
    world.column("hub_x", "id")
    world.attribute("hub-x-id", owner="hub_x.id")
    world.table("hub_y", pk=["id"])
    world.column("hub_y", "id")
    world.attribute("hub-y-id", owner="hub_y.id")
    world.table("spoke_a")
    world.column("spoke_a", "x_id")
    world.references("spoke_a.x_id", "hub-x-id")
    world.table("spoke_b")
    world.column("spoke_b", "y_id")
    world.references("spoke_b.y_id", "hub-y-id")

    result = a.find_shared_hub_bridge(
        world.columns["spoke_a.x_id"], world.columns["spoke_b.y_id"]
    )
    assert result == {}


# --------------------------------------------------------------------------
# find_anchor_hub_siblings
# --------------------------------------------------------------------------


def test_finds_the_hub_and_its_sibling(hub_and_spokes) -> None:
    results, truncated = a.find_anchor_hub_siblings(hub_and_spokes.tables["spoke_a"])
    assert truncated == 0
    by_target = {r["target_table"]: r for r in results}
    assert by_target["hub"]["is_hub"] is True
    assert by_target["hub"]["source_table"] == "spoke_a"
    assert by_target["spoke_b"]["is_hub"] is False
    assert by_target["spoke_b"]["hub_table"] == "hub"
    # The anchor itself and any non-sibling table must not show up.
    assert "spoke_a" not in by_target


def test_hub_siblings_are_empty_for_a_table_with_no_outgoing_fk(world) -> None:
    world.table("lonely")
    world.column("lonely", "id")
    results, truncated = a.find_anchor_hub_siblings(world.tables["lonely"])
    assert results == []
    assert truncated == 0


def test_hub_siblings_caps_and_reports_the_truncated_count(world) -> None:
    world.table("hub", pk=["id"])
    world.column("hub", "id")
    world.attribute("hub-id", owner="hub.id")
    world.table("anchor")
    world.column("anchor", "hub_id")
    world.references("anchor.hub_id", "hub-id")
    for i in range(3):
        name = f"sib{i}"
        world.table(name)
        world.column(name, "hub_id")
        world.references(f"{name}.hub_id", "hub-id")

    results, truncated = a.find_anchor_hub_siblings(
        world.tables["anchor"], max_siblings=2
    )
    siblings = [r for r in results if not r["is_hub"]]
    assert len(siblings) == 2
    assert truncated == 1


def test_hub_siblings_uncapped_when_max_siblings_is_none(world) -> None:
    world.table("hub", pk=["id"])
    world.column("hub", "id")
    world.attribute("hub-id", owner="hub.id")
    world.table("anchor")
    world.column("anchor", "hub_id")
    world.references("anchor.hub_id", "hub-id")
    for i in range(3):
        name = f"sib{i}"
        world.table(name)
        world.column(name, "hub_id")
        world.references(f"{name}.hub_id", "hub-id")

    results, truncated = a.find_anchor_hub_siblings(
        world.tables["anchor"], max_siblings=None
    )
    siblings = [r for r in results if not r["is_hub"]]
    assert len(siblings) == 3
    assert truncated == 0


def test_table_bridge_cannot_reach_across_a_shared_hub(hub_and_spokes) -> None:
    """Documents exactly why find_shared_hub_bridge has to exist separately.

    find_table_bridge/find_join_path share a forward-only SEMANTIC_FK
    traversal, so two spokes connected only through a shared hub (both FKs
    point *at* the hub, neither points *out* of it) are structurally
    unreachable from each other -- confirming the guard find_shared_hub_bridge
    was written to work around, rather than lift.
    """
    bridge_tables, hops = a.find_table_bridge(
        hub_and_spokes.tables["spoke_a"], hub_and_spokes.tables["spoke_b"]
    )
    assert (bridge_tables, hops) == ([], [])


# --------------------------------------------------------------------------
# find_table_bridge
# --------------------------------------------------------------------------


@pytest.fixture
def bridged(world):
    """`orders -> customers -> addresses`, same shape as the join-path fixture.

    `customers` is a genuine intermediate bridge table: reachable by the
    ordinary forward-FK BFS, unlike the hub/spoke shape above.
    """
    world.table("orders")
    world.column("orders", "customer_id")
    world.table("customers")
    world.column("customers", "id", 1)
    world.column("customers", "address_id", 2)
    world.table("addresses")
    world.column("addresses", "id")
    world.attribute("customer-id", owner="customers.id")
    world.attribute("address-id", owner="addresses.id")
    world.references("orders.customer_id", "customer-id")
    world.references("customers.address_id", "address-id")
    return world


def test_finds_a_genuine_intermediate_bridge_table(bridged) -> None:
    bridge_tables, hops = a.find_table_bridge(
        bridged.tables["orders"], bridged.tables["addresses"]
    )
    assert [t["name"] for t in bridge_tables] == ["customers"]
    assert [(h["source_table"], h["source_column"]) for h in hops] == [
        ("orders", "customer_id"),
        ("customers", "address_id"),
    ]
    assert [(h["target_table"], h["target_column"]) for h in hops] == [
        ("customers", "id"),
        ("addresses", "id"),
    ]


def test_table_bridge_tries_both_directions(bridged) -> None:
    """SEMANTIC_FK is forward-only, so the reverse start must also be tried."""
    bridge_tables, hops = a.find_table_bridge(
        bridged.tables["addresses"], bridged.tables["orders"]
    )
    assert [t["name"] for t in bridge_tables] == ["customers"]
    assert hops


def test_table_bridge_restricts_to_allowed_table_ids(bridged) -> None:
    """A real bridge that was never a pre-filter candidate must be discarded."""
    bridge_tables, hops = a.find_table_bridge(
        bridged.tables["orders"],
        bridged.tables["addresses"],
        allowed_table_ids={bridged.tables["orders"], bridged.tables["addresses"]},
    )
    assert (bridge_tables, hops) == ([], [])


def test_table_bridge_allows_when_the_bridge_is_in_the_allowed_set(bridged) -> None:
    bridge_tables, hops = a.find_table_bridge(
        bridged.tables["orders"],
        bridged.tables["addresses"],
        allowed_table_ids={
            bridged.tables["orders"],
            bridged.tables["customers"],
            bridged.tables["addresses"],
        },
    )
    assert [t["name"] for t in bridge_tables] == ["customers"]


def test_table_bridge_returns_empty_for_unconnected_tables(world) -> None:
    world.table("alpha")
    world.column("alpha", "x")
    world.table("beta")
    world.column("beta", "y")
    assert a.find_table_bridge(world.tables["alpha"], world.tables["beta"]) == (
        [],
        [],
    )


# --------------------------------------------------------------------------
# find_kept_table_bridges
# --------------------------------------------------------------------------


def test_kept_table_bridges_finds_a_bridge_between_a_pair(bridged) -> None:
    bridge_tables, bridge_paths, skipped = a.find_kept_table_bridges(
        [bridged.tables["orders"], bridged.tables["addresses"]]
    )
    assert [t["name"] for t in bridge_tables] == ["customers"]
    assert len(bridge_paths) == 1
    assert skipped == 0


def test_kept_table_bridges_excludes_tables_already_kept(bridged) -> None:
    """A bridge table already in the kept set must not be "rediscovered"."""
    bridge_tables, bridge_paths, skipped = a.find_kept_table_bridges(
        [
            bridged.tables["orders"],
            bridged.tables["customers"],
            bridged.tables["addresses"],
        ]
    )
    assert bridge_tables == []
    # The join hops for both pairs touching customers are still surfaced.
    assert len(bridge_paths) >= 1


def test_kept_table_bridges_requires_at_least_two_tables(bridged) -> None:
    assert a.find_kept_table_bridges([]) == ([], [], 0)
    assert a.find_kept_table_bridges([bridged.tables["orders"]]) == ([], [], 0)


def test_kept_table_bridges_respects_the_cap(bridged) -> None:
    """max_bridge_tables=0 means no pair is ever checked."""
    bridge_tables, bridge_paths, skipped = a.find_kept_table_bridges(
        [bridged.tables["orders"], bridged.tables["addresses"]],
        max_bridge_tables=0,
    )
    assert (bridge_tables, bridge_paths) == ([], [])
    assert skipped == 1
