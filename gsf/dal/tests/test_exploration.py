# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Exploration graphs.

**The degree invariant is this file's reason to exist**, and the plan names it
in the module docstring: a table's ``relationship_count`` in the graph
payload must equal the ``total`` its related-nodes page reports. The two numbers
come from different code — one counts edges, the other counts neighbours — so
they agree only if "related" means exactly the same thing in both. When they
drift, the UI draws a node labelled "5 relationships" whose panel lists 4, and
nothing anywhere fails.

The fixture is therefore built to make the two disagree if anything is wrong:
edges by shared SQL *and* by foreign key, foreign keys in both directions, a
pair connected by both mechanisms at once, and a self-referential key that must
count as nothing.
"""

from __future__ import annotations

import os
import uuid

import pytest

pytest.importorskip("sqlalchemy")

from gsf.dal import exploration as e  # noqa: E402
from gsf.dal import schema as s  # noqa: E402
from gsf.dal.session import store  # noqa: E402
from gsf.semantic.constants import SEMANTIC_SOURCE  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def require_schema():
    if not os.environ.get("POSTGRES_USER"):
        pytest.skip("POSTGRES_* not set")
    try:
        store().query_read(f"SELECT 1 FROM {s.SCHEMA}.sql_query_table LIMIT 1")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"gsf schema unavailable (alembic upgrade head): {exc}")


def _add(table, **values) -> str:
    return store().query_write(table.insert().values(**values).returning(table.c.id))[
        0
    ]["id"]


def _link(table, **values) -> None:
    store().query_write(table.insert().values(**values))


class World:
    def __init__(self, prefix: str) -> None:
        self.prefix = prefix
        self.database = _add(s.catalog_database, name=prefix)
        self.schema = _add(s.catalog_schema, database_id=self.database, name="public")
        self.tables: dict[str, str] = {}
        self.columns: dict[str, str] = {}
        self.terms: dict[str, str] = {}
        self.zones: dict[str, str] = {}
        self.queries: list[str] = []

    def table(self, name: str, columns: tuple[str, ...] = ("id",)) -> str:
        tid = _add(s.catalog_table, schema_id=self.schema, name=f"{self.prefix}_{name}")
        self.tables[name] = tid
        for position, column in enumerate(columns, start=1):
            self.columns[f"{name}.{column}"] = _add(
                s.catalog_column,
                table_id=tid,
                name=column,
                ordinal_position=position,
            )
        return tid

    def statement(self, sql: str, *tables: str) -> str:
        qid = _add(s.sql_query, sql_full_query=sql)
        self.queries.append(qid)
        for table in tables:
            _link(s.sql_query_table, sql_query_id=qid, table_id=self.tables[table])
        return qid

    def foreign_key(self, source: str, target: str) -> None:
        _link(
            s.column_foreign_key,
            source_column_id=self.columns[source],
            target_column_id=self.columns[target],
        )

    def term(self, name: str, *, represents: tuple[str, ...] = ()) -> str:
        tid = _add(s.term, name=f"{self.prefix}-{name}", source=SEMANTIC_SOURCE)
        self.terms[name] = tid
        for table in represents:
            _link(s.table_term, table_id=self.tables[table], term_id=tid)
        return tid

    def zone(self, name: str, *tables: str) -> str:
        zid = _add(s.zone, name=f"{self.prefix}-{name}")
        self.zones[name] = zid
        for table in tables:
            _link(s.zone_target, zone_id=zid, table_id=self.tables[table])
        return zid


@pytest.fixture
def world():
    w = World(f"x-{uuid.uuid4().hex[:8]}")
    yield w
    for zone_id in w.zones.values():
        store().query_write(s.zone.delete().where(s.zone.c.id == zone_id))
    store().query_write(
        s.catalog_database.delete().where(s.catalog_database.c.id == w.database)
    )
    store().query_write(s.term.delete().where(s.term.c.name.like(f"{w.prefix}%")))
    for query_id in w.queries:
        store().query_write(s.sql_query.delete().where(s.sql_query.c.id == query_id))


@pytest.fixture
def graph(world):
    """Four tables wired so every edge mechanism is exercised at once.

    * ``orders`` ↔ ``customers`` — a **foreign key and** a shared statement, so
      the two must merge into one edge rather than appear twice;
    * ``orders`` ↔ ``items`` — a foreign key only, pointing the other way;
    * ``orders`` ↔ ``audit`` — a shared statement only;
    * ``orders.parent_id`` → ``orders.id`` — self-referential, and must count
      as no relationship at all.
    """
    world.table("orders", columns=("id", "customer_id", "parent_id"))
    world.table("customers", columns=("id",))
    world.table("items", columns=("id", "order_id"))
    world.table("audit", columns=("id",))

    world.foreign_key("orders.customer_id", "customers.id")
    world.foreign_key("items.order_id", "orders.id")
    world.foreign_key("orders.parent_id", "orders.id")

    world.statement(
        f"SELECT 1 -- {world.prefix} orders+customers", "orders", "customers"
    )
    world.statement(f"SELECT 2 -- {world.prefix} orders+audit", "orders", "audit")
    # No text: must not create an edge anywhere.
    world.statement("   ", "orders", "items")
    return world


def _mine(edges, world):
    ids = set(world.tables.values())
    return [e for e in edges if e["source"] in ids and e["target"] in ids]


# --------------------------------------------------------------------------
# The degree invariant
# --------------------------------------------------------------------------


def test_graph_degree_equals_the_related_page_total(graph) -> None:
    """The invariant, asserted for every table in the fixture.

    The graph counts edges; the page counts neighbours. They are different code
    reading one definition of "related", and this is the test that says so.
    """
    payload = e.fetch_data_exploration_graph()
    by_id = {node["id"]: node for node in payload["nodes"]}

    for name, table_id in graph.tables.items():
        node = by_id[table_id]
        page = e.fetch_exploration_related_nodes(table_id, "data")
        assert node["relationship_count"] == page["total"], (
            f"{name}: graph says {node['relationship_count']}, "
            f"page says {page['total']}"
        )


def test_the_expected_degrees(graph) -> None:
    """Pinned explicitly, so the invariant above cannot pass by being 0 == 0."""
    totals = {
        name: e.fetch_exploration_related_nodes(table_id, "data")["total"]
        for name, table_id in graph.tables.items()
    }
    assert totals == {"orders": 3, "customers": 1, "items": 1, "audit": 1}


def test_a_self_referential_key_is_not_a_relationship(graph) -> None:
    """`orders.parent_id -> orders.id` must not make orders its own neighbour."""
    page = e.fetch_exploration_related_nodes(graph.tables["orders"], "data")
    assert graph.tables["orders"] not in {node["id"] for node in page["nodes"]}


def test_degrees_survive_truncation(graph) -> None:
    """A neighbour left out by `limit` is still a neighbour.

    Counting after truncation would shrink a node's number to match the subset
    drawn, contradicting the page behind it.
    """
    payload = e.fetch_data_exploration_graph(limit=1)
    assert len(payload["nodes"]) == 1
    node = payload["nodes"][0]
    page = e.fetch_exploration_related_nodes(node["id"], "data")
    assert node["relationship_count"] == page["total"]


# --------------------------------------------------------------------------
# Edges
# --------------------------------------------------------------------------


def test_a_pair_connected_both_ways_is_one_merged_edge(graph) -> None:
    edges = {
        (edge["source"], edge["target"]): edge
        for edge in _mine(e.fetch_data_exploration_edges(), graph)
    }
    key = tuple(sorted((graph.tables["orders"], graph.tables["customers"])))
    edge = edges[key]

    assert edge["via_foreign_key"] is True
    assert edge["queries"] == [f"SELECT 1 -- {graph.prefix} orders+customers"]
    assert len(edge["foreign_keys"]) == 1


def test_a_foreign_key_only_pair_carries_no_queries(graph) -> None:
    edges = {
        (edge["source"], edge["target"]): edge
        for edge in _mine(e.fetch_data_exploration_edges(), graph)
    }
    key = tuple(sorted((graph.tables["orders"], graph.tables["items"])))
    assert edges[key]["queries"] == []
    assert edges[key]["via_foreign_key"] is True


def test_a_sql_only_pair_is_not_flagged_as_a_foreign_key(graph) -> None:
    edges = {
        (edge["source"], edge["target"]): edge
        for edge in _mine(e.fetch_data_exploration_edges(), graph)
    }
    key = tuple(sorted((graph.tables["orders"], graph.tables["audit"])))
    assert edges[key]["via_foreign_key"] is False
    assert edges[key]["foreign_keys"] == []


def test_a_blank_statement_makes_no_edge(world) -> None:
    """Otherwise the same pair is connected in one view and not another."""
    world.table("alpha")
    world.table("beta")
    world.statement("   ", "alpha", "beta")
    assert _mine(e.fetch_data_exploration_edges(), world) == []


def test_foreign_key_columns_follow_the_normalised_edge_direction(world) -> None:
    """The subtle one: a reversed FK row must have its column pair swapped too.

    Edge keys are normalised to (min id, max id). If the columns are not swapped
    with them, `foreign_keys[].source_column` names a column on the *target*
    table — which renders the join backwards, and looks entirely plausible.
    """
    world.table("alpha", columns=("id", "beta_id"))
    world.table("beta", columns=("id",))
    world.foreign_key("alpha.beta_id", "beta.id")

    edge = _mine(e.fetch_data_exploration_edges(), world)[0]
    detail = edge["foreign_keys"][0]

    # Whichever way the ids sorted, source_column belongs to edge["source"].
    owner = "alpha" if edge["source"] == world.tables["alpha"] else "beta"
    expected = {"alpha": ("beta_id", "id"), "beta": ("id", "beta_id")}[owner]
    assert (detail["source_column"], detail["target_column"]) == expected


def test_edges_are_deterministically_ordered(graph) -> None:
    edges = _mine(e.fetch_data_exploration_edges(), graph)
    assert edges == sorted(edges, key=lambda edge: (edge["source"], edge["target"]))


# --------------------------------------------------------------------------
# Zone scoping
# --------------------------------------------------------------------------


def test_edges_need_both_ends_in_zone(graph) -> None:
    """An edge to an invisible table is not a half-edge; it is no edge."""
    zone_id = graph.zone("Sales", "orders", "customers")
    edges = _mine(e.fetch_data_exploration_edges(zone_ids=[zone_id]), graph)

    assert len(edges) == 1
    assert {edges[0]["source"], edges[0]["target"]} == {
        graph.tables["orders"],
        graph.tables["customers"],
    }


def test_the_degree_invariant_holds_under_scoping(graph) -> None:
    """The invariant is worth nothing if it only holds for an admin."""
    zone_id = graph.zone("Sales", "orders", "customers")
    payload = e.fetch_data_exploration_graph(zone_ids=[zone_id])
    by_id = {node["id"]: node for node in payload["nodes"]}

    assert set(by_id) == {graph.tables["orders"], graph.tables["customers"]}
    for table_id, node in by_id.items():
        page = e.fetch_exploration_related_nodes(table_id, "data", zone_ids=[zone_id])
        assert node["relationship_count"] == page["total"] == 1


def test_an_out_of_zone_node_has_no_related_page(graph) -> None:
    zone_id = graph.zone("Sales", "orders")
    page = e.fetch_exploration_related_nodes(
        graph.tables["audit"], "data", zone_ids=[zone_id]
    )
    assert page == {"nodes": [], "total": 0}


def test_no_zone_ids_means_unscoped_not_empty(graph) -> None:
    assert len(_mine(e.fetch_data_exploration_edges(zone_ids=None), graph)) == 3
    assert e.fetch_data_exploration_edges(zone_ids=[]) == []


# --------------------------------------------------------------------------
# Table details
# --------------------------------------------------------------------------


def test_table_details_lists_statements_and_terms(world) -> None:
    world.table("orders", columns=("id",))
    world.statement(f"SELECT 1 -- {world.prefix}", "orders")
    world.term("Order", represents=("orders",))

    details = e.fetch_table_exploration_details(world.tables["orders"])

    assert [q["sql"] for q in details["queries"]] == [f"SELECT 1 -- {world.prefix}"]
    assert [t["name"] for t in details["terms"]] == [f"{world.prefix}-Order"]
    assert details["terms_total"] == 1


def test_table_details_finds_terms_by_either_route(world) -> None:
    """REPRESENTS and through a column's attribute — the same paths as related."""
    world.table("orders", columns=("id",))
    world.term("Order", represents=("orders",))
    world.term("Customer")
    attribute = _add(
        s.column_attribute,
        name=f"{world.prefix}-customer-id",
        source_column="id",
        term_name=f"{world.prefix}-Customer",
        table_id=world.tables["orders"],
    )
    _link(
        s.column_attribute_term,
        attribute_id=attribute,
        term_id=world.terms["Customer"],
    )
    _link(
        s.column_semantic_fk,
        column_id=world.columns["orders.id"],
        attribute_id=attribute,
    )

    details = e.fetch_table_exploration_details(world.tables["orders"])
    assert {t["name"] for t in details["terms"]} == {
        f"{world.prefix}-Order",
        f"{world.prefix}-Customer",
    }
    assert details["terms_total"] == 2

    store().query_write(
        s.column_attribute.delete().where(s.column_attribute.c.id == attribute)
    )


def test_table_details_pages_terms_and_reports_the_full_total(world) -> None:
    world.table("orders", columns=("id",))
    for i in range(3):
        world.term(f"Term{i}", represents=("orders",))

    page = e.fetch_table_exploration_details(world.tables["orders"], limit=2)
    assert len(page["terms"]) == 2
    assert page["terms_total"] == 3


def test_table_details_and_the_edge_set_disagree_about_blank(world) -> None:
    """An inconsistency in the original, preserved and pinned here.

    `_non_empty_sql` trims before deciding, so a whitespace-only statement makes
    no edge. This list only drops *falsy* text, so the same statement is listed
    among the table's queries. A user therefore sees a query on the detail panel
    that corresponds to no line on the graph.

    Harmless enough to leave — nothing is lost, and a whitespace-only statement
    should not exist in the first place — but "the two filters differ" is worth
    one failing test if anyone unifies them by accident.
    """
    world.table("orders")
    world.table("audit")
    world.statement("   ", "orders", "audit")

    details = e.fetch_table_exploration_details(world.tables["orders"])
    assert [q["sql"] for q in details["queries"]] == ["   "]
    assert _mine(e.fetch_data_exploration_edges(), world) == []


def test_table_details_drops_an_empty_statement(world) -> None:
    """Empty *is* dropped — it is only whitespace that slips through."""
    world.table("orders")
    world.statement("", "orders")
    assert e.fetch_table_exploration_details(world.tables["orders"])["queries"] == []


def test_an_out_of_zone_table_has_no_details(world) -> None:
    world.table("orders")
    world.table("secrets")
    zone_id = world.zone("Sales", "orders")
    assert e.fetch_table_exploration_details(
        world.tables["secrets"], zone_ids=[zone_id]
    ) == {"queries": [], "terms": [], "terms_total": 0}


# --------------------------------------------------------------------------
# The data graph payload
# --------------------------------------------------------------------------


def test_graph_nodes_carry_their_counts_and_zones(world) -> None:
    world.table("orders", columns=("id", "total"))
    world.statement(f"SELECT 1 -- {world.prefix}", "orders")
    world.term("Order", represents=("orders",))
    zone_id = world.zone("Sales", "orders")

    node = next(
        n
        for n in e.fetch_data_exploration_graph()["nodes"]
        if n["id"] == world.tables["orders"]
    )
    assert node["columns_count"] == 2
    assert node["sql_count"] == 1
    assert node["terms_count"] == 1
    assert [z["id"] for z in node["zones"]] == [zone_id]
    assert node["database_name"] == world.prefix


def test_links_are_filtered_to_drawn_nodes(graph) -> None:
    """A link to a truncated node would point at nothing the client has."""
    payload = e.fetch_data_exploration_graph(limit=2)
    drawn = {node["id"] for node in payload["nodes"]}
    for link in payload["links"]:
        assert link["source"] in drawn and link["target"] in drawn


def test_the_limit_is_clamped_in_both_directions(graph) -> None:
    """Whatever a caller asks for, the response stays bounded."""
    assert len(e.fetch_data_exploration_graph(limit=0)["nodes"]) == 1
    assert (
        len(e.fetch_data_exploration_graph(limit=10_000)["nodes"])
        <= e.MAX_EXPLORATION_GRAPH_NODES
    )


# --------------------------------------------------------------------------
# The semantic graph
# --------------------------------------------------------------------------


@pytest.fixture
def semantic(world):
    """Two terms sharing a table, and one that shares nothing."""
    world.table("orders", columns=("id",))
    world.table("island", columns=("id",))
    world.term("Order", represents=("orders",))
    world.term("Customer", represents=("orders",))
    world.term("Lonely", represents=("island",))
    return world


def test_semantic_nodes_and_links(semantic) -> None:
    payload = e.fetch_semantic_exploration_graph()
    by_id = {node["id"]: node for node in payload["nodes"]}

    assert by_id[semantic.terms["Order"]]["relationship_count"] == 1
    assert by_id[semantic.terms["Lonely"]]["relationship_count"] == 0

    key = tuple(sorted((semantic.terms["Order"], semantic.terms["Customer"])))
    assert {"source": key[0], "target": key[1]} in payload["links"]


def test_semantic_degree_equals_the_related_page_total(semantic) -> None:
    """The same invariant as the data layer, on the other graph."""
    payload = e.fetch_semantic_exploration_graph()
    by_id = {node["id"]: node for node in payload["nodes"]}

    for term_id in semantic.terms.values():
        page = e.fetch_exploration_related_nodes(term_id, "semantic")
        assert by_id[term_id]["relationship_count"] == page["total"]


def test_a_semantic_link_is_undirected_and_appears_once(semantic) -> None:
    payload = e.fetch_semantic_exploration_graph()
    pairs = [
        tuple(sorted((link["source"], link["target"]))) for link in payload["links"]
    ]
    assert len(pairs) == len(set(pairs))


def test_semantic_related_pages_by_name(semantic) -> None:
    page = e.fetch_exploration_related_nodes(
        semantic.terms["Order"], "semantic", limit=1
    )
    assert [node["name"] for node in page["nodes"]] == [f"{semantic.prefix}-Customer"]
    assert page["total"] == 1


def test_an_unknown_layer_raises(world) -> None:
    """A typo in a route parameter should not silently return an empty graph."""
    with pytest.raises(ValueError, match="Unsupported Exploration layer"):
        e.fetch_exploration_related_nodes("some-id", "nonsense")


def test_the_zones_map_is_the_one_from_pg_zones() -> None:
    """Re-exported, not reimplemented.

    Two spellings of the zone-resolution rule is how a viewer sees a chip on one
    screen and not another, so this asserts they are literally one function.
    """
    from gsf.dal import zones

    assert e.fetch_table_zones_map is zones.fetch_table_zones_map
