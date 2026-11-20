# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PathRAG-style join-route ranking on hand-written table graphs.

The fixture is the part of Pagila where routes tie: an address is reachable
from a rental through the customer who rented or through the staff member who
handled it, and the two routes mean different things.
"""

from __future__ import annotations

import random
import uuid

import pytest

from auto_ontology.dal.join_route_ranking import (
    DEFAULT_ALPHA,
    DEFAULT_BOOST,
    MAX_ENUMERATED_ROUTES,
    RankedRoute,
    enumerate_routes,
    flow_resources,
    rank_routes,
)

# (fk_table, fk_col, owner_table, owner_col); each FK is walkable both ways and
# keyed by its owner column, as in _semantic_table_adjacency.
PAGILA = [
    ("customer", "address_id", "address", "address_id"),
    ("customer", "store_id", "store", "store_id"),
    ("staff", "address_id", "address", "address_id"),
    ("staff", "store_id", "store", "store_id"),
    ("store", "address_id", "address", "address_id"),
    ("address", "city_id", "city", "city_id"),
    ("rental", "customer_id", "customer", "customer_id"),
    ("rental", "staff_id", "staff", "staff_id"),
    ("rental", "inventory_id", "inventory", "inventory_id"),
    ("inventory", "store_id", "store", "store_id"),
]


Edge = tuple[str, str, str, str]


def build(
    edges: list[Edge], ids: dict[str, str] | None = None
) -> dict[str, list[tuple[str, str, str, str]]]:
    """Adjacency from ``(fk_table, fk_col, owner_table, owner_col)`` edges.

    *ids* maps table names to the ids the adjacency uses; unmapped names are
    their own ids.
    """
    mapping = ids or {}

    def tid(name: str) -> str:
        return mapping.get(name, name)

    adjacency: dict[str, list[tuple[str, str, str, str]]] = {}
    for fk_t, fk_c, own_t, own_c in edges:
        src, dst = f"{fk_t}.{fk_c}", f"{own_t}.{own_c}"
        adjacency.setdefault(tid(fk_t), []).append((tid(own_t), src, dst, dst))
        adjacency.setdefault(tid(own_t), []).append((tid(fk_t), dst, src, dst))
    for value in adjacency.values():
        value.sort()
    return adjacency


def names_of(route: RankedRoute, ids: dict[str, str] | None = None) -> list[str]:
    back = {v: k for k, v in (ids or {}).items()}
    return [back.get(t, t) for t in route.tables]


def test_flow_decays_by_alpha_and_fan_out() -> None:
    resources = flow_resources(build(PAGILA), "address")
    # address joins customer, staff, store and city: four neighbours.
    assert resources["address"] == 1.0
    for neighbour in ("customer", "staff", "store", "city"):
        assert resources[neighbour] == pytest.approx(DEFAULT_ALPHA / 4)
    # rental is reached from both customer and staff (three neighbours each).
    assert resources["rental"] == pytest.approx(
        2 * DEFAULT_ALPHA * (DEFAULT_ALPHA / 4) / 3
    )


def test_symmetric_routes_tie_without_a_question_signal() -> None:
    ranked = rank_routes(build(PAGILA), "address", "rental", k=5)
    top = ranked[:2]
    assert {tuple(names_of(r)) for r in top} == {
        ("address", "customer", "rental"),
        ("address", "staff", "rental"),
    }
    assert top[0].score == top[1].score
    assert all(len(r.hops) <= 3 for r in ranked)


@pytest.mark.parametrize("asked", ["customer", "staff"])
def test_the_question_boost_breaks_the_tie(asked: str) -> None:
    ranked = rank_routes(build(PAGILA), "address", "rental", k=3, boost_tables={asked})
    assert names_of(ranked[0]) == ["address", asked, "rental"]
    assert ranked[0].score > ranked[0].reliability


def test_order_does_not_depend_on_how_ids_were_generated() -> None:
    tables = {t for e in PAGILA for t in (e[0], e[2])}
    expected = None
    for seed in range(20):
        rng = random.Random(seed)
        ids = {t: str(uuid.UUID(int=rng.getrandbits(128))) for t in tables}
        names = {v: k for k, v in ids.items()}
        ranked = rank_routes(
            build(PAGILA, ids), ids["address"], ids["rental"], k=5, table_names=names
        )
        order = [tuple(names_of(r, ids)) for r in ranked]
        expected = expected or order
        assert order == expected


def test_tables_reached_through_a_hub_carry_less_resource() -> None:
    # a -> hub -> m -> b and a -> y -> n -> b. The hub also joins five other
    # tables, so m (reached only through it) gets a smaller share than n, and
    # the route through the hub ranks lower. Same length, same anchor fan-out.
    edges = [
        ("hub", "a_id", "a", "id"),
        ("m", "hub_id", "hub", "id"),
        ("m", "b_id", "b", "id"),
        ("y", "a_id", "a", "id"),
        ("n", "y_id", "y", "id"),
        ("n", "b_id", "b", "id"),
    ]
    edges += [(f"leaf{i}", "hub_id", "hub", "id") for i in range(5)]
    adjacency = build(edges)
    resources = flow_resources(adjacency, "a")
    assert resources["hub"] == resources["y"]
    assert resources["m"] < resources["n"]

    ranked = rank_routes(adjacency, "a", "b", k=2)
    assert [names_of(r) for r in ranked] == [
        ["a", "y", "n", "b"],
        ["a", "hub", "m", "b"],
    ]
    assert ranked[0].score > ranked[1].score


def test_equal_routes_into_a_shared_destination_tie_even_through_a_hub() -> None:
    # Scoring sums resource over a route's tables, so two 2-hop routes whose
    # middle tables are both direct neighbours of the anchor tie: the hub's
    # fan-out only dilutes what flows *past* it. Ties fall back to names.
    edges = [("x", "a_id", "a", "id"), ("x", "b_id", "b", "id")]
    edges += [("hub", "a_id", "a", "id"), ("hub", "b_id", "b", "id")]
    edges += [(f"leaf{i}", "hub_id", "hub", "id") for i in range(5)]
    ranked = rank_routes(build(edges), "a", "b", k=2)
    assert ranked[0].score == ranked[1].score
    assert [names_of(r)[1] for r in ranked] == ["hub", "x"]


def test_routes_are_simple_and_bounded() -> None:
    adjacency = build(PAGILA)
    routes = enumerate_routes(adjacency, "address", "rental", max_extra_hops=1)
    assert routes
    for tables, hops in routes:
        assert len(set(tables)) == len(tables)
        assert len(hops) == len(tables) - 1
        assert len(hops) <= 3  # shortest is 2
    shortest_only = enumerate_routes(adjacency, "address", "rental", max_extra_hops=0)
    assert len(shortest_only) == 2
    assert all(len(hops) == 2 for _, hops in shortest_only)


def test_hops_pair_the_real_join_columns() -> None:
    (best,) = rank_routes(
        build(PAGILA), "address", "rental", k=1, boost_tables={"staff"}
    )
    assert best.hops == (
        ("address.address_id", "staff.address_id", "address.address_id"),
        ("staff.staff_id", "rental.staff_id", "staff.staff_id"),
    )


def test_unreachable_same_and_zero_k() -> None:
    adjacency = build(PAGILA)
    adjacency["island"] = []
    assert rank_routes(adjacency, "address", "island", k=3) == []
    assert rank_routes(adjacency, "address", "address", k=3) == []
    assert rank_routes(adjacency, "address", "rental", k=0) == []


def test_each_boosted_table_multiplies_the_score() -> None:
    # a -> x -> y -> b, with both middle tables boosted.
    edges = [("x", "a_id", "a", "id"), ("y", "x_id", "x", "id")]
    edges += [("y", "b_id", "b", "id")]
    adjacency = build(edges)
    (plain,) = rank_routes(adjacency, "a", "b", k=1)
    (boosted,) = rank_routes(adjacency, "a", "b", k=1, boost_tables={"x", "y"})
    assert boosted.reliability == plain.reliability
    assert boosted.score == pytest.approx(
        plain.reliability * (1 + DEFAULT_BOOST) ** 2, abs=1e-5
    )


def test_enumeration_stops_at_the_cap_within_one_table() -> None:
    # One table pair joined by more parallel columns than the cap allows.
    adjacency = {
        "a": [("b", f"a.c{i:03}", f"b.c{i:03}", f"k{i:03}") for i in range(201)],
        "b": [],
    }
    routes = enumerate_routes(adjacency, "a", "b")
    assert len(routes) == MAX_ENUMERATED_ROUTES


def test_parallel_joins_tie_break_by_column_name_not_id() -> None:
    adjacency = {
        "a": [("b", "id-2", "id-9", "k1"), ("b", "id-1", "id-8", "k2")],
        "b": [],
    }
    names = {"id-1": "z_ref", "id-8": "z_id", "id-2": "a_ref", "id-9": "a_id"}
    ranked = rank_routes(adjacency, "a", "b", k=2, column_names=names)
    assert [route.hops[0][0] for route in ranked] == ["id-2", "id-1"]


def test_schema_qualified_names_break_ties_between_same_named_tables() -> None:
    # Two `customer` tables in different schemas; the qualified name decides.
    edges = [("t1", "a_id", "a", "id"), ("t1", "b_id", "b", "id")]
    edges += [("t2", "a_id", "a", "id"), ("t2", "b_id", "b", "id")]
    names = {"t1": "sales.customer", "t2": "archive.customer"}
    ranked = rank_routes(build(edges), "a", "b", k=2, table_names=names)
    assert [route.tables[1] for route in ranked] == ["t2", "t1"]


def test_the_cap_keeps_a_shorter_route_whatever_the_ids() -> None:
    # 200 three-hop routes a -> x -> u_i -> b, and one two-hop route
    # a -> z -> b. `x` sorts before `z`, so an id-ordered walk would fill the
    # cap through `x` and never reach `z`.
    adjacency: dict[str, list[tuple[str, str, str, str]]] = {
        "a": [("x", "a.x", "x.a", "k"), ("z", "a.z", "z.a", "k")],
        "x": [(f"u{i:03}", f"x.u{i}", f"u{i}.x", "k") for i in range(200)],
        "z": [("b", "z.b", "b.z", "k")],
        "b": [],
    }
    for i in range(200):
        adjacency[f"u{i:03}"] = [("b", f"u{i}.b", f"b.u{i}", "k")]

    routes = enumerate_routes(adjacency, "a", "b", max_extra_hops=1)
    assert len(routes) == MAX_ENUMERATED_ROUTES
    assert routes[0][0] == ("a", "z", "b")

    (best,) = rank_routes(adjacency, "a", "b", k=1)
    assert best.tables == ("a", "z", "b")


def test_within_one_length_the_cap_keeps_boosted_routes_first() -> None:
    # 250 two-hop routes a -> m_i -> b; only m_249 is boosted and it sorts last.
    adjacency: dict[str, list[tuple[str, str, str, str]]] = {
        "a": [(f"m{i:03}", f"a.m{i}", f"m{i}.a", "k") for i in range(250)],
        "b": [],
    }
    for i in range(250):
        adjacency[f"m{i:03}"] = [("b", f"m{i}.b", f"b.m{i}", "k")]

    (best,) = rank_routes(adjacency, "a", "b", k=1, boost_tables={"m249"})
    assert best.tables == ("a", "m249", "b")
