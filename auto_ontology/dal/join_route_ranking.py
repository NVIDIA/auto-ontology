# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rank alternative join routes between two tables, PathRAG-style.

Pure computation over the default join edges that
:func:`auto_ontology.dal.attributes._semantic_table_adjacency` builds; nothing
here touches the database, so the scoring is unit-testable on a hand-written
graph.

**Why more than one route.** Two tables are often joinable by more than one
equally short route (``address -> customer -> rental`` and ``address -> staff
-> rental``), and the routes mean different things. The single-route BFS keeps
whichever it reaches first, which follows the order of table UUIDs. Returning
a ranked handful lets the SQL step choose by meaning.

**Scoring** follows PathRAG (Chen et al., 2025, arXiv:2502.14902), sections
3.2 and 3.3:

* A unit of resource starts at the anchor table and flows outward one layer at
  a time. Each table passes ``alpha * S(u) / |N(u)|`` to every neighbour, so
  resource decays with distance and is diluted by fan-out: a table reached
  only through a hub that joins to everything gets a small share, and a route
  that continues through it scores lower. (Two routes whose middle tables are
  both direct neighbours of the anchor still tie, hub or not; the dilution
  applies to what flows past the hub.) A table that still has resource but would pass on less than
  ``theta`` per neighbour stops expanding. Each table is set once, in the
  first layer that reaches it.
* A route's **reliability** is the resource summed over its tables, divided by
  its number of hops.

PathRAG then multiplies by how well a path's start matches the query. The
join-route counterpart used here is a **question boost**: every intermediate
table that holds one of the question's retrieved attributes multiplies the
score by ``1 + boost``. Two routes that are structurally identical (same
length, same degrees) have the same reliability, and the boost is what tells
them apart.

**Bounds.** Routes are simple (no repeated table), use the same directed
adjacency as the single-route BFS (so every hop is a join that BFS would also
accept), and are at most ``max_extra_hops`` longer than the shortest route and
never longer than ``max_hops``. Enumeration prunes with the remaining distance
to the destination, so it only walks routes that can still finish within
bounds.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

#: ``adjacency[table_id]`` lists ``(next_table_id, source_col_id, target_col_id,
#: key_id)``.
Adjacency = Mapping[str, Sequence[tuple[str, str, str, str]]]
#: ``(source_col_id, target_col_id, key_id)``.
Hop = tuple[str, str, str]

DEFAULT_ALPHA = 0.7
DEFAULT_THETA = 0.005
DEFAULT_BOOST = 0.5
#: Hard ceiling on routes enumerated per pair, so a dense schema cannot turn
#: one prompt into a combinatorial walk.
MAX_ENUMERATED_ROUTES = 200


@dataclass(frozen=True)
class RankedRoute:
    """One candidate route, anchor table first."""

    #: Table ids along the route, anchor first and destination last.
    tables: tuple[str, ...]
    #: ``(source_col_id, target_col_id, key_id)`` per hop, in order.
    hops: tuple[Hop, ...]
    reliability: float
    score: float


def flow_resources(
    adjacency: Adjacency,
    source: str,
    *,
    alpha: float = DEFAULT_ALPHA,
    theta: float = DEFAULT_THETA,
) -> dict[str, float]:
    """Resource reaching each table from *source* (PathRAG eq. 2 and 3)."""
    resources = {source: 1.0}
    frontier = [source]
    while frontier:
        incoming: dict[str, float] = defaultdict(float)
        for table in frontier:
            neighbours = {edge[0] for edge in adjacency.get(table, ())}
            if not neighbours:
                continue
            share = alpha * resources[table] / len(neighbours)
            if resources[table] / len(neighbours) < theta:
                continue
            for neighbour in neighbours:
                if neighbour not in resources:
                    incoming[neighbour] += share
        resources.update(incoming)
        frontier = sorted(incoming)
    return resources


def _distances_to(adjacency: Adjacency, dest: str) -> dict[str, int]:
    """Hops from every table to *dest*, walking edges backwards."""
    reverse: dict[str, set[str]] = defaultdict(set)
    for table, edges in adjacency.items():
        for next_table, *_ in edges:
            reverse[next_table].add(table)
    distance = {dest: 0}
    queue = deque([dest])
    while queue:
        table = queue.popleft()
        for previous in reverse.get(table, ()):
            if previous not in distance:
                distance[previous] = distance[table] + 1
                queue.append(previous)
    return distance


def enumerate_routes(
    adjacency: Adjacency,
    anchor: str,
    dest: str,
    *,
    max_extra_hops: int = 1,
    max_hops: int = 8,
) -> list[tuple[tuple[str, ...], tuple[Hop, ...]]]:
    """Simple routes from *anchor* to *dest* within the length bounds."""
    if anchor == dest:
        return []
    distance = _distances_to(adjacency, dest)
    if anchor not in distance:
        return []
    limit = min(distance[anchor] + max_extra_hops, max_hops)

    routes: list[tuple[tuple[str, ...], tuple[Hop, ...]]] = []
    stack: list[tuple[tuple[str, ...], tuple[Hop, ...]]] = [((anchor,), ())]
    while stack and len(routes) < MAX_ENUMERATED_ROUTES:
        tables, hops = stack.pop()
        here = tables[-1]
        for next_table, source_col, target_col, key_id in sorted(
            adjacency.get(here, ()), reverse=True
        ):
            if next_table in tables:
                continue
            remaining = distance.get(next_table)
            if remaining is None or len(hops) + 1 + remaining > limit:
                continue
            route = (tables + (next_table,), hops + ((source_col, target_col, key_id),))
            if next_table == dest:
                routes.append(route)
            else:
                stack.append(route)
    return routes


def rank_routes(
    adjacency: Adjacency,
    anchor: str,
    dest: str,
    *,
    k: int,
    boost_tables: Collection[str] = (),
    table_names: Mapping[str, str] | None = None,
    alpha: float = DEFAULT_ALPHA,
    theta: float = DEFAULT_THETA,
    boost: float = DEFAULT_BOOST,
    max_extra_hops: int = 1,
    max_hops: int = 8,
) -> list[RankedRoute]:
    """The *k* best routes, best first.

    Ties are broken by hop count, then by the table names along the route
    (falling back to ids), so the order never depends on how ids were
    generated.
    """
    if k < 1:
        return []
    resources = flow_resources(adjacency, anchor, alpha=alpha, theta=theta)
    names = table_names or {}
    boosted = set(boost_tables)
    ranked: list[RankedRoute] = []
    for tables, hops in enumerate_routes(
        adjacency, anchor, dest, max_extra_hops=max_extra_hops, max_hops=max_hops
    ):
        reliability = sum(resources.get(t, 0.0) for t in tables) / len(hops)
        factor = 1.0 + boost * sum(1 for t in tables[1:-1] if t in boosted)
        ranked.append(
            RankedRoute(
                tables, hops, round(reliability, 6), round(reliability * factor, 6)
            )
        )
    ranked.sort(
        key=lambda r: (
            -r.score,
            len(r.hops),
            tuple(names.get(t, t) for t in r.tables),
            r.hops,
        )
    )
    return ranked[:k]
