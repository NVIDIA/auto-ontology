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
  applies to what flows past the hub.) A table stops expanding when
  ``S(u) / |N(u)|`` is below ``theta``. That check runs **before** the
  ``alpha`` decay, as in PathRAG equation 3, so tune ``theta`` against the
  undecayed share: with 150 neighbours, ``1/150`` passes the default
  ``theta`` even though each neighbour receives only ``0.7/150``. Each table
  is set once, in the first layer that reaches it.
* A route's **reliability** is the resource summed over its tables, divided by
  its number of hops.

PathRAG then multiplies by how well a path's start matches the query. The
join-route counterpart used here is a **question boost**: every intermediate
table that holds one of the question's retrieved attributes multiplies the
score by ``1 + boost``, so ``n`` such tables multiply it by
``(1 + boost) ** n``. Two routes that are structurally identical (same
length, same degrees) have the same reliability, and the boost is what tells
them apart.

**Bounds.** Routes are simple (no repeated table), use the same directed
adjacency as the single-route BFS (so every hop is a join that BFS would also
accept), and are at most ``max_extra_hops`` longer than the shortest route and
never longer than ``max_hops``. Enumeration prunes with the remaining distance
to the destination, so it only walks routes that can still finish within
bounds. It collects routes shortest first and, within a length, follows the
edges most likely to score well first (boosted tables, then more resource,
then names), so the :data:`MAX_ENUMERATED_ROUTES` cap drops the least
promising candidates. Within one length this is a heuristic order, not an
exact best-first search.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

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
    """Resource reaching each table from *source* (PathRAG eq. 2 and 3).

    A table expands only when ``S(u) / |N(u)|`` reaches *theta*; the share it
    then passes on is ``alpha`` times that.
    """
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


#: Orders the edges leaving one table during enumeration; smaller is visited
#: first. Receives ``(next_table_id, source_col_id, target_col_id, key_id)``.
EdgePriority = Callable[[tuple[str, str, str, str]], Any]


def enumerate_routes(
    adjacency: Adjacency,
    anchor: str,
    dest: str,
    *,
    max_extra_hops: int = 1,
    max_hops: int = 8,
    edge_priority: EdgePriority | None = None,
) -> list[tuple[tuple[str, ...], tuple[Hop, ...]]]:
    """Simple routes from *anchor* to *dest* within the length bounds.

    Routes are collected shortest first: every route of the shortest length,
    then the next length, until :data:`MAX_ENUMERATED_ROUTES` is reached. So
    when the cap cuts, it cuts longer routes, never a shorter one. Within one
    length the walk is depth-first and visits a table's edges in
    *edge_priority* order (by edge, ids included, when not given), so the
    routes the cap keeps are the most promising ones rather than whichever
    ids sort first.
    """
    if anchor == dest:
        return []
    distance = _distances_to(adjacency, dest)
    if anchor not in distance:
        return []
    limit = min(distance[anchor] + max_extra_hops, max_hops)
    order: EdgePriority = edge_priority or (lambda edge: edge)

    routes: list[tuple[tuple[str, ...], tuple[Hop, ...]]] = []
    for length in range(distance[anchor], limit + 1):
        stack: list[tuple[tuple[str, ...], tuple[Hop, ...]]] = [((anchor,), ())]
        while stack:
            tables, hops = stack.pop()
            depth = len(hops) + 1
            candidates = []
            for edge in adjacency.get(tables[-1], ()):
                next_table = edge[0]
                remaining = distance.get(next_table)
                if next_table in tables or remaining is None:
                    continue
                if depth + remaining > length:
                    continue
                if next_table == dest and depth != length:
                    continue
                candidates.append(edge)
            candidates.sort(key=order)
            deeper = []
            for next_table, source_col, target_col, key_id in candidates:
                route = (
                    tables + (next_table,),
                    hops + ((source_col, target_col, key_id),),
                )
                if next_table == dest:
                    routes.append(route)
                    if len(routes) >= MAX_ENUMERATED_ROUTES:
                        return routes
                else:
                    deeper.append(route)
            # The stack pops from the end, so the best candidate goes on last.
            stack.extend(reversed(deeper))
    return routes


def rank_routes(
    adjacency: Adjacency,
    anchor: str,
    dest: str,
    *,
    k: int,
    boost_tables: Collection[str] = (),
    table_names: Mapping[str, str] | None = None,
    column_names: Mapping[str, str] | None = None,
    alpha: float = DEFAULT_ALPHA,
    theta: float = DEFAULT_THETA,
    boost: float = DEFAULT_BOOST,
    max_extra_hops: int = 1,
    max_hops: int = 8,
) -> list[RankedRoute]:
    """The *k* best routes, best first.

    Ties are broken by hop count, then by the table names along the route,
    then by the column names of each hop, so the order never depends on how
    ids were generated as long as *table_names* (schema-qualified, so
    same-named tables in two schemas differ) and *column_names* cover the
    route. Ids are the last resort.
    """
    if k < 1:
        return []
    resources = flow_resources(adjacency, anchor, alpha=alpha, theta=theta)
    names = table_names or {}
    columns = column_names or {}
    boosted = set(boost_tables)
    ranked: list[RankedRoute] = []

    def edge_priority(edge: tuple[str, str, str, str]) -> tuple[Any, ...]:
        next_table, source_col, target_col, _ = edge
        return (
            next_table not in boosted,
            -resources.get(next_table, 0.0),
            names.get(next_table, next_table),
            columns.get(source_col, source_col),
            columns.get(target_col, target_col),
            edge,
        )

    for tables, hops in enumerate_routes(
        adjacency,
        anchor,
        dest,
        max_extra_hops=max_extra_hops,
        max_hops=max_hops,
        edge_priority=edge_priority,
    ):
        reliability = sum(resources.get(t, 0.0) for t in tables) / len(hops)
        factor = (1.0 + boost) ** sum(1 for t in tables[1:-1] if t in boosted)
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
            tuple(
                (columns.get(src, src), columns.get(dst, dst)) for src, dst, _ in r.hops
            ),
            r.hops,
        )
    )
    return ranked[:k]
