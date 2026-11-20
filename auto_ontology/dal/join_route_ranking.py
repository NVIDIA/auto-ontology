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
never longer than ``max_hops``.

**Search.** :func:`rank_routes` returns the exact top *k* without listing
every route. It searches table sequences best first, ordered by an upper
bound on the score any completion can still reach: the resource collected so
far, plus the destination's, plus the largest resource any unvisited table
holds for each table still to add, divided by the length, times the boost of
every unvisited boosted table that could still fit. A route's score never
exceeds the bound of any route it extends, so once *k* finished routes beat
the best bound left in the queue, no unexplored route can beat them. Parallel
joins between the same two tables share one table sequence and one score; they
are expanded only when that sequence finishes, and only the first *k* by
column name. :data:`MAX_SEARCH_STATES` caps the partial routes the search
expands, so a dense schema cannot turn one prompt into a combinatorial walk;
only a search that hits it can miss a better route.
"""

from __future__ import annotations

import heapq
import itertools
from collections import defaultdict, deque
from collections.abc import Collection, Mapping, Sequence
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
#: Hard ceiling on the partial routes one :func:`rank_routes` call expands.
MAX_SEARCH_STATES = 10_000
#: Scores are reported rounded to 6 places, and routes equal after rounding
#: tie-break by name, so the search keeps going while a bound is this close.
_EPSILON = 1e-6


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
    if k < 1 or anchor == dest:
        return []
    distance = _distances_to(adjacency, dest)
    if anchor not in distance:
        return []
    limit = min(distance[anchor] + max_extra_hops, max_hops)
    resources = flow_resources(adjacency, anchor, alpha=alpha, theta=theta)
    names = table_names or {}
    columns = column_names or {}
    boosted = set(boost_tables)
    gain = 1.0 + boost

    # Parallel joins collapse into one step per neighbouring table, their
    # hops ordered by column name for the tie-break.
    steps: dict[str, dict[str, list[Hop]]] = defaultdict(lambda: defaultdict(list))
    for table, edges in adjacency.items():
        for next_table, source_col, target_col, key_id in edges:
            if next_table in distance:
                steps[table][next_table].append((source_col, target_col, key_id))

    def hop_key(hop: Hop) -> tuple[Any, ...]:
        return (columns.get(hop[0], hop[0]), columns.get(hop[1], hop[1]), hop)

    for neighbours in steps.values():
        for hops in neighbours.values():
            hops.sort(key=hop_key)

    middle = [t for t in distance if t not in (anchor, dest)]
    best_middle = max((resources.get(t, 0.0) for t in middle), default=0.0)
    dest_resource = resources.get(dest, 0.0)

    def bound(tables: tuple[str, ...], collected: float, lifts: int) -> float:
        """Best score any finished extension of *tables* can reach."""
        hops = len(tables) - 1
        spare_boosts = len(boosted.difference(tables, (dest,)))
        best = 0.0
        for length in range(hops + distance[tables[-1]], limit + 1):
            added = length - hops - 1  # tables still to add before dest
            reliability = (collected + dest_resource + added * best_middle) / length
            factor = gain ** (lifts + (min(added, spare_boosts) if gain > 1 else 0))
            best = max(best, reliability * factor)
        return best

    def finished_score(tables: tuple[str, ...], lifts: int) -> tuple[float, float]:
        # ``sum`` is compensated, so a score does not depend on summation order.
        reliability = sum(resources.get(t, 0.0) for t in tables) / (len(tables) - 1)
        return reliability, reliability * gain**lifts

    def order(tables: tuple[str, ...]) -> tuple[Any, ...]:
        return (len(tables), tuple(names.get(t, t) for t in tables), tables)

    # Entries: (-priority, order, finished, tables, collected, lifts).
    queue: list[tuple[Any, ...]] = []
    start = (anchor,)
    first = resources.get(anchor, 0.0)
    heapq.heappush(
        queue, (-bound(start, first, 0), order(start), False, start, first, 0)
    )
    found: list[RankedRoute] = []
    expanded = 0
    while queue:
        neg, _, finished, tables, collected, lifts = heapq.heappop(queue)
        if len(found) >= k and -neg < found[k - 1].score - _EPSILON:
            break
        if finished:
            reliability, score = finished_score(tables, lifts)
            choices = [steps[a][b] for a, b in itertools.pairwise(tables)]
            for hops in itertools.islice(itertools.product(*choices), k):
                found.append(RankedRoute(tables, tuple(hops), reliability, score))
            continue
        expanded += 1
        if expanded > MAX_SEARCH_STATES:
            break
        for next_table in steps[tables[-1]]:
            remaining = distance[next_table]
            if next_table in tables or len(tables) + remaining > limit:
                continue
            route = tables + (next_table,)
            if next_table == dest:
                _, score = finished_score(route, lifts)
                heapq.heappush(queue, (-score, order(route), True, route, 0.0, lifts))
                continue
            gathered = collected + resources.get(next_table, 0.0)
            lifted = lifts + (next_table in boosted)
            heapq.heappush(
                queue,
                (
                    -bound(route, gathered, lifted),
                    order(route),
                    False,
                    route,
                    gathered,
                    lifted,
                ),
            )

    found = [
        RankedRoute(r.tables, r.hops, round(r.reliability, 6), round(r.score, 6))
        for r in found
    ]
    found.sort(
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
    return found[:k]
