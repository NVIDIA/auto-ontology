# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read paths over the AGE graph.

Everything here runs in :func:`auto_ontology.graph.age.read_transaction`: read
only, and rolled back. Vertices and edges are reported by their **Postgres
ids** (the ``id`` property every vertex carries), never by AGE's internal
graph ids, so a caller can take any id straight to the REST API.

The neighbourhood walk is a breadth-first expansion, one hop at a time,
rather than a single variable-length ``MATCH``: a variable-length pattern
enumerates *paths*, which grow with the product of the fan-outs at each hop.

Each hop is plain SQL over AGE's own storage rather than Cypher. AGE plans an
undirected ``MATCH (a)-[]-(b)`` with ``b`` unbound as a scan of every vertex
table joined to the edges, so its cost follows the size of the graph. The
edge tables carry ``start_id``/``end_id`` indexes (created by the sync), and
reading ``_ag_label_edge`` through them costs the index entries of the
frontier's own edges. A hop reads those ids only, keeps at most the vertices
the cap still has room for (plus one, to know whether it cut anything), and
fetches labels and properties for the kept vertices alone.
"""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy.engine import Engine

from auto_ontology.graph import age
from auto_ontology.graph.mapping import edge_labels, vertex_labels
from auto_ontology.semantic.constants import LABEL_TERM

MAX_DEPTH = 3
DEFAULT_MAX_NODES = 200
MAX_NODES = 1000


def _engine(engine: Engine | None) -> Engine:
    if engine is not None:
        return engine
    from auto_ontology.dal.session import get_engine

    return get_engine()


def _vertex(label: str, props: dict[str, Any]) -> dict[str, Any]:
    props = dict(props)
    return {"id": props.pop("id"), "label": label, "properties": props}


def graph_status(engine: Engine | None = None, *, graph: str | None = None) -> dict:
    """Whether the graph exists, and how many vertices and edges each label has."""
    graph = graph or age.graph_name()
    with age.read_transaction(_engine(engine), graph) as conn:
        if not age.graph_exists(conn, graph):
            return {
                "graph": graph,
                "built": False,
                "synced_at": None,
                "vertices": {},
                "edges": {},
            }
        (synced_at,) = conn.exec_driver_sql(
            "SELECT obj_description(%s::regnamespace, 'pg_namespace')", (graph,)
        ).one()
        labels = conn.exec_driver_sql(
            "SELECT l.name, l.kind FROM ag_catalog.ag_label l "
            "JOIN ag_catalog.ag_graph g ON g.graphid = l.graph "
            "WHERE g.name = %s AND l.name NOT LIKE %s",
            (graph, "\\_ag\\_label\\_%"),
        ).all()
        counts: dict[str, dict[str, int]] = {"v": {}, "e": {}}
        for name, kind in labels:
            (count,) = conn.exec_driver_sql(
                f'SELECT count(*) FROM "{graph}"."{age.identifier(name)}"'
            ).one()
            counts[kind][name] = int(count)
    order_v = {label: i for i, label in enumerate(vertex_labels())}
    order_e = {label: i for i, label in enumerate(edge_labels())}
    return {
        "graph": graph,
        "built": True,
        "synced_at": synced_at,
        "vertices": dict(
            sorted(counts["v"].items(), key=lambda kv: order_v.get(kv[0], 99))
        ),
        "edges": dict(
            sorted(counts["e"].items(), key=lambda kv: order_e.get(kv[0], 99))
        ),
    }


_NEIGHBOUR_IDS = """
SELECT n.other::text FROM (
    SELECT end_id AS other FROM {g}._ag_label_edge
     WHERE start_id = ANY(%(frontier)s::text[]::graphid[])
    UNION
    SELECT start_id FROM {g}._ag_label_edge
     WHERE end_id = ANY(%(frontier)s::text[]::graphid[])
) n
WHERE n.other <> ALL(%(known)s::text[]::graphid[])
ORDER BY n.other
LIMIT %(limit)s
"""

_VERTICES = """
SELECT v.id::text, l.name, v.properties::text
  FROM {g}._ag_label_vertex v
  JOIN ag_catalog.ag_label l ON l.relation = v.tableoid
 WHERE v.id = ANY(%(ids)s::text[]::graphid[])
"""

_KEPT_EDGES = """
SELECT e.start_id::text, e.end_id::text, l.name, e.properties::text
  FROM {g}._ag_label_edge e
  JOIN ag_catalog.ag_label l ON l.relation = e.tableoid
 WHERE (e.start_id = ANY(%(frontier)s::text[]::graphid[])
        AND e.end_id = ANY(%(kept)s::text[]::graphid[]))
    OR (e.end_id = ANY(%(frontier)s::text[]::graphid[])
        AND e.start_id = ANY(%(kept)s::text[]::graphid[]))
"""


def _collect_kept_edges(
    conn: Any,
    g: str,
    frontier: list[str],
    nodes: dict[str, dict[str, Any]],
    edges: dict[tuple[str, str, str], dict[str, Any]],
) -> None:
    """Add the edges between *frontier* and any kept vertex, without expanding."""
    if not frontier:
        return
    rows = conn.exec_driver_sql(
        _KEPT_EDGES.format(g=g), {"frontier": frontier, "kept": list(nodes)}
    )
    for start, end, rel, props in rows:
        source, target = nodes[start], nodes[end]
        edges.setdefault(
            (rel, start, end),
            {
                "type": rel,
                "source": source["id"],
                "source_label": source["label"],
                "target": target["id"],
                "target_label": target["label"],
                "properties": age.parse_agtype(props) or {},
            },
        )


def term_neighborhood(
    term_id: str,
    *,
    depth: int = 1,
    max_nodes: int = DEFAULT_MAX_NODES,
    engine: Engine | None = None,
    graph: str | None = None,
) -> dict[str, Any] | None:
    """Vertices and edges within *depth* hops of one Term.

    Returns ``None`` when the Term is not in the graph. ``truncated`` is set
    when *max_nodes* stopped the walk early, so a caller knows the picture is
    partial rather than complete. Every edge between two returned vertices is
    included.

    A vertex is identified by its id *and* label (a Concept shares its id with
    the Term it was created from), so each edge names both labels too.
    """
    if not 1 <= depth <= MAX_DEPTH:
        raise ValueError(f"depth must be between 1 and {MAX_DEPTH}")
    if not 1 <= max_nodes <= MAX_NODES:
        raise ValueError(f"max_nodes must be between 1 and {MAX_NODES}")
    graph = graph or age.graph_name()
    g = age.quoted_graph(graph)

    with age.read_transaction(_engine(engine), graph) as conn:
        if not age.graph_exists(conn, graph):
            raise age.GraphNotBuilt(
                f"graph {graph!r} has not been built; run "
                "`python -m auto_ontology.graph sync`"
            )
        start = conn.exec_driver_sql(
            f'SELECT v.id::text, v.properties::text FROM {g}."{LABEL_TERM}" v '
            "WHERE v.properties @> %(match)s::agtype",
            {"match": json.dumps({"id": term_id})},
        ).first()
        if start is None:
            return None

        start_gid, start_props = start
        # Keyed by AGE graph id, which is unique where the Postgres id is not.
        nodes: dict[str, dict[str, Any]] = {
            start_gid: _vertex(LABEL_TERM, age.parse_agtype(start_props))
        }
        edges: dict[tuple[str, str, str], dict[str, Any]] = {}
        frontier = [start_gid]
        truncated = False

        for _ in range(depth):
            room = max_nodes - len(nodes)
            found = [
                gid
                for (gid,) in conn.exec_driver_sql(
                    _NEIGHBOUR_IDS.format(g=g),
                    {
                        "frontier": frontier,
                        "known": list(nodes),
                        "limit": room + 1,
                    },
                )
            ]
            if len(found) > room:
                truncated = True
            kept = found[:room]
            if kept:
                for gid, label, props in conn.exec_driver_sql(
                    _VERTICES.format(g=g), {"ids": kept}
                ):
                    nodes[gid] = _vertex(label, age.parse_agtype(props))
            # Edges are read once the hop's vertices are settled, so every edge
            # between two kept vertices is reported and none to a dropped one.
            _collect_kept_edges(conn, g, frontier, nodes, edges)
            frontier = kept
            if not frontier:
                break
        # The last hop's vertices are not expanded, but edges among kept
        # vertices that start there (a Tag on both a Term and its Table)
        # still belong to the picture.
        _collect_kept_edges(conn, g, frontier, nodes, edges)

    return {
        "term_id": term_id,
        "depth": depth,
        "truncated": truncated,
        "nodes": list(nodes.values()),
        "edges": list(edges.values()),
    }


def run_cypher(
    query: str,
    params: dict[str, Any] | None = None,
    *,
    columns: tuple[str, ...] = ("v",),
    engine: Engine | None = None,
    graph: str | None = None,
) -> list[tuple[Any, ...]]:
    """Run an arbitrary Cypher read for operators (the CLI), never writes."""
    graph = graph or age.graph_name()
    with age.read_transaction(_engine(engine), graph) as conn:
        if not age.graph_exists(conn, graph):
            raise age.GraphNotBuilt(f"graph {graph!r} has not been built")
        return age.cypher(conn, graph, query, params, columns)
