# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read paths over the AGE graph.

Everything here runs in :func:`auto_ontology.graph.age.read_transaction`: read
only, and rolled back. Vertices and edges are reported by their **Postgres
ids** (the ``id`` property every vertex carries), never by AGE's internal
graph ids, so a caller can take any id straight to the REST API.

The neighbourhood walk is a breadth-first expansion, one Cypher statement per
hop, rather than a single variable-length ``MATCH``. A variable-length pattern
enumerates *paths*, and around a Term that represents a wide table the number
of paths grows with the product of the fan-outs at each hop; expanding the
frontier once per hop costs the number of edges actually touched, and the
vertex cap stops it before it grows past what a caller can read.
"""

from __future__ import annotations

from collections import defaultdict
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
    with age.read_transaction(_engine(engine)) as conn:
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


_EXPAND = (
    "UNWIND $ids AS i "
    "MATCH (a:{label} {{id: i}})-[e]-(b) "
    "RETURN a.id, label(e), start_id(e) = id(a), properties(e), label(b), properties(b)"
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
    partial rather than complete.
    """
    if not 1 <= depth <= MAX_DEPTH:
        raise ValueError(f"depth must be between 1 and {MAX_DEPTH}")
    if not 1 <= max_nodes <= MAX_NODES:
        raise ValueError(f"max_nodes must be between 1 and {MAX_NODES}")
    graph = graph or age.graph_name()

    with age.read_transaction(_engine(engine)) as conn:
        if not age.graph_exists(conn, graph):
            raise age.GraphNotBuilt(
                f"graph {graph!r} has not been built; run "
                "`python -m auto_ontology.graph sync`"
            )
        start = age.cypher(
            conn,
            graph,
            f"MATCH (t:{LABEL_TERM} {{id: $id}}) RETURN properties(t)",
            {"id": term_id},
            columns=("props",),
        )
        if not start:
            return None

        nodes: dict[str, dict[str, Any]] = {term_id: _vertex(LABEL_TERM, start[0][0])}
        edges: dict[tuple[str, str, str], dict[str, Any]] = {}
        frontier: dict[str, list[str]] = {LABEL_TERM: [term_id]}
        truncated = False

        for _ in range(depth):
            next_frontier: dict[str, list[str]] = defaultdict(list)
            for label, ids in frontier.items():
                rows = age.cypher(
                    conn,
                    graph,
                    _EXPAND.format(label=age.identifier(label)),
                    {"ids": ids},
                    columns=("a", "rel", "outgoing", "eprops", "blabel", "bprops"),
                )
                for a_id, rel, outgoing, eprops, b_label, b_props in rows:
                    b_id = b_props["id"]
                    if b_id not in nodes:
                        if len(nodes) >= max_nodes:
                            truncated = True
                            continue
                        nodes[b_id] = _vertex(b_label, b_props)
                        next_frontier[b_label].append(b_id)
                    source, target = (a_id, b_id) if outgoing else (b_id, a_id)
                    edges.setdefault(
                        (rel, source, target),
                        {
                            "type": rel,
                            "source": source,
                            "target": target,
                            "properties": eprops,
                        },
                    )
            frontier = next_frontier
            if not frontier:
                break

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
    with age.read_transaction(_engine(engine)) as conn:
        if not age.graph_exists(conn, graph):
            raise age.GraphNotBuilt(f"graph {graph!r} has not been built")
        return age.cypher(conn, graph, query, params, columns)
