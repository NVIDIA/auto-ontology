# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rebuild the AGE graph from the catalog and semantic tables.

A sync is a full rebuild inside **one transaction**: drop the graph, create it,
create every label, index vertices on their properties, load vertices, then
load edges. Readers on other connections keep seeing the previous graph until
the commit, and a failure part-way leaves the previous graph in place. There
is no incremental mode: the source of truth is Postgres, the projection is
derived, and a rebuild cannot drift from it the way a stream of deltas can.

The vertex index is what makes edge loading tractable. Each edge row is placed
with ``MATCH (a:L {id: ...})``, which AGE answers with a ``@>`` containment
test on the label table's ``properties`` column; a GIN index on that column
turns it from a scan per edge into a lookup (measured on 20k vertices: 200
edges in 5.1s without it, 0.02s with it).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import Connection, and_, select
from sqlalchemy.engine import Engine

from auto_ontology.graph import age
from auto_ontology.graph.mapping import EDGES, NODES, EdgeSpec, NodeSpec

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 1000


@dataclass
class SyncReport:
    """What a sync wrote, per label, for logs and the CLI."""

    graph: str
    vertices: dict[str, int] = field(default_factory=dict)
    edges: dict[str, int] = field(default_factory=dict)
    #: Edge rows whose source or target vertex was not found. Non-zero means
    #: the mapping and the data disagree, which is worth a warning.
    unmatched_edges: int = 0
    seconds: float = 0.0
    synced_at: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "graph": self.graph,
            "synced_at": self.synced_at,
            "vertices": dict(self.vertices),
            "edges": dict(self.edges),
            "unmatched_edges": self.unmatched_edges,
            "seconds": round(self.seconds, 3),
        }


def _batches(
    rows: Sequence[dict[str, Any]], size: int
) -> Iterator[list[dict[str, Any]]]:
    for start in range(0, len(rows), size):
        yield list(rows[start : start + size])


def _strip_nulls(row: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    """Drop NULLs: an absent property and a null one read the same in Cypher."""
    return {k: age.to_agtype_value(v) for k, v in row if v is not None}


def vertex_rows(conn: Connection, spec: NodeSpec) -> list[dict[str, Any]]:
    cols = [spec.table.c.id, *(spec.table.c[name] for name in spec.properties)]
    result = conn.execute(select(*cols).order_by(spec.table.c.id))
    return [_strip_nulls(row._mapping.items()) for row in result]


def edge_rows(conn: Connection, spec: EdgeSpec) -> list[dict[str, Any]]:
    src = spec.table.c[spec.source_column]
    dst = spec.table.c[spec.target_column]
    props = [spec.table.c[name] for name in spec.properties]
    query = (
        select(src.label("s"), dst.label("t"), *props)
        .where(and_(src.is_not(None), dst.is_not(None)))
        .order_by(src, dst)
    )
    rows = []
    for row in conn.execute(query):
        mapping = dict(row._mapping)
        rows.append(
            {
                "s": mapping.pop("s"),
                "t": mapping.pop("t"),
                "p": _strip_nulls(mapping.items()),
            }
        )
    return rows


def create_vertices_statement(label: str) -> str:
    age.identifier(label)
    return f"UNWIND $rows AS r CREATE (n:{label}) SET n = r RETURN count(n)"


def create_edges_statement(spec: EdgeSpec) -> str:
    for name in (spec.label, spec.source_label, spec.target_label):
        age.identifier(name)
    return (
        "UNWIND $rows AS r "
        f"MATCH (a:{spec.source_label} {{id: r.s}}), (b:{spec.target_label} {{id: r.t}}) "
        # AGE's SET takes a map variable, not a property expression, so the
        # edge properties are bound with WITH first.
        "WITH a, b, r.p AS p "
        f"CREATE (a)-[e:{spec.label}]->(b) SET e = p "
        "RETURN count(e)"
    )


def _create_schema(conn: Connection, graph: str) -> None:
    if age.graph_exists(conn, graph):
        conn.exec_driver_sql("SELECT drop_graph(%s, true)", (graph,))
    conn.exec_driver_sql("SELECT create_graph(%s)", (graph,))
    for spec in NODES:
        conn.exec_driver_sql("SELECT create_vlabel(%s, %s)", (graph, spec.label))
        conn.exec_driver_sql(
            f'CREATE INDEX ON "{graph}"."{age.identifier(spec.label)}" USING gin (properties)'
        )
    for label in dict.fromkeys(spec.label for spec in EDGES):
        conn.exec_driver_sql("SELECT create_elabel(%s, %s)", (graph, label))
        # Neighbourhood reads expand from a vertex to its edges; without
        # these, every hop scans the whole edge table.
        table = f'"{graph}"."{age.identifier(label)}"'
        conn.exec_driver_sql(f"CREATE INDEX ON {table} (start_id)")
        conn.exec_driver_sql(f"CREATE INDEX ON {table} (end_id)")


def _load(
    conn: Connection,
    graph: str,
    statement: str,
    rows: list[dict[str, Any]],
    batch_size: int,
) -> int:
    written = 0
    for batch in _batches(rows, batch_size):
        ((count,),) = age.cypher(conn, graph, statement, {"rows": batch})
        written += int(count)
    return written


def sync_graph(
    engine: Engine | None = None,
    *,
    graph: str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> SyncReport:
    """Rebuild the graph from Postgres in one transaction and report counts."""
    if engine is None:
        from auto_ontology.dal.session import get_engine

        engine = get_engine()
    graph = graph or age.graph_name()
    report = SyncReport(graph=graph)
    started = time.monotonic()

    with engine.begin() as conn:
        age.ensure_extension(conn)
        age.prepare(conn)
        _create_schema(conn, graph)
        for spec in NODES:
            rows = vertex_rows(conn, spec)
            report.vertices[spec.label] = _load(
                conn, graph, create_vertices_statement(spec.label), rows, batch_size
            )
        for vertex_label in report.vertices:
            # Fresh statistics so the planner uses the GIN index for the
            # edge MATCHes below instead of guessing from an empty table.
            conn.exec_driver_sql(f'ANALYZE "{graph}"."{vertex_label}"')
        for spec in EDGES:
            rows = edge_rows(conn, spec)
            written = _load(conn, graph, create_edges_statement(spec), rows, batch_size)
            report.edges[spec.label] = report.edges.get(spec.label, 0) + written
            report.unmatched_edges += len(rows) - written
        # Recorded on the graph's own schema so it commits, or rolls back,
        # together with the graph it describes.
        # COMMENT takes only a literal; an ISO timestamp contains no quote.
        synced_at = datetime.now(timezone.utc).isoformat()
        conn.exec_driver_sql(f"COMMENT ON SCHEMA \"{graph}\" IS '{synced_at}'")
        report.synced_at = synced_at

    report.seconds = time.monotonic() - started
    if report.unmatched_edges:
        logger.warning(
            "AGE sync: %d edge rows had no matching vertex", report.unmatched_edges
        )
    logger.info("AGE sync finished: %s", report.as_dict())
    return report


def sync_if_enabled() -> SyncReport | None:
    """Rebuild the graph when ``AGE_GRAPH_ENABLED`` is set; never raise.

    Called at the end of a semantic compilation pass. The graph is derived, so
    a failed sync is logged and the pass still counts as whatever it was:
    turning a projection error into a failed compilation would hide the
    compilation's real outcome behind an optional feature.
    """
    if not age.enabled():
        return None
    try:
        return sync_graph()
    except Exception:  # noqa: BLE001 — logged, see docstring
        logger.exception("AGE sync after semantic compilation failed")
        return None
