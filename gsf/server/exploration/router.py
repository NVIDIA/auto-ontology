# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""API route handlers for data- and semantic-layer Exploration graphs."""

from __future__ import annotations

from fastapi import APIRouter, Query

from gsf.server.exploration import service

router = APIRouter()

_LIMIT_QUERY = Query(
    default=service.MAX_EXPLORATION_GRAPH_NODES,
    ge=1,
    description=(
        f"Max nodes returned; always capped at {service.MAX_EXPLORATION_GRAPH_NODES} "
        "server-side."
    ),
)


@router.get("/exploration/edges")
def list_data_exploration_edges(
    zone_ids: list[str] | None = Query(default=None),
) -> dict:
    """Table connections backed by a shared SQL query or a foreign key, scoped to visible tables."""
    rows = service.fetch_data_exploration_edges(zone_ids=zone_ids)
    return {"data": rows, "count": len(rows)}


@router.get("/exploration/graph")
def get_data_exploration_graph(
    zone_ids: list[str] | None = Query(default=None),
    limit: int = _LIMIT_QUERY,
) -> dict:
    """Return the full data-layer Exploration graph (``{nodes, links}``).

    Lets the client render the data graph from one request instead of
    walking the catalog tree (databases → schemas → tables) with a request
    per level. Zone-scoped when zone_ids are provided. Node count is capped
    at ``MAX_EXPLORATION_GRAPH_NODES`` regardless of *limit*.
    """
    return {
        "data": service.fetch_data_exploration_graph(zone_ids=zone_ids, limit=limit)
    }


@router.get("/exploration/tables/{table_id}/details")
def get_table_exploration_details(
    table_id: str,
    zone_ids: list[str] | None = Query(default=None),
) -> dict:
    """Columns-adjacent SQL and Term details for Exploration modals."""
    return {
        "data": service.fetch_table_exploration_details(table_id, zone_ids=zone_ids)
    }


@router.get("/exploration/tables/zones")
def list_table_zones(zone_ids: list[str] | None = Query(default=None)) -> dict:
    """Return ``{table_id: [zone, ...]}`` for every visible Table.

    Used by the Exploration graph to render Zone chips on every data node
    without a per-node request.
    """
    return {"data": service.fetch_table_zones_map(zone_ids=zone_ids)}


@router.get("/exploration/semantic-graph")
def get_semantic_exploration_graph(
    zone_ids: list[str] | None = Query(default=None),
    limit: int = _LIMIT_QUERY,
) -> dict:
    """Return the full semantic-layer Exploration graph (``{nodes, links}``).

    Lets the client render the semantic graph from one request instead of
    fetching related terms once per term (an N+1). Zone-scoped when zone_ids
    are provided. Node count is capped at ``MAX_EXPLORATION_GRAPH_NODES``
    regardless of *limit*.
    """
    return {
        "data": service.fetch_semantic_exploration_graph(
            zone_ids=zone_ids,
            limit=limit,
        )
    }
