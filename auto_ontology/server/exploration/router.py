# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""API route handlers for data- and semantic-layer Exploration graphs."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Path, Query

from auto_ontology.server.exploration import service
from auto_ontology.server.pagination import LIMIT_QUERY, SKIP_QUERY
from auto_ontology.server.responses import (
    ColumnAttributeExplorationDetailsResponse,
    ColumnExplorationDetailsResponse,
    DataExplorationGraphResponse,
    ExplorationEdgeListResponse,
    ExplorationLinkPathResponse,
    ExplorationRelatedNodesResponse,
    SemanticExplorationGraphResponse,
    SqlAttributeExplorationDetailsResponse,
    SqlExplorationDetailsResponse,
    TableExplorationDetailsResponse,
    TableZonesResponse,
    TermExplorationDetailsResponse,
)

router = APIRouter()

_LIMIT_QUERY = Query(
    default=service.MAX_EXPLORATION_GRAPH_NODES,
    ge=1,
    description=(
        f"Max nodes returned; always capped at {service.MAX_EXPLORATION_GRAPH_NODES} "
        "server-side."
    ),
)


@router.get("/exploration/edges", response_model=ExplorationEdgeListResponse)
def list_data_exploration_edges() -> dict:
    """Table connections backed by a shared SQL query or a foreign key."""
    rows = service.fetch_data_exploration_edges(zone_ids=None)
    return {"data": rows, "count": len(rows)}


@router.get("/exploration/graph", response_model=DataExplorationGraphResponse)
def get_data_exploration_graph(limit: int = _LIMIT_QUERY) -> dict:
    """Return the full data-layer Exploration graph (``{nodes, links}``).

    Lets the client render the data graph from one request instead of
    walking the catalog tree (databases → schemas → tables) with a request
    per level. Node count is capped at ``MAX_EXPLORATION_GRAPH_NODES``
    regardless of *limit*.
    """
    return {"data": service.fetch_data_exploration_graph(zone_ids=None, limit=limit)}


@router.get(
    "/exploration/tables/{table_id}/details",
    response_model=TableExplorationDetailsResponse,
)
def get_table_exploration_details(
    table_id: str,
    skip: int = SKIP_QUERY,
    limit: int | None = LIMIT_QUERY,
) -> dict:
    """Columns-adjacent SQL and one ordered page of Term details for an Exploration modal."""
    return {
        "data": service.fetch_table_exploration_details(
            table_id, zone_ids=None, skip=skip, limit=limit
        )
    }


@router.get(
    "/exploration/terms/{term_id}/details",
    response_model=TermExplorationDetailsResponse,
)
def get_term_exploration_details(
    term_id: str,
    skip: int = SKIP_QUERY,
    limit: int | None = LIMIT_QUERY,
) -> dict:
    """One ordered page of Tables linked to a Term, for expanding it on the Exploration graph."""
    return {
        "data": service.fetch_term_exploration_details(
            term_id, zone_ids=None, skip=skip, limit=limit
        )
    }


@router.get(
    "/exploration/columns/{column_id}/details",
    response_model=ColumnExplorationDetailsResponse,
)
def get_column_exploration_details(column_id: str) -> dict:
    """The Column's own ColumnAttribute, FK target, referencing FK Columns and Sql queries, for expanding it on the Exploration graph."""
    return {"data": service.fetch_column_exploration_details(column_id, zone_ids=None)}


@router.get(
    "/exploration/column-attributes/{attr_id}/details",
    response_model=ColumnAttributeExplorationDetailsResponse,
)
def get_column_attribute_exploration_details(
    attr_id: str,
    skip: int = SKIP_QUERY,
    limit: int | None = LIMIT_QUERY,
) -> dict:
    """The ColumnAttribute's own owning Term and one ordered page of every
    linked Column, for expanding it on the Exploration graph.
    """
    return {
        "data": service.fetch_column_attribute_exploration_details(
            attr_id, zone_ids=None, skip=skip, limit=limit
        )
    }


@router.get(
    "/exploration/sql-attributes/{attr_id}/details",
    response_model=SqlAttributeExplorationDetailsResponse,
)
def get_sql_attribute_exploration_details(attr_id: str) -> dict:
    """A SqlAttribute's own Sql query and owning Term, for expanding it on the Exploration graph."""
    return {
        "data": service.fetch_sql_attribute_exploration_details(attr_id, zone_ids=None)
    }


@router.get(
    "/exploration/sql/{sql_id}/details",
    response_model=SqlExplorationDetailsResponse,
)
def get_sql_exploration_details(sql_id: str) -> dict:
    """CustomAnalysis, Column, Table and SqlAttribute nodes hanging off this Sql node, for expanding it on the Exploration graph."""
    return {"data": service.fetch_sql_exploration_details(sql_id, zone_ids=None)}


@router.get(
    "/exploration/terms/{term_id}/path/{other_term_id}",
    response_model=ExplorationLinkPathResponse,
)
def get_semantic_link_path(term_id: str, other_term_id: str) -> dict:
    """The real hop chain(s) behind one term↔term Exploration graph edge.

    Lets the client graft the actual Table (and, for a HAS_ATTRIBUTE/
    SEMANTIC_FK hop, Column/ColumnAttribute) nodes on and highlight them
    once the edge between ``term_id``/``other_term_id`` is clicked, instead
    of just showing the collapsed `GraphLink.relationship_types` label.
    """
    return {
        "data": service.fetch_semantic_link_path(term_id, other_term_id, zone_ids=None)
    }


@router.get(
    "/exploration/nodes/{node_id}/relationships",
    response_model=ExplorationRelatedNodesResponse,
)
def get_exploration_node_relationships(
    node_id: str = Path(
        description="Table id when *layer* is ``data``, Term id when it is ``semantic``."
    ),
    layer: Literal["data", "semantic"] = Query(
        description=(
            "Which graph to walk: ``data`` relates tables sharing a SQL query or "
            "a foreign key, ``semantic`` relates terms sharing a table."
        )
    ),
    skip: int = SKIP_QUERY,
    limit: int | None = LIMIT_QUERY,
) -> dict:
    """Return one ordered page of nodes related to ``node_id``."""
    return {
        "data": service.fetch_exploration_related_nodes(
            node_id,
            layer,
            zone_ids=None,
            skip=skip,
            limit=limit,
        )
    }


@router.get("/exploration/tables/zones", response_model=TableZonesResponse)
def list_table_zones() -> dict:
    """Return ``{table_id: [zone, ...]}`` for every Table.

    Used by the Exploration graph to render Zone chips on every data node
    without a per-node request.
    """
    return {"data": service.fetch_table_zones_map(zone_ids=None)}


@router.get(
    "/exploration/semantic-graph", response_model=SemanticExplorationGraphResponse
)
def get_semantic_exploration_graph(limit: int = _LIMIT_QUERY) -> dict:
    """Return the full semantic-layer Exploration graph (``{nodes, links}``).

    Lets the client render the semantic graph from one request instead of
    fetching related terms once per term (an N+1). Node count is capped at
    ``MAX_EXPLORATION_GRAPH_NODES`` regardless of *limit*.
    """
    return {
        "data": service.fetch_semantic_exploration_graph(
            zone_ids=None,
            limit=limit,
        )
    }
