# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read-only routes over the Apache AGE projection.

Both routes answer 503 while ``AGE_GRAPH_ENABLED`` is unset, so a deployment
without AGE publishes the same OpenAPI surface and simply reports the feature
as off. Neither route accepts Cypher: the traversal is fixed server-side, which
keeps the agent-facing surface about meaning (a Term and what surrounds it)
rather than an open query language over the catalog.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Path, Query
from pydantic import Field

from auto_ontology.graph import age, queries
from auto_ontology.server.models import ApiModel

router = APIRouter()
logger = logging.getLogger(__name__)


class GraphStatusResponse(ApiModel):
    enabled: bool = Field(description="Whether AGE_GRAPH_ENABLED is set.")
    graph: str | None = Field(default=None, description="AGE graph name.")
    built: bool = Field(default=False, description="Whether the graph exists.")
    synced_at: str | None = Field(
        default=None, description="ISO timestamp of the last completed sync."
    )
    vertices: dict[str, int] = Field(
        default_factory=dict, description="Vertex count per label."
    )
    edges: dict[str, int] = Field(
        default_factory=dict, description="Edge count per type."
    )


class GraphNode(ApiModel):
    id: str = Field(description="The object's Auto Ontology id.")
    label: str = Field(description="Term, Table, Column, ColumnAttribute, ...")
    properties: dict[str, Any] = Field(default_factory=dict)


class GraphEdge(ApiModel):
    type: str = Field(description="REPRESENTS, CONTAINS, PROPERTY_OF, ...")
    source: str
    target: str
    properties: dict[str, Any] = Field(default_factory=dict)


class TermNeighborhoodResponse(ApiModel):
    term_id: str
    depth: int
    truncated: bool = Field(description="True when max_nodes stopped the walk early.")
    nodes: list[GraphNode]
    edges: list[GraphEdge]


def _require_enabled() -> None:
    if not age.enabled():
        raise HTTPException(
            status_code=503,
            detail="The AGE graph projection is not enabled (set AGE_GRAPH_ENABLED=true).",
        )


@router.get("/graph/status", response_model=GraphStatusResponse)
def graph_status() -> dict:
    """Report whether the AGE graph projection is on, built, and how large it is.

    ``synced_at`` is when the graph was last rebuilt from the catalog. Edits
    made after it are not in the graph until the next sync, which runs at the
    end of every semantic compilation pass.
    """
    if not age.enabled():
        return {"enabled": False}
    try:
        return {"enabled": True, **queries.graph_status()}
    except age.AgeUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get(
    "/graph/terms/{term_id}/neighborhood", response_model=TermNeighborhoodResponse
)
def term_neighborhood(
    term_id: str = Path(description="Term id."),
    depth: int = Query(
        default=1,
        ge=1,
        le=queries.MAX_DEPTH,
        description="Hops to expand from the Term.",
    ),
    max_nodes: int = Query(
        default=queries.DEFAULT_MAX_NODES,
        ge=1,
        le=queries.MAX_NODES,
        description="Stop expanding once this many vertices are collected.",
    ),
) -> dict:
    """Return the vertices and edges within *depth* hops of one Term.

    Every vertex is reported by its Auto Ontology id and label, every edge by
    its type and endpoint ids, so the result can be followed up with the
    regular Terms, Tables and attribute routes. ``truncated`` is true when the
    walk stopped at *max_nodes*.
    """
    _require_enabled()
    try:
        result = queries.term_neighborhood(term_id, depth=depth, max_nodes=max_nodes)
    except age.GraphNotBuilt as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except age.AgeUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if result is None:
        raise HTTPException(
            status_code=404, detail=f"Term {term_id!r} is not in the graph"
        )
    return result
