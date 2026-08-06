# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""API routes for grading entity coverage of a free-text question."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from gsf.retrieval.entity_coverage.state import DEFAULT_MAX_DISTANCE
from gsf.server.metadata import service as dal

router = APIRouter()


class EntityCoverageRequest(BaseModel):
    """Payload for the entity-coverage route."""

    question: str = Field(..., min_length=1)
    max_distance: float = Field(
        default=DEFAULT_MAX_DISTANCE,
        gt=0.0,
        description=(
            "Maximum L2 vector distance for a candidate to count "
            "(lower score = closer match)."
        ),
    )


@router.post("/question-entity-coverage")
def entity_coverage(body: EntityCoverageRequest) -> dict:
    """Return ranked semantic candidates and a 0–1 entity coverage grade.

    Extracts entities from the question, retrieves semantic candidates, filters
    by vector distance, enriches via Neo4j, and grades how many entities have
    at least one covering ColumnAttribute candidate.

    Returns 422 when the flow cannot produce a result for the question.
    """
    try:
        result = dal.entity_coverage(
            body.question,
            max_distance=body.max_distance,
        )
    except dal.PredictionFlowError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"data": result}
