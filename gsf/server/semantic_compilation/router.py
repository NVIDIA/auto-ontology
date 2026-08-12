# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""API route for triggering semantic compilation on the ingestion service.

The settings UI stores the on/off flag directly in Postgres (via Prisma); these
routes only relay the "run it now" and "reset the layer" requests to the
ingestion service, mirroring how connection changes trigger ingest (see
``connections/service.py``).
"""

from __future__ import annotations

import logging

from fastapi import APIRouter

from gsf.dal.terms import semantic_layer_calculated
from gsf.server.ingestion.proxy import trigger_semantic_compile, trigger_semantic_reset
from gsf.server.responses import SemanticStatusResponse, StatusResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/semantic-compilation/trigger", status_code=202, response_model=StatusResponse
)
async def trigger() -> dict[str, str]:
    """Ask the ingestion service to run (and schedule) semantic compilation."""
    trigger_semantic_compile()
    return {"status": "accepted"}


@router.post(
    "/semantic-compilation/reset", status_code=202, response_model=StatusResponse
)
async def reset() -> dict[str, str]:
    """Ask the ingestion service to rebuild every database's semantic layer.

    The service deletes the semantic nodes and embeddings and compiles them
    again, so the layer is rebuilt without waiting for the next scheduled run.
    """
    trigger_semantic_reset()
    return {"status": "accepted"}


@router.get("/semantic-compilation/status", response_model=SemanticStatusResponse)
def semantic_status() -> dict[str, bool]:
    """Report whether the semantic layer has been calculated (any Term exists)."""
    return {"calculated": semantic_layer_calculated()}
