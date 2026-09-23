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

from auto_ontology.dal.terms import semantic_layer_calculated
from auto_ontology.ingestion_service.history import (
    get_last_failure_if_most_recent,
    get_last_successful_run,
)
from auto_ontology.server.ingestion.proxy import (
    is_semantic_compilation_running,
    trigger_semantic_compile,
    trigger_semantic_reset,
)
from auto_ontology.server.responses import SemanticStatusResponse, StatusResponse

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
def semantic_status() -> dict[str, bool | str | None]:
    """Report whether the semantic layer has been calculated (any Term exists),
    whether the ingestion service is compiling it right now, when the last
    pass completed successfully, and when it last failed.

    ``running`` needs the live ingestion process, so it goes through the HTTP
    proxy and degrades to ``False`` if that's unreachable. ``last_success_at``
    and ``last_failure_at`` are read straight from the ``semantic_compilation_history``
    table instead — a temporarily unreachable ingestion service has no bearing
    on Postgres, so history stays visible even while it's down (see
    ``auto_ontology/server/ingestion/proxy.py``).
    """
    return {
        "calculated": semantic_layer_calculated(),
        "running": is_semantic_compilation_running(),
        "last_success_at": get_last_successful_run(),
        "last_failure_at": get_last_failure_if_most_recent(),
    }
