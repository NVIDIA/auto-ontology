# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""API route for triggering semantic compilation on the ingestion service.

The settings UI stores the on/off flag directly in Postgres (via Prisma); this
route only relays the "run it now" request to the ingestion service, mirroring
how connection changes trigger ingest (see ``connections/service.py``).
"""

from __future__ import annotations

import logging

from fastapi import APIRouter

from gsf.dal.terms import semantic_layer_calculated
from gsf.server.ingestion.proxy import trigger_semantic_compile

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/semantic-compilation/trigger", status_code=202)
async def trigger() -> dict[str, str]:
    """Ask the ingestion service to run (and schedule) semantic compilation."""
    trigger_semantic_compile()
    return {"status": "accepted"}


@router.get("/semantic-compilation/status")
def semantic_status() -> dict[str, bool]:
    """Report whether the semantic layer has been calculated (any Term exists)."""
    return {"calculated": semantic_layer_calculated()}
