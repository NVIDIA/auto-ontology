# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP API routes for the ingestion service."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, Request
from gsf.ingestion_service.ingest import (
    trigger_delete_ingest,
    trigger_ingest,
    trigger_reset_semantic,
)
from gsf.server.responses import StatusResponse

router = APIRouter()


@router.get("/health", response_model=StatusResponse)
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/ingest", status_code=202, response_model=StatusResponse)
async def ingest_connection(connection: dict[str, Any] = Body(...)) -> dict[str, str]:
    """Trigger a non-blocking ingest for a single connection."""
    trigger_ingest(connection)
    return {"status": "accepted"}


@router.post("/ingest/delete", status_code=202, response_model=StatusResponse)
async def ingest_delete(database_name: str) -> dict[str, str]:
    """Trigger a non-blocking reset of a database's ingested data.

    ``database_name`` is required: this endpoint always targets one database.
    """
    trigger_delete_ingest(database_name)
    return {"status": "accepted"}


@router.post("/semantic/reset", status_code=202, response_model=StatusResponse)
async def semantic_reset(
    request: Request, database_name: str | None = None
) -> dict[str, str]:
    """Trigger a non-blocking reset of a database's semantic layer.

    Deletes the semantic nodes and embeddings, then asks for a compilation pass
    so the layer is rebuilt without waiting for the scheduler's next run.
    Omitting ``database_name`` resets every database. The deletion runs on its
    own thread while the compilation pass runs on the scheduler's task, so the
    two overlap.
    """
    trigger_reset_semantic(database_name)

    scheduler = request.app.state.semantic_scheduler

    scheduler.abort()
    if not scheduler.start():
        scheduler.trigger()
    return {"status": "accepted"}


@router.post("/semantic/compile", status_code=202, response_model=StatusResponse)
async def semantic_compile(request: Request) -> dict[str, str]:
    """Trigger a non-blocking semantic compilation pass.

    Ensures the scheduler is running first (``start()`` is idempotent), so this
    also takes effect when compilation is enabled while the service is already
    up — no restart needed. Runs on the scheduler's own task; when it finishes,
    the next automatic run is rescheduled for 24h later.
    """
    scheduler = request.app.state.semantic_scheduler
    # start() performs an immediate startup run on its own. Only ask for an
    # extra run when the scheduler was already running; otherwise the startup
    # run and the trigger would compile twice.
    if not scheduler.start():
        scheduler.trigger()
    return {"status": "accepted"}


@router.post("/semantic/stop", status_code=202, response_model=StatusResponse)
async def semantic_stop(request: Request) -> dict[str, str]:
    """Ask the in-flight semantic compilation pass to stop.

    The database being compiled finishes first — that work runs in a thread that
    cannot be interrupted — so the pass ends at the next database boundary. The
    scheduler stays alive and its next automatic run happens on schedule.
    """
    request.app.state.semantic_scheduler.abort()
    return {"status": "accepted"}
