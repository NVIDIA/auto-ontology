# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Health endpoints.

Two, deliberately. ``/health`` proves the dependencies are usable and backs
**readiness**. ``/health/live`` proves only that this process is serving and
backs **liveness** — a liveness probe that depends on Postgres asks the kubelet
to restart a healthy backend whenever the database is slow or its pool is
saturated, which removes capacity exactly when there is least to spare and can
kill a process mid-write.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from gsf.server.health.checks import check_schema, check_store
from gsf.server.responses import HealthResponse

router = APIRouter()


@router.get(
    "/health",
    responses={
        200: {"model": HealthResponse, "description": "All dependencies reachable"},
        503: {"model": HealthResponse, "description": "Postgres unreachable"},
    },
)
def health() -> JSONResponse:
    """Readiness probe for the API service.

    Two independent checks, because they fail independently and only one of
    them used to be made:

    * **postgres** — can this process get a usable connection out of its pool.
    * **migrations** — is that database at the revision this build expects.

    A reachable database with no tables in it passes the first and fails the
    second, and until the second existed this endpoint answered
    ``200 {"status": "ok"}`` for exactly that state while every catalog
    endpoint returned 500. A readiness probe that a load balancer believes has
    to be right about the second case too.

    This backs **readiness** only. Liveness is /health/live, which checks
    nothing: restarting the process cannot fix an unreachable database, so a
    liveness probe gated on one turns a database blip into a rolling restart.
    """
    postgres = check_store()
    migrations = check_schema()
    healthy = postgres["status"] == "ok" and migrations["status"] == "ok"
    body: dict[str, Any] = {
        "status": "ok" if healthy else "degraded",
        "postgres": postgres,
        "migrations": migrations,
    }
    return JSONResponse(status_code=200 if healthy else 503, content=body)


@router.get(
    "/health/live",
    responses={200: {"description": "The process is serving requests"}},
)
def liveness() -> JSONResponse:
    """Liveness only: no dependency is checked, and this must never fail.

    Restarting the process cannot fix an unreachable database, so making this
    depend on one converts a database blip into a rolling restart.
    """
    return JSONResponse(status_code=200, content={"status": "ok"})
