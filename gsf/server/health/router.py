# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Liveness/readiness endpoint — probes Postgres connectivity."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from gsf.dal.connections import verify_connectivity
from gsf.server.responses import HealthResponse

router = APIRouter()


def _check_store() -> dict[str, str]:
    """Probe the pooled connection the application actually uses.

    Through the DAL rather than a fresh ``psycopg.connect``: a raw connection
    proves the server is reachable, which is not the same as proving this
    process can get a usable connection out of its pool — and the pool is what
    every request depends on.
    """
    try:
        verify_connectivity()
        return {"status": "ok"}
    except Exception as exc:
        return {"status": "error", "detail": (str(exc) or type(exc).__name__)[:200]}


@router.get(
    "/health",
    responses={
        200: {"model": HealthResponse, "description": "All dependencies reachable"},
        503: {"model": HealthResponse, "description": "Postgres unreachable"},
    },
)
def health() -> JSONResponse:
    postgres = _check_store()
    healthy = postgres["status"] == "ok"
    body: dict[str, Any] = {
        "status": "ok" if healthy else "degraded",
        "postgres": postgres,
    }
    return JSONResponse(status_code=200 if healthy else 503, content=body)
