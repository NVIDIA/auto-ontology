# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Liveness/readiness endpoint — probes Neo4j and Postgres connectivity."""

from __future__ import annotations

from typing import Any

import psycopg
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from gsf.neo4j.connections import verify_connectivity
from gsf.vdb.config import get_postgres_connection_string

router = APIRouter()


def _check_neo4j() -> dict[str, str]:
    try:
        verify_connectivity()
        return {"status": "ok"}
    except Exception as exc:
        return {"status": "error", "detail": (str(exc) or type(exc).__name__)[:200]}


def _check_postgres() -> dict[str, str]:
    try:
        with psycopg.connect(
            get_postgres_connection_string(), connect_timeout=3
        ) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        return {"status": "ok"}
    except Exception as exc:
        return {"status": "error", "detail": (str(exc) or type(exc).__name__)[:200]}


@router.get("/health")
def health() -> JSONResponse:
    neo4j = _check_neo4j()
    postgres = _check_postgres()
    healthy = neo4j["status"] == "ok" and postgres["status"] == "ok"
    body: dict[str, Any] = {
        "status": "ok" if healthy else "degraded",
        "neo4j": neo4j,
        "postgres": postgres,
    }
    return JSONResponse(status_code=200 if healthy else 503, content=body)
