# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""API routes for UI-managed database connections."""

from __future__ import annotations

import logging
import os
from typing import Any, TypedDict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from gsf.dal.connections import list_connections as _list_connections
from gsf.server.connections import service

logger = logging.getLogger(__name__)

router = APIRouter()


class ConnectionBody(BaseModel):
    connection: dict[str, Any]


class PublicConnection(TypedDict):
    database_name: str
    connection: dict[str, Any]


def _serialize_connection(connection: dict[str, Any]) -> PublicConnection:
    """Shape a connection object into the public connection payload."""
    return {
        "database_name": str(connection.get("database") or ""),
        "connection": connection,
    }


@router.get("/connections/source")
def is_env_source() -> bool:
    """Report whether connections are managed via the ``CONNECTION_STRINGS`` env."""
    connection_strings = os.environ.get("CONNECTION_STRINGS", "")
    connection_strings_exists = [
        cs.strip() for cs in connection_strings.split(",") if cs.strip()
    ]
    return bool(connection_strings_exists)


@router.get("/connections")
def list_connections() -> dict:
    rows = [_serialize_connection(conn) for conn in _list_connections()]
    return {"data": rows, "count": len(rows)}


@router.post("/connections/test")
def test_connection(body: ConnectionBody) -> dict:
    try:
        service.test_connection(body.connection)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Connection test failed: {exc}",
        ) from exc
    return {"success": True}


@router.post("/connections", status_code=201)
def create_connection(body: ConnectionBody) -> dict:
    try:
        row = service.create_connection(connection=body.connection)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Failed to create connection: {exc}",
        ) from exc

    return {"data": _serialize_connection(row)}


@router.delete("/connections/{database_name}")
def delete_connection(database_name: str) -> dict:
    try:
        row = service.delete_connection(database_name)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Failed to delete connection: {exc}",
        ) from exc

    if row is None:
        raise HTTPException(status_code=404, detail="Connection not found")

    return {"data": row}
