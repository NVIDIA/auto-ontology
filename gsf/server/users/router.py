# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""API route handlers for User nodes in Neo4j.

All endpoints are server-side only — there is no public-facing UI for these
routes.  They are called by Next.js server actions / route handlers that
already hold a validated Better Auth session.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, EmailStr

from gsf.neo4j import users as dal

router = APIRouter()


class UserCreate(BaseModel):
    id: str
    email: EmailStr
    name: str
    role: str


# ---------------------------------------------------------------------------
# GET /api/users
# ---------------------------------------------------------------------------


@router.get("/users")
def list_users() -> dict:
    """Return all User nodes stored in Neo4j."""
    rows = dal.list_users()
    return {"data": rows, "count": len(rows)}


# ---------------------------------------------------------------------------
# GET /api/users/{user_id}
# ---------------------------------------------------------------------------


@router.get("/users/{user_id}")
def get_user(user_id: str) -> dict:
    """Return a single User node by id.

    Returns 404 when no User with *user_id* exists in the graph.
    """
    row = dal.get_user(user_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"User {user_id!r} not found")
    return {"data": row}


# ---------------------------------------------------------------------------
# POST /api/users
# ---------------------------------------------------------------------------


@router.post("/users", status_code=201)
def upsert_user(body: UserCreate) -> dict:
    """Create or update a User node in Neo4j.

    Idempotent: if a node with the same *id* already exists it is updated
    in place.  The PostgreSQL ``id`` from Better Auth is used as the graph
    node's natural key so both stores stay in sync.
    """
    row = dal.upsert_user(
        user_id=body.id,
        email=body.email,
        name=body.name,
        role=body.role,
    )
    return {"data": row}
