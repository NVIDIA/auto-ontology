# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""API route handlers for User nodes in Neo4j.

Only viewers have a ``:User`` node — admins need none, since their access
is derived entirely from PostgreSQL (see
``gsf.server.users.postgres_dal``).  ``GET``/``POST`` here reflect that: an
admin id will 404 on read, and ``upsert_user`` removes the node (if any)
instead of creating one.

All endpoints are server-side only — there is no public-facing UI for these
routes.  They are called by Next.js server actions / route handlers that
already hold a validated Better Auth session.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, EmailStr

from gsf.dal import users as dal

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
    """Return all viewer User nodes stored in Neo4j. Admins have no node."""
    rows = dal.list_users()
    return {"data": rows, "count": len(rows)}


# ---------------------------------------------------------------------------
# GET /api/users/{user_id}
# ---------------------------------------------------------------------------


@router.get("/users/{user_id}")
def get_user(user_id: str) -> dict:
    """Return a single viewer User node by id.

    Returns 404 when no User with *user_id* exists in the graph — which is
    always the case for admins, since they have no node.
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
    """Sync a user into Neo4j according to its PostgreSQL role.

    Viewers get a ``:User`` node created/updated in place, keyed by the
    PostgreSQL ``id`` from Better Auth.  Admins get no node — if *body.id*
    already had one (e.g. before being promoted), it is deleted along with
    any zone grants.
    """
    row = dal.upsert_user(
        user_id=body.id,
        email=body.email,
        name=body.name,
        role=body.role,
    )
    return {"data": row}
