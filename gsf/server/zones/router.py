# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""API route handlers for semantic zones."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from gsf.dal import users as users_dal
from gsf.server.zones import dal

router = APIRouter()


class ZoneCreate(BaseModel):
    name: str
    description: str | None = None
    color: str
    items: list[str]
    created_by: str  # user_id of the requesting user (must be admin)


class ZoneUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    color: str | None = None
    items: list[str] | None = None


class ZoneAccessGrant(BaseModel):
    user_id: str


def _require_admin(user_id: str) -> None:
    """Raise 403 when *user_id* is not an admin (or does not exist)."""
    user = users_dal.get_user(user_id)
    if user is None:
        raise HTTPException(status_code=403, detail="Forbidden")
    if user.get("role") != "admin":
        raise HTTPException(
            status_code=403, detail="Only admins can perform this action"
        )


@router.get("/zones")
def list_zones(uid: str = Query(..., description="Requesting user id")) -> dict:
    """Zones visible to *uid*.

    Admins see all zones; viewers see only zones they have been granted access to.
    """
    rows = dal.list_zones(uid)
    return {"data": rows, "count": len(rows)}


@router.get("/zones/{zone_id}")
def get_zone(
    zone_id: str,
    uid: str = Query(..., description="Requesting user id"),
) -> dict:
    """One zone with its catalog data items.

    Returns 404 when the zone does not exist or the user has no access.
    """
    row = dal.get_zone_by_id(zone_id, user_id=uid)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Zone {zone_id!r} not found")
    return {"data": row}


@router.post("/zones", status_code=201)
def create_zone(body: ZoneCreate) -> dict:
    """Create a zone and attach catalog items.  Requires admin role."""
    _require_admin(body.created_by)
    normalized_name = body.name.strip()
    if normalized_name == "":
        raise HTTPException(status_code=400, detail="Zone name is required")
    try:
        row = dal.create_zone(
            name=normalized_name,
            description=body.description,
            color=body.color,
            item_ids=body.items,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"data": row}


@router.patch("/zones/{zone_id}")
def update_zone(zone_id: str, body: ZoneUpdate) -> dict:
    """Update zone fields and optionally replace catalog data items."""
    field_updates: dict[str, str | None] = {}
    provided_fields = body.model_fields_set

    if "name" in provided_fields:
        raw_name = body.name or ""
        normalized_name = raw_name.strip()
        if normalized_name == "":
            raise HTTPException(status_code=400, detail="Zone name is required")
        field_updates["name"] = normalized_name

    if "description" in provided_fields:
        normalized_description = (body.description or "").strip()
        field_updates["description"] = normalized_description or None

    if "color" in provided_fields:
        field_updates["color"] = body.color

    try:
        row = dal.update_zone(
            zone_id=zone_id,
            updates=field_updates,
            item_ids=body.items if "items" in provided_fields else None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if row is None:
        raise HTTPException(status_code=404, detail=f"Zone {zone_id!r} not found")
    return {"data": row}


@router.delete("/zones/{zone_id}")
def delete_zone(zone_id: str) -> dict:
    """Delete one zone by id."""
    if not dal.delete_zone(zone_id):
        raise HTTPException(status_code=404, detail=f"Zone {zone_id!r} not found")
    return {"data": {"id": zone_id}}


# ---------------------------------------------------------------------------
# Zone access management  (admin → viewer)
# ---------------------------------------------------------------------------


@router.get("/zones/{zone_id}/access")
def list_zone_access(
    zone_id: str,
    admin_uid: str = Query(..., description="Admin user id performing the request"),
) -> dict:
    """List all users that currently have access to a zone.  Requires admin."""
    _require_admin(admin_uid)
    rows = dal.list_zone_users(zone_id)
    return {"data": rows, "count": len(rows)}


@router.post("/zones/{zone_id}/access", status_code=201)
def grant_zone_access(
    zone_id: str,
    body: ZoneAccessGrant,
    admin_uid: str = Query(..., description="Admin user id performing the request"),
) -> dict:
    """Grant a user access to a zone.  Requires admin."""
    _require_admin(admin_uid)
    granted = dal.grant_zone_access(zone_id=zone_id, user_id=body.user_id)
    if not granted:
        raise HTTPException(
            status_code=404,
            detail=f"Zone {zone_id!r} or user {body.user_id!r} not found",
        )
    return {"data": {"zone_id": zone_id, "user_id": body.user_id}}


@router.delete("/zones/{zone_id}/access/{user_id}")
def revoke_zone_access(
    zone_id: str,
    user_id: str,
    admin_uid: str = Query(..., description="Admin user id performing the request"),
) -> dict:
    """Revoke a user's access to a zone.  Requires admin."""
    _require_admin(admin_uid)
    revoked = dal.revoke_zone_access(zone_id=zone_id, user_id=user_id)
    if not revoked:
        raise HTTPException(
            status_code=404,
            detail=f"No access record found for user {user_id!r} on zone {zone_id!r}",
        )
    return {"data": {"zone_id": zone_id, "user_id": user_id}}
