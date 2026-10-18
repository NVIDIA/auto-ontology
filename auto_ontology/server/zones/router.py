# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""API route handlers for semantic zones."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from auto_ontology.dal import zones as dal
from auto_ontology.server.owners import (
    UnassignableOwner,
    assignable_member_ids,
    owner_from_body,
)
from auto_ontology.server.users import postgres_dal
from auto_ontology.server.responses import IdResponse, ZoneListResponse, ZoneResponse

router = APIRouter()

#: Illumex names this ``zone.write``. Here the write action is ``manage``.
ZONE_MANAGE = "zone:manage"


class ZoneCreate(BaseModel):
    name: str
    description: str | None = None
    color: str
    items: list[str]
    created_by: str  # user_id of the requesting user (must be admin)
    owner_id: str | None = None
    members: list[str] = []


class ZoneUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    color: str | None = None
    items: list[str] | None = None
    owner_id: str | None = None
    members: list[str] | None = None


class ZoneStatusUpdate(BaseModel):
    enabled: bool


def _require_admin(user_id: str) -> None:
    """Raise 403 when *user_id* is not an admin per PostgreSQL."""
    if not postgres_dal.is_admin(user_id):
        raise HTTPException(
            status_code=403, detail="Only admins can perform this action"
        )


def _members(values: list[str] | None) -> list[str]:
    """Participant ids, each a user or group that can manage zones."""
    try:
        return assignable_member_ids(values, ZONE_MANAGE)
    except UnassignableOwner as exc:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Zone member {exc.owner_id!r} is not a user or group "
                "that can manage zones"
            ),
        ) from exc


@router.get("/zones", response_model=ZoneListResponse)
def list_zones(uid: str = Query(..., description="Requesting user id")) -> dict:
    """Return every zone to the requesting user."""
    rows = dal.list_zones()
    return {"data": rows, "count": len(rows)}


@router.get("/zones/{zone_id}", response_model=ZoneResponse)
def get_zone(
    zone_id: str,
    uid: str = Query(..., description="Requesting user id"),
) -> dict:
    """Return one zone with its catalog data items."""
    row = dal.get_zone_by_id(zone_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Zone {zone_id!r} not found")
    return {"data": row}


@router.post("/zones", status_code=201, response_model=ZoneResponse)
def create_zone(body: ZoneCreate) -> dict:
    """Create a zone and attach catalog items. Requires admin role.

    ``owner_id`` is one user or one group that can manage zones, or omitted
    to leave the zone unowned. ``members`` are users and groups with that
    same permission; each is a participant and can see the zone's data. A
    participant is not the owner. Returns 400 when an id cannot be assigned.
    """
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
            owner_id=owner_from_body(body.owner_id, ZONE_MANAGE),
            member_ids=_members(body.members),
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"data": row}


@router.patch("/zones/{zone_id}", response_model=ZoneResponse)
def update_zone(zone_id: str, body: ZoneUpdate) -> dict:
    """Update zone fields and optionally replace catalog items or participants.

    ``owner_id`` and ``members``, when sent, follow the same rules as create.
    Omit either to leave it unchanged. An empty ``members`` list clears
    participants. Returns 400 when an id cannot be assigned.
    """
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

    if "owner_id" in provided_fields:
        field_updates["owner_id"] = owner_from_body(body.owner_id, ZONE_MANAGE)

    try:
        row = dal.update_zone(
            zone_id=zone_id,
            updates=field_updates,
            item_ids=body.items if "items" in provided_fields else None,
            member_ids=(
                _members(body.members)
                if "members" in provided_fields and body.members is not None
                else None
            ),
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if row is None:
        raise HTTPException(status_code=404, detail=f"Zone {zone_id!r} not found")
    return {"data": row}


@router.delete("/zones/{zone_id}", response_model=IdResponse)
def delete_zone(zone_id: str) -> dict:
    """Delete one zone by id."""
    if not dal.delete_zone(zone_id):
        raise HTTPException(status_code=404, detail=f"Zone {zone_id!r} not found")
    return {"data": {"id": zone_id}}


@router.patch("/zones/{zone_id}/status", response_model=ZoneResponse)
def set_zone_status(
    zone_id: str,
    body: ZoneStatusUpdate,
    admin_uid: str = Query(..., description="Admin user id performing the request"),
) -> dict:
    """Enable or disable a zone.  Requires admin role.

    Disabling clears the zone's ``enabled`` flag
    (re-enabling reverses it), which immediately revokes the catalog access
    the zone granted without deleting it.
    """
    _require_admin(admin_uid)
    row = dal.set_zone_enabled(zone_id, body.enabled)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Zone {zone_id!r} not found")
    return {"data": row}
