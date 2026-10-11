# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""API route handlers for accounts and groups.

Groups are stored here. Accounts are not: they stay in Better Auth's
``frontend."user"`` table, which this service only reads. There is no
Frontegg. A group's role is ``admin`` or ``viewer``, and ``permissions`` on
the response is that role expanded — see ``permissions.py``.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from auto_ontology.dal import groups as dal
from auto_ontology.dal.schema import GROUP_ROLE_ADMIN, GROUP_ROLE_VIEWER
from auto_ontology.server.identity import resolve_internal_user
from auto_ontology.server.responses import (
    GroupListResponse,
    GroupResponse,
    IdResponse,
    UsersAndGroupsResponse,
)
from auto_ontology.server.users import postgres_dal
from auto_ontology.server.users.permissions import permissions_for_role

router = APIRouter()

#: Longest name a group may have, counted after trimming.
MAX_GROUP_NAME_LENGTH = 100

_ROLES = (GROUP_ROLE_ADMIN, GROUP_ROLE_VIEWER)


class GroupCreate(BaseModel):
    name: str
    description: str | None = None
    color: str
    role: str
    user_ids: list[str] = []


class GroupUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    color: str | None = None
    role: str | None = None
    user_ids: list[str] | None = None


def _require_admin(request: Request) -> str:
    """The caller, when the gateway named an admin.

    Writes are refused without that. Reads are not: the gateway in front of
    this service is the permission gate for listing, the same way it is for
    zones, and FastAPI is only reachable on the private network.
    """
    user_id = resolve_internal_user(request, required=False)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authenticated user required")
    if not postgres_dal.is_admin(user_id):
        raise HTTPException(
            status_code=403, detail="Only admins can perform this action"
        )
    return user_id


def _name(name: str) -> str:
    normalized = name.strip()
    if normalized == "":
        raise HTTPException(status_code=400, detail="Group name is required")
    if len(normalized) > MAX_GROUP_NAME_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Group name must be at most {MAX_GROUP_NAME_LENGTH} characters",
        )
    return normalized


def _color(color: str) -> str:
    normalized = color.strip()
    if normalized == "":
        raise HTTPException(status_code=400, detail="Group color is required")
    return normalized


def _role(role: str) -> str:
    normalized = role.strip()
    if normalized not in _ROLES:
        raise HTTPException(status_code=400, detail="Role must be admin or viewer")
    return normalized


def _description(description: str | None) -> str | None:
    if description is None:
        return None
    return description.strip() or None


def _user_ids(raw: list[str]) -> list[str]:
    seen: set[str] = set()
    user_ids: list[str] = []
    for item in raw:
        user_id = item.strip()
        if user_id == "":
            raise HTTPException(status_code=400, detail="User id is required")
        if user_id in seen:
            continue
        seen.add(user_id)
        user_ids.append(user_id)
    return user_ids


def _present(group: dict[str, Any]) -> dict[str, Any]:
    """Add the role list and the permissions that role expands to."""
    role = group["role"]
    presented = dict(group)
    presented["roles"] = [{"id": role, "name": role}]
    presented["permissions"] = permissions_for_role(role)
    return presented


def _translate(exc: Exception) -> HTTPException:
    if isinstance(exc, dal.GroupNameTaken):
        return HTTPException(status_code=409, detail=str(exc))
    if isinstance(exc, dal.UnknownGroupMembers):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, ValueError):
        return HTTPException(status_code=400, detail=str(exc))
    raise exc


@router.get("/users", response_model=UsersAndGroupsResponse)
def list_users_and_groups() -> dict:
    """Return every account and every group.

    Each account lists the ids of the groups it belongs to. Each group lists
    its members, its role, and the permissions that role grants. A membership
    whose account has been deleted stays on the group, with an empty name,
    and is absent from ``users``.
    """
    directory = dal.list_users_and_groups()
    return {
        "data": {
            "users": directory["users"],
            "groups": [_present(group) for group in directory["groups"]],
        }
    }


@router.get("/users/{user_id}/groups", response_model=GroupListResponse)
def list_groups_for_user(user_id: str) -> dict:
    """Return the groups one account belongs to.

    404 when the account does not exist. An account that exists and belongs
    to nothing answers 200 with an empty list — that is the empty state, not
    a missing account.
    """
    if dal.get_account(user_id) is None:
        raise HTTPException(status_code=404, detail=f"User {user_id!r} not found")
    rows = [_present(group) for group in dal.groups_for_user(user_id)]
    return {"data": rows, "count": len(rows)}


@router.get("/groups", response_model=GroupListResponse)
def list_groups() -> dict:
    """Return every group, ordered by name, with its members."""
    rows = [_present(group) for group in dal.list_groups()]
    return {"data": rows, "count": len(rows)}


@router.get("/groups/{group_id}", response_model=GroupResponse)
def get_group(group_id: str) -> dict:
    """Return one group.

    404 for an unknown id rather than an empty group: the caller asked for a
    row it expects to exist.
    """
    group = dal.get_group(group_id)
    if group is None:
        raise HTTPException(status_code=404, detail=f"Group {group_id!r} not found")
    return {"data": _present(group)}


@router.post("/groups", status_code=201, response_model=GroupResponse)
def create_group(request: Request, body: GroupCreate) -> dict:
    """Create a group and its members. Requires the admin role.

    The caller is the gateway's identity header. A missing header is 401; a
    caller who is not an admin is 403.

    409 when another group already holds the name, compared
    case-insensitively on the trimmed value. 400 when a member id is not an
    account, or when the name, color, or role is missing.
    """
    _require_admin(request)
    try:
        group = dal.create_group(
            name=_name(body.name),
            description=_description(body.description),
            color=_color(body.color),
            role=_role(body.role),
            user_ids=_user_ids(body.user_ids),
        )
    except (dal.GroupNameTaken, dal.UnknownGroupMembers, ValueError) as exc:
        raise _translate(exc) from exc
    return {"data": _present(group)}


@router.patch("/groups/{group_id}", response_model=GroupResponse)
def update_group(request: Request, group_id: str, body: GroupUpdate) -> dict:
    """Update a group's fields and, when sent, replace its members.

    Requires the admin role, for the same reason create does.

    Omitting ``user_ids`` leaves membership alone. Sending a list replaces
    it, and an empty list clears it. A patch that names no field returns the
    group unchanged.

    404 for an unknown id, 409 for a name another group holds, 400 for a
    member id that is not an account.
    """
    _require_admin(request)
    provided = body.model_fields_set
    updates: dict[str, Any] = {}
    if "name" in provided:
        updates["name"] = _name(body.name or "")
    if "description" in provided:
        updates["description"] = _description(body.description)
    if "color" in provided:
        updates["color"] = _color(body.color or "")
    if "role" in provided:
        updates["role"] = _role(body.role or "")

    user_ids = (
        _user_ids(body.user_ids)
        if "user_ids" in provided and body.user_ids is not None
        else None
    )
    if not updates and user_ids is None:
        group = dal.get_group(group_id)
        if group is None:
            raise HTTPException(status_code=404, detail=f"Group {group_id!r} not found")
        return {"data": _present(group)}

    try:
        group = dal.update_group(group_id=group_id, updates=updates, user_ids=user_ids)
    except (dal.GroupNameTaken, dal.UnknownGroupMembers, ValueError) as exc:
        raise _translate(exc) from exc
    if group is None:
        raise HTTPException(status_code=404, detail=f"Group {group_id!r} not found")
    return {"data": _present(group)}


@router.delete("/groups/{group_id}", response_model=IdResponse)
def delete_group(request: Request, group_id: str) -> dict:
    """Delete one group. Its membership is deleted with it.

    Requires the admin role. 404 rather than 204 for an id that is not
    there: the caller deleted from a list it has already read, so a missing
    group means that list is stale.
    """
    _require_admin(request)
    if not dal.delete_group(group_id):
        raise HTTPException(status_code=404, detail=f"Group {group_id!r} not found")
    return {"data": {"id": group_id}}
