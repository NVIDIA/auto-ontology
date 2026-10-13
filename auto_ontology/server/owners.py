# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Who may be chosen as an owner, and who may edit a row.

An owner is one account or one group. The id is stored on the row so the
page can say who is responsible. It does not decide who may edit. Editing
is the role's permission.

An admin may always edit: that role holds every write. Anyone else may edit
when their role grants the permission the row asks for. A viewer does not
grant ``catalog:edit``, so a viewer cannot edit a catalog row.

The gateway is what names the caller. A write that arrives without that
header is not checked here: FastAPI is only reachable on the private network,
and the header is not authenticated on its own.
"""

from __future__ import annotations

from fastapi import HTTPException, Request

from auto_ontology.dal import groups as groups_dal
from auto_ontology.dal.owners import EditSubject
from auto_ontology.server.identity import resolve_internal_user
from auto_ontology.server.users import postgres_dal
from auto_ontology.server.users.permissions import permissions_for_role


class UnassignableOwner(ValueError):
    """The id is not an account or a group allowed to own this kind of row."""

    def __init__(self, owner_id: str) -> None:
        self.owner_id = owner_id
        super().__init__(
            f"Owner {owner_id!r} is not a user or group that can edit this"
        )


def normalize_owner_id(value: str | None) -> str | None:
    """Blank means unassigned. A non-blank id is returned trimmed."""
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def decide_edit(*, role: str | None, permission: str) -> bool:
    """Whether a role that holds *permission* may edit.

    The stored owner is not an argument. An admin always may. Anyone else
    may when *permission* is one their role grants.
    """
    if role == "admin":
        return True
    return permission in permissions_for_role(role or "")


def can_edit_entity(*, user_id: str, permission: str) -> bool:
    """:func:`decide_edit` with the account's role read for it."""
    return decide_edit(role=postgres_dal.get_user_role(user_id), permission=permission)


def require_assignable(owner_id: str | None, permission: str) -> str | None:
    """*owner_id* when it may own a row gated by *permission*, or null to clear.

    Raises :class:`UnassignableOwner` when the id names nothing, or names an
    account or group whose role does not grant *permission*.
    """
    if owner_id is None:
        return None
    account = groups_dal.get_account(owner_id)
    if account is not None:
        role = account["role"] or "viewer"
        if permission in permissions_for_role(role):
            return owner_id
        raise UnassignableOwner(owner_id)
    group = groups_dal.get_group(owner_id)
    if group is not None and permission in permissions_for_role(group["role"]):
        return owner_id
    raise UnassignableOwner(owner_id)


def enforce_edit(request: Request, subject: EditSubject) -> None:
    """403 when the gateway named a caller whose role lacks the permission.

    No header means no check. See the module docstring. The row's owner is
    not consulted.
    """
    user_id = resolve_internal_user(request, required=False)
    if not user_id:
        return
    if can_edit_entity(user_id=user_id, permission=subject.permission):
        return
    raise HTTPException(
        status_code=403, detail="You do not have permission to edit this"
    )


def owner_from_body(value: str | None, permission: str) -> str | None:
    """The owner a write asked for, or 400 when it cannot be assigned."""
    try:
        return require_assignable(normalize_owner_id(value), permission)
    except UnassignableOwner as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
