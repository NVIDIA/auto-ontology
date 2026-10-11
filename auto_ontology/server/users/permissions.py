# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Permissions a group's role expands to.

Illumex asks Frontegg which permissions each of a group's roles grants, then
returns that set on the group. There is no Frontegg here. A group carries one
of the two roles an account already has, and this module is the expansion.

The lists mirror the Auto Ontology resources in ``frontend/auth/auth-access.ts``.
Better Auth's built-in ``user`` and ``session`` statements are left out on
purpose: those govern account administration, which stays on the account's
own role and is not something a group grants. A change to the frontend
statements has to land here in the same change, or a group and an account
with the same role will disagree about what that role allows.
"""

from __future__ import annotations

from auto_ontology.dal.schema import GROUP_ROLE_ADMIN, GROUP_ROLE_VIEWER

_Statement = tuple[str, str]

_VIEWER: tuple[_Statement, ...] = (
    ("acronym", "read"),
    ("acronym", "create"),
    ("acronym", "update"),
    ("acronym", "delete"),
    ("prompt", "read"),
    ("prompt", "create"),
    ("prompt", "update"),
    ("prompt", "delete"),
    ("analysis", "read"),
    ("analysis", "manage"),
    ("conversation", "read"),
    ("conversation", "write"),
    ("conversation", "delete"),
    ("chat", "use"),
    ("catalog", "read"),
    ("catalog", "edit"),
    ("connection", "read"),
    ("zone", "read"),
    ("tag", "read"),
    ("apiToken", "manage"),
)

_ADMIN: tuple[_Statement, ...] = (
    ("analytics", "read"),
    ("acronym", "read"),
    ("acronym", "create"),
    ("acronym", "update"),
    ("acronym", "delete"),
    ("prompt", "read"),
    ("prompt", "create"),
    ("prompt", "update"),
    ("prompt", "delete"),
    ("sso", "read"),
    ("sso", "manage"),
    ("conversation", "read"),
    ("conversation", "write"),
    ("conversation", "delete"),
    ("chat", "use"),
    ("analysis", "read"),
    ("analysis", "manage"),
    ("catalog", "read"),
    ("catalog", "edit"),
    ("connection", "read"),
    ("connection", "manage"),
    ("zone", "read"),
    ("zone", "manage"),
    ("group", "read"),
    ("group", "manage"),
    ("tag", "read"),
    ("tag", "manage"),
    ("semanticCompilation", "read"),
    ("semanticCompilation", "manage"),
    ("visualization", "read"),
    ("visualization", "manage"),
    ("pii", "read"),
    ("pii", "manage"),
    ("modelInterchange", "export"),
    ("modelInterchange", "import"),
    ("apiToken", "manage"),
)

_BY_ROLE: dict[str, tuple[_Statement, ...]] = {
    GROUP_ROLE_ADMIN: _ADMIN,
    GROUP_ROLE_VIEWER: _VIEWER,
}


def permissions_for_role(role: str) -> list[str]:
    """``resource:action`` strings for *role*, or nothing for an unknown one.

    An unknown role grants nothing. The column constraint already limits
    stored roles to the two known ones; this is the read path's own refusal
    to invent permissions for a value it does not recognise.
    """
    statements = _BY_ROLE.get(role)
    if statements is None:
        return []
    return [f"{resource}:{action}" for resource, action in statements]
