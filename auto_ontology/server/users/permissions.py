# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Permissions a group's role expands to.

Illumex asks Frontegg which permissions each of a group's roles grants, then
returns that set on the group. There is no Frontegg here. A group carries one
of the two roles an account already has, and this module is the expansion.

The lists are ``frontend/auth/access.json``. The Next.js auth layer imports
that same file, so a group's ``permissions`` and an account's role describe
one vocabulary. Better Auth's built-in ``user`` and ``session`` statements
are not in the file: those govern account administration, which stays on the
account's own role and is not something a group grants.
"""

from __future__ import annotations

import json
from pathlib import Path

from auto_ontology.dal.schema import GROUP_ROLE_ADMIN, GROUP_ROLE_VIEWER

_ACCESS_FILE = Path(__file__).resolve().parents[3] / "frontend" / "auth" / "access.json"
_ROLES = (GROUP_ROLE_ADMIN, GROUP_ROLE_VIEWER)


def _load() -> dict[str, list[str]]:
    raw = json.loads(_ACCESS_FILE.read_text())
    loaded: dict[str, list[str]] = {}
    for role in _ROLES:
        resources = raw[role]
        loaded[role] = [
            f"{resource}:{action}"
            for resource, actions in resources.items()
            for action in actions
        ]
    return loaded


_BY_ROLE = _load()


def permissions_for_role(role: str) -> list[str]:
    """``resource:action`` strings for *role*, or nothing for an unknown one.

    An unknown role grants nothing. The column constraint already limits
    stored roles to the two known ones; this is the read path's own refusal
    to invent permissions for a value it does not recognise.
    """
    return list(_BY_ROLE.get(role, ()))
