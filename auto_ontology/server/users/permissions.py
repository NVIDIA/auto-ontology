# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Permissions a group's role expands to.

Illumex asks Frontegg which permissions each of a group's roles grants, then
returns that set on the group. There is no Frontegg here. A group carries one
of the two roles an account already has, and this module is the expansion.

The lists are generated into this package from ``frontend/auth/access.json``,
which the Next.js auth layer imports directly, so a group's ``permissions``
and an account's role describe one vocabulary. Better Auth's built-in
``user`` and ``session`` statements are not in the file: those govern account
administration, which stays on the account's own role and is not something a
group grants.
"""

from __future__ import annotations

from importlib import resources
import json
from typing import Any

from auto_ontology.dal.schema import GROUP_ROLE_ADMIN, GROUP_ROLE_VIEWER

_ACCESS_RESOURCE = resources.files(__package__).joinpath("access.json")
_ROLES = (GROUP_ROLE_ADMIN, GROUP_ROLE_VIEWER)


def _invalid(message: str) -> RuntimeError:
    return RuntimeError(f"Invalid packaged permission resource access.json: {message}")


def _resources_for_role(raw: Any, role: str) -> dict[str, list[str]]:
    if not isinstance(raw, dict):
        raise _invalid("top-level value must be an object")
    resources_by_role = raw.get(role)
    if not isinstance(resources_by_role, dict):
        raise _invalid(f"missing object for role {str(role)!r}")

    validated: dict[str, list[str]] = {}
    for resource, actions in resources_by_role.items():
        if not isinstance(resource, str) or not resource:
            raise _invalid(f"role {str(role)!r} contains an invalid resource name")
        if (
            not isinstance(actions, list)
            or not actions
            or not all(isinstance(action, str) and action for action in actions)
        ):
            raise _invalid(
                f"actions for role {str(role)!r}, resource {resource!r} "
                "must be a non-empty string list"
            )
        validated[resource] = actions
    return validated


def _load() -> dict[str, list[str]]:
    try:
        text = _ACCESS_RESOURCE.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(
            "Packaged permission resource access.json is missing or unreadable"
        ) from exc
    except UnicodeError as exc:
        raise _invalid("content must be UTF-8") from exc
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise _invalid(f"malformed JSON ({exc.msg})") from exc

    loaded: dict[str, list[str]] = {}
    for role in _ROLES:
        resources = _resources_for_role(raw, role)
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
