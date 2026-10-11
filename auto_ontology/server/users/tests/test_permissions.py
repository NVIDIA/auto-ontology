# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A group's permissions are its role, expanded from the shared access file.

``frontend/auth/access.json`` is also what the Next.js auth layer imports, so
these tests only have to pin the differences that matter: an admin group can
manage groups and zones, a viewer group cannot, and a viewer group's
permissions are a subset of an admin's.
"""

from __future__ import annotations

from auto_ontology.server.users.permissions import permissions_for_role


def test_an_admin_group_can_manage_groups() -> None:
    granted = permissions_for_role("admin")
    assert "group:read" in granted
    assert "group:manage" in granted
    assert "group:read" not in permissions_for_role("viewer")


def test_an_admin_group_can_manage_zones() -> None:
    assert "zone:manage" in permissions_for_role("admin")
    assert "zone:read" in permissions_for_role("admin")


def test_a_viewer_group_can_read_zones_but_not_manage_them() -> None:
    granted = permissions_for_role("viewer")
    assert "zone:read" in granted
    assert "catalog:edit" in granted
    assert "zone:manage" not in granted
    assert "connection:manage" not in granted


def test_viewer_permissions_are_a_subset_of_admin() -> None:
    assert set(permissions_for_role("viewer")) <= set(permissions_for_role("admin"))


def test_an_unknown_role_grants_nothing() -> None:
    assert permissions_for_role("owner") == []


def test_builtin_account_permissions_are_not_on_a_group() -> None:
    granted = set(permissions_for_role("admin"))
    assert not any(
        item.startswith("user:") or item.startswith("session:") for item in granted
    )
