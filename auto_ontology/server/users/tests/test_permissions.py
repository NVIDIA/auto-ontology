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

from pathlib import Path

import pytest

from auto_ontology.server.users import permissions as permissions_module
from auto_ontology.server.users.permissions import permissions_for_role

ROOT = Path(__file__).resolve().parents[4]


def test_an_admin_group_can_manage_groups() -> None:
    granted = permissions_for_role("admin")
    assert "group:read" in granted
    assert "group:manage" in granted
    assert "group:read" not in permissions_for_role("viewer")


def test_an_admin_group_can_manage_zones() -> None:
    assert "zone:manage" in permissions_for_role("admin")
    assert "zone:read" in permissions_for_role("admin")


def test_a_viewer_group_can_read_but_not_write() -> None:
    granted = permissions_for_role("viewer")
    assert "zone:read" in granted
    assert "catalog:read" in granted
    assert "chat:use" in granted
    assert "catalog:edit" not in granted
    assert "zone:manage" not in granted
    assert "connection:manage" not in granted
    # A viewer mints tokens for their own account. The token carries this role,
    # so it cannot edit the catalog.
    assert "apiToken:manage" in granted


def test_viewer_permissions_are_a_subset_of_admin() -> None:
    assert set(permissions_for_role("viewer")) <= set(permissions_for_role("admin"))


def test_an_unknown_role_grants_nothing() -> None:
    assert permissions_for_role("owner") == []


def test_builtin_account_permissions_are_not_on_a_group() -> None:
    granted = set(permissions_for_role("admin"))
    assert not any(
        item.startswith("user:") or item.startswith("session:") for item in granted
    )


def test_packaged_resource_matches_the_frontend_source() -> None:
    source = ROOT / "frontend" / "auth" / "access.json"
    packaged = ROOT / "auto_ontology" / "server" / "users" / "access.json"
    assert packaged.read_bytes() == source.read_bytes(), (
        "Regenerate the packaged permission resource with: "
        "uv run python -m dev_tools.generate_permission_resource"
    )


def test_a_missing_packaged_resource_has_a_clear_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        permissions_module, "_ACCESS_RESOURCE", tmp_path / "missing.json"
    )
    with pytest.raises(RuntimeError, match="missing or unreadable"):
        permissions_module._load()


def test_a_malformed_packaged_resource_has_a_clear_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    malformed = tmp_path / "access.json"
    malformed.write_text('{"admin": ', encoding="utf-8")
    monkeypatch.setattr(permissions_module, "_ACCESS_RESOURCE", malformed)
    with pytest.raises(RuntimeError, match="malformed JSON"):
        permissions_module._load()


def test_a_missing_role_has_a_clear_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    incomplete = tmp_path / "access.json"
    incomplete.write_text('{"admin": {}}', encoding="utf-8")
    monkeypatch.setattr(permissions_module, "_ACCESS_RESOURCE", incomplete)
    with pytest.raises(RuntimeError, match="missing object for role 'viewer'"):
        permissions_module._load()
