# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Who may edit a row through an account role or its owner group."""

from auto_ontology.dal.owners import CATALOG_EDIT, EditSubject
from auto_ontology.server import owners
from auto_ontology.server.owners import can_edit_entity, decide_edit, normalize_owner_id

#: A permission a viewer does not hold, so the tests can tell a granted
#: permission from a role that simply lacks it.
_WITHOUT = "zone:manage"


def test_admin_can_edit() -> None:
    assert decide_edit(role="admin", permission=CATALOG_EDIT)
    assert decide_edit(role="admin", permission=_WITHOUT)


def test_a_viewer_cannot_edit_the_catalog() -> None:
    assert not decide_edit(role="viewer", permission=CATALOG_EDIT)
    assert not decide_edit(role="viewer", permission=_WITHOUT)


def test_a_missing_role_cannot_edit() -> None:
    assert not decide_edit(role=None, permission=CATALOG_EDIT)


def test_viewer_can_edit_row_owned_by_their_admin_group(monkeypatch) -> None:
    monkeypatch.setattr(owners.postgres_dal, "get_user_role", lambda _user_id: "viewer")
    monkeypatch.setattr(
        owners.groups_dal,
        "get_group",
        lambda _group_id: {
            "role": "admin",
            "users": [{"id": "viewer-1"}],
        },
    )

    assert can_edit_entity(
        user_id="viewer-1",
        subject=EditSubject(permission=CATALOG_EDIT, owner_id="group-1"),
    )


def test_viewer_cannot_use_an_admin_group_that_does_not_own_the_row(
    monkeypatch,
) -> None:
    monkeypatch.setattr(owners.postgres_dal, "get_user_role", lambda _user_id: "viewer")
    monkeypatch.setattr(owners.groups_dal, "get_group", lambda _owner_id: None)

    assert not can_edit_entity(
        user_id="viewer-1",
        subject=EditSubject(permission=CATALOG_EDIT, owner_id="another-owner"),
    )


def test_owner_group_must_grant_the_required_permission(monkeypatch) -> None:
    monkeypatch.setattr(owners.postgres_dal, "get_user_role", lambda _user_id: "viewer")
    monkeypatch.setattr(
        owners.groups_dal,
        "get_group",
        lambda _group_id: {
            "role": "viewer",
            "users": [{"id": "viewer-1"}],
        },
    )

    assert not can_edit_entity(
        user_id="viewer-1",
        subject=EditSubject(permission=CATALOG_EDIT, owner_id="group-1"),
    )


def test_blank_owner_id_means_unassigned() -> None:
    assert normalize_owner_id("  ") is None
    assert normalize_owner_id(" user-1 ") == "user-1"
    assert normalize_owner_id(None) is None
