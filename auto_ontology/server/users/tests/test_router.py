# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Group writes require an admin. The role on the group becomes permissions."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from auto_ontology.dal.groups import GroupNameTaken, UnknownGroupMembers
from auto_ontology.server.users import router as users_router
from auto_ontology.server import owners
from auto_ontology.server.users.router import (
    GroupCreate,
    GroupUpdate,
    create_group,
    delete_group,
    get_group,
    list_groups,
    list_owners,
    release_user_ownership,
    update_group,
)

_HEADER = b"x-auto-ontology-user-id"


def _request(user_id: str | None) -> Request:
    headers = []
    if user_id is not None:
        headers.append((_HEADER, user_id.encode()))
    return Request({"type": "http", "headers": headers})


def _stored(**overrides: Any) -> dict[str, Any]:
    group = {
        "id": "group-1",
        "name": "Finance",
        "description": "",
        "color": "#0ea5e9",
        "role": "viewer",
        "users": [],
        "created": "2026-10-11T00:00:00+00:00",
    }
    group.update(overrides)
    return group


@pytest.fixture
def admin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(users_router.postgres_dal, "is_admin", lambda user_id: True)


def test_create_without_a_caller_is_refused() -> None:
    with pytest.raises(HTTPException) as refused:
        create_group(
            _request(None),
            GroupCreate(name="Finance", color="#0ea5e9", role="viewer"),
        )
    assert refused.value.status_code == 401


def test_a_viewer_cannot_create_a_group(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(users_router.postgres_dal, "is_admin", lambda user_id: False)
    with pytest.raises(HTTPException) as refused:
        create_group(
            _request("user-1"),
            GroupCreate(name="Finance", color="#0ea5e9", role="viewer"),
        )
    assert refused.value.status_code == 403


def test_owner_list_names_only_the_caller_in_a_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        users_router.dal,
        "list_users_and_groups",
        lambda: {
            "users": [
                {
                    "id": "viewer-1",
                    "name": "Ada",
                    "email": "ada@nvidia.com",
                    "role": "viewer",
                },
                {
                    "id": "viewer-2",
                    "name": "",
                    "email": "bea@nvidia.com",
                    "role": "viewer",
                },
            ],
            "groups": [
                {
                    "id": "group-1",
                    "name": "Developers",
                    "color": "#0ea5e9",
                    "role": "admin",
                    "users": [{"id": "viewer-1"}, {"id": "viewer-2"}],
                }
            ],
        },
    )

    owners = {
        owner["id"]: owner["member_ids"]
        for owner in list_owners(_request("viewer-1"))["data"]
    }

    assert owners["viewer-1"] == []
    assert owners["viewer-2"] == []
    assert owners["group-1"] == ["viewer-1"]
    assert list_owners(_request(None))["data"][-1]["member_ids"] == []


def test_group_reads_without_a_caller_are_refused() -> None:
    with pytest.raises(HTTPException) as list_refused:
        list_groups(_request(None))
    assert list_refused.value.status_code == 401

    with pytest.raises(HTTPException) as get_refused:
        get_group(_request(None), "group-1")
    assert get_refused.value.status_code == 401


def test_a_viewer_cannot_read_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(users_router.postgres_dal, "is_admin", lambda user_id: False)

    with pytest.raises(HTTPException) as list_refused:
        list_groups(_request("viewer-1"))
    assert list_refused.value.status_code == 403

    with pytest.raises(HTTPException) as get_refused:
        get_group(_request("viewer-1"), "group-1")
    assert get_refused.value.status_code == 403


def test_an_admin_create_expands_the_role(
    admin: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        users_router.dal,
        "create_group",
        lambda **kwargs: _stored(role=kwargs["role"], name=kwargs["name"]),
    )

    response = create_group(
        _request("admin-1"),
        GroupCreate(name=" Finance ", color=" #0EA5E9 ", role="admin"),
    )

    group = response["data"]
    assert group["name"] == "Finance"
    assert group["color"] == "#0ea5e9"
    assert group["roles"] == [{"id": "admin", "name": "admin"}]
    assert "zone:manage" in group["permissions"]


def test_a_taken_name_is_a_conflict(
    admin: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _taken(**kwargs: Any) -> dict[str, Any]:
        raise GroupNameTaken(kwargs["name"])

    monkeypatch.setattr(users_router.dal, "create_group", _taken)
    with pytest.raises(HTTPException) as refused:
        create_group(
            _request("admin-1"),
            GroupCreate(name="Finance", color="#0ea5e9", role="viewer"),
        )
    assert refused.value.status_code == 409


def test_an_unknown_member_is_rejected(
    admin: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _unknown(**kwargs: Any) -> dict[str, Any]:
        raise UnknownGroupMembers(kwargs["user_ids"])

    monkeypatch.setattr(users_router.dal, "create_group", _unknown)
    with pytest.raises(HTTPException) as refused:
        create_group(
            _request("admin-1"),
            GroupCreate(
                name="Finance",
                color="#0ea5e9",
                role="viewer",
                user_ids=["missing"],
            ),
        )
    assert refused.value.status_code == 400


def test_a_blank_role_is_rejected(admin: None) -> None:
    with pytest.raises(HTTPException) as refused:
        create_group(
            _request("admin-1"),
            GroupCreate(name="Finance", color="#0ea5e9", role="owner"),
        )
    assert refused.value.status_code == 400


def test_an_unknown_color_is_rejected(admin: None) -> None:
    with pytest.raises(HTTPException) as refused:
        create_group(
            _request("admin-1"),
            GroupCreate(name="Finance", color="not-a-color", role="viewer"),
        )
    assert refused.value.status_code == 400
    assert "Group color must be one of" in refused.value.detail


def test_replacing_members_is_passed_through(
    admin: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}

    def _update(**kwargs: Any) -> dict[str, Any]:
        seen.update(kwargs)
        return _stored()

    monkeypatch.setattr(users_router.dal, "update_group", _update)
    update_group(
        _request("admin-1"),
        "group-1",
        GroupUpdate(user_ids=["user-2", "user-2", " user-3 "]),
    )
    assert seen["user_ids"] == ["user-2", "user-3"]
    assert seen["updates"] == {}


def test_delete_requires_an_admin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(users_router.postgres_dal, "is_admin", lambda user_id: False)
    with pytest.raises(HTTPException) as refused:
        delete_group(_request("user-1"), "group-1")
    assert refused.value.status_code == 403


def test_demoting_a_user_releases_objects_they_can_no_longer_own(
    admin: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}
    monkeypatch.setattr(
        owners.postgres_dal, "get_user_role", lambda user_id, **_kwargs: "viewer"
    )

    def _release(account_id: str, *, catalog: bool, zone: bool) -> None:
        seen["account_id"] = account_id
        seen["catalog"] = catalog
        seen["zone"] = zone

    monkeypatch.setattr(owners.owners_dal, "release_account", _release)

    assert release_user_ownership("alice", _request("admin-1")) == {
        "data": {"id": "alice"}
    }
    assert seen == {"account_id": "alice", "catalog": True, "zone": True}


def test_an_admin_role_keeps_ownership(
    admin: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}
    monkeypatch.setattr(
        owners.postgres_dal, "get_user_role", lambda user_id, **_kwargs: "admin"
    )

    def _release(account_id: str, *, catalog: bool, zone: bool) -> None:
        seen["catalog"] = catalog
        seen["zone"] = zone

    monkeypatch.setattr(owners.owners_dal, "release_account", _release)

    release_user_ownership("alice", _request("admin-1"))

    assert seen == {"catalog": False, "zone": False}


def test_a_role_lookup_failure_is_not_a_missing_user(
    admin: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _lookup(user_id: str, *, strict: bool = False) -> str | None:
        if strict:
            raise RuntimeError("database unavailable")
        return None

    monkeypatch.setattr(owners.postgres_dal, "get_user_role", _lookup)
    monkeypatch.setattr(
        owners.owners_dal,
        "release_account",
        lambda *_args, **_kwargs: pytest.fail("an outage must not clear rows"),
    )

    with pytest.raises(RuntimeError, match="database unavailable"):
        release_user_ownership("alice", _request("admin-1"))


def test_releasing_a_missing_user_does_not_clear_rows(
    admin: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        owners.postgres_dal, "get_user_role", lambda user_id, **_kwargs: None
    )
    monkeypatch.setattr(
        owners.owners_dal,
        "release_account",
        lambda *_args, **_kwargs: pytest.fail("a missing account must not be cleared"),
    )

    with pytest.raises(HTTPException) as refused:
        release_user_ownership("missing", _request("admin-1"))

    assert refused.value.status_code == 404


def test_releasing_ownership_requires_an_admin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(users_router.postgres_dal, "is_admin", lambda user_id: False)

    with pytest.raises(HTTPException) as refused:
        release_user_ownership("alice", _request("viewer-1"))

    assert refused.value.status_code == 403
