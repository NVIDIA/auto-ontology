# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Zone member checks on create and update."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import HTTPException

from auto_ontology.server import owners
from auto_ontology.server.zones import router
from auto_ontology.server.zones.router import (
    ZoneCreate,
    ZoneUpdate,
    create_zone,
    update_zone,
)


def _directory(monkeypatch: pytest.MonkeyPatch) -> None:
    accounts = {"admin-1": "admin", "viewer-1": "viewer"}
    groups = {"admins": "admin", "viewers": "viewer"}
    monkeypatch.setattr(
        owners.groups_dal,
        "get_account",
        lambda user_id: {"role": accounts[user_id]} if user_id in accounts else None,
    )
    monkeypatch.setattr(
        owners.groups_dal,
        "get_group",
        lambda group_id: {"role": groups[group_id]} if group_id in groups else None,
    )


def test_an_update_rejects_a_member_who_cannot_manage_zones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _directory(monkeypatch)
    monkeypatch.setattr(
        router.dal,
        "update_zone",
        lambda **_kwargs: pytest.fail("invalid members must not be written"),
    )

    with pytest.raises(HTTPException) as caught:
        update_zone("zone-1", ZoneUpdate(members=["viewer-1", "admin-1"]))

    assert caught.value.status_code == 400


def test_renaming_a_zone_leaves_members_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    written: dict[str, Any] = {}

    def fake_update(**kwargs: Any) -> dict[str, Any]:
        written.update(kwargs)
        return {"id": kwargs["zone_id"]}

    monkeypatch.setattr(router.dal, "update_zone", fake_update)

    update_zone("zone-1", ZoneUpdate(name="Finance"))

    assert written["member_ids"] is None


def test_create_rejects_a_member_who_cannot_manage_zones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _directory(monkeypatch)
    monkeypatch.setattr(router.postgres_dal, "is_admin", lambda _user_id: True)
    monkeypatch.setattr(
        router.dal,
        "create_zone",
        lambda **_kwargs: pytest.fail("invalid members must not be written"),
    )

    with pytest.raises(HTTPException) as caught:
        create_zone(
            ZoneCreate(
                name="Finance",
                color="#112233",
                items=[],
                created_by="admin-1",
                members=["viewer-1"],
            )
        )

    assert caught.value.status_code == 400
