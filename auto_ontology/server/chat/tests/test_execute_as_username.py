# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The chat endpoint's choice of ClickHouse user, and its refusals."""

from __future__ import annotations

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from auto_ontology.server.chat.router import _execute_as_username
from auto_ontology.server.identity import INTERNAL_USER_HEADER

_FLAG = "auto_ontology.connectors.execute_as.any_connection_uses_execute_as_user"
_EMAIL = "auto_ontology.server.users.postgres_dal.get_user_email"


def _request(user_id: str | None = None) -> Request:
    headers = []
    if user_id is not None:
        headers.append((INTERNAL_USER_HEADER.encode(), user_id.encode()))
    return Request({"type": "http", "headers": headers})


def test_no_flagged_connection_needs_no_user(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_FLAG, lambda: False)

    assert _execute_as_username(_request()) is None


def test_the_users_email_local_part_is_the_clickhouse_user(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_FLAG, lambda: True)
    monkeypatch.setattr(_EMAIL, lambda user_id: f"{user_id}@nvidia.com")

    assert _execute_as_username(_request("alice.smith")) == "alice.smith"


def test_a_request_without_a_user_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_FLAG, lambda: True)

    with pytest.raises(HTTPException) as caught:
        _execute_as_username(_request())

    assert caught.value.status_code == 401


def test_a_user_without_an_email_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_FLAG, lambda: True)
    monkeypatch.setattr(_EMAIL, lambda user_id: None)

    with pytest.raises(HTTPException) as caught:
        _execute_as_username(_request("u1"))

    assert caught.value.status_code == 403


def test_an_unusable_email_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_FLAG, lambda: True)
    monkeypatch.setattr(_EMAIL, lambda user_id: "a`b@nvidia.com")

    with pytest.raises(HTTPException) as caught:
        _execute_as_username(_request("u1"))

    assert caught.value.status_code == 403
