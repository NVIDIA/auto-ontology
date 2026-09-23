# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the trusted Next.js-to-FastAPI identity contract."""

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from auto_ontology.server.identity import resolve_internal_user


def _request(headers: dict[str, str]) -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/chat/completions",
            "headers": [
                (key.lower().encode("ascii"), value.encode("ascii"))
                for key, value in headers.items()
            ],
        }
    )


def test_forwarded_header_resolves_user() -> None:
    user_id = resolve_internal_user(
        _request({"x-auto-ontology-user-id": "user-1"}),
        required=True,
    )

    assert user_id == "user-1"


def test_missing_required_identity_is_rejected() -> None:
    with pytest.raises(HTTPException) as exc_info:
        resolve_internal_user(_request({}), required=True)

    assert exc_info.value.status_code == 401


def test_stateless_request_without_identity_remains_supported() -> None:
    assert resolve_internal_user(_request({}), required=False) is None
