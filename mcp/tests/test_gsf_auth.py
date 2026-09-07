# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verifying a token GSF issued, and telling rejection apart from an outage."""

from __future__ import annotations

import asyncio
from collections.abc import Callable

import httpx
import pytest
from mcp.server.auth.provider import AccessToken

from gsf_mcp.gsf_auth import SESSION_PATH, GsfTokenVerifier

API_URL = "https://gsf.example"
TOKEN = "opaque-token"

Handler = Callable[[httpx.Request], httpx.Response]


def _verify(handler: Handler, token: str = TOKEN) -> AccessToken | None:
    verifier = GsfTokenVerifier(
        API_URL, timeout_s=5.0, transport=httpx.MockTransport(handler)
    )
    return asyncio.run(verifier.verify_token(token))


def test_a_live_grant_identifies_its_owner() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == SESSION_PATH
        assert request.headers["Authorization"] == f"Bearer {TOKEN}"
        return httpx.Response(
            200,
            json={
                "userId": "user-1",
                "clientId": "client-9",
                "scopes": "openid profile email",
            },
        )

    access = _verify(handler)

    assert access is not None
    assert access.subject == "user-1"
    assert access.client_id == "client-9"
    assert access.scopes == ["openid", "profile", "email"]
    # Forwarded upstream verbatim, so it has to survive verification unchanged.
    assert access.token == TOKEN


def test_an_unknown_token_is_refused_not_an_error() -> None:
    # Better Auth answers an expired or unknown token with a literal null body
    # and a 200, so the status alone does not settle it.
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=b"null", headers={"content-type": "application/json"}
        )

    assert _verify(handler) is None


def test_a_body_that_is_not_json_is_refused_not_a_crash() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>signed out</html>")

    assert _verify(handler) is None


def test_a_grant_without_an_owner_is_refused() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"clientId": "client-9"})

    assert _verify(handler) is None


def test_a_grant_carrying_no_scopes_still_verifies() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"userId": "user-1", "clientId": "c"})

    access = _verify(handler)

    assert access is not None
    assert access.scopes == []


def test_an_unreachable_gsf_is_not_reported_as_a_bad_token() -> None:
    # Answering "sign in again" during an outage would send everyone through a
    # sign-in that cannot succeed either.
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="upstream down")

    with pytest.raises(httpx.HTTPStatusError):
        _verify(handler)
