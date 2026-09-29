# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verifying a token Auto Ontology issued, and telling rejection apart from an outage."""

from __future__ import annotations

import asyncio
from collections.abc import Callable

import httpx
import pytest
from mcp.server.auth.provider import AccessToken

from auto_ontology_mcp.auto_ontology_auth import (
    CLIENT_ID_CLAIM,
    SCOPE_CLAIM,
    USERINFO_PATH,
    AutoOntologyTokenVerifier,
)

API_URL = "https://auto_ontology.example"
TOKEN = "opaque-token"

Handler = Callable[[httpx.Request], httpx.Response]


def _verify(handler: Handler, token: str = TOKEN) -> AccessToken | None:
    verifier = AutoOntologyTokenVerifier(
        API_URL, timeout_s=5.0, transport=httpx.MockTransport(handler)
    )
    return asyncio.run(verifier.verify_token(token))


def test_a_live_grant_identifies_its_owner() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == USERINFO_PATH
        assert request.headers["Authorization"] == f"Bearer {TOKEN}"
        return httpx.Response(
            200,
            json={
                "sub": "user-1",
                CLIENT_ID_CLAIM: "client-9",
                SCOPE_CLAIM: "openid profile email",
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
    # UserInfo rejects an expired or unknown bearer as invalid_token.
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "invalid_token"})

    assert _verify(handler) is None


def test_a_body_that_is_not_json_is_refused_not_a_crash() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>signed out</html>")

    assert _verify(handler) is None


def test_a_grant_without_an_owner_is_refused() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={CLIENT_ID_CLAIM: "client-9"})

    assert _verify(handler) is None


def test_a_grant_carrying_no_scopes_still_verifies() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"sub": "user-1", CLIENT_ID_CLAIM: "c"})

    access = _verify(handler)

    assert access is not None
    assert access.scopes == []


def test_an_unreachable_auto_ontology_is_not_reported_as_a_bad_token() -> None:
    # Answering "sign in again" during an outage would send everyone through a
    # sign-in that cannot succeed either.
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="upstream down")

    with pytest.raises(httpx.HTTPStatusError):
        _verify(handler)
