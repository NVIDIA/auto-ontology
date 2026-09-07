# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sign-in, where GSF itself is the authorization server.

This server could instead sit between the client and the deployment's identity
provider. That works, but it has to be told a client id, a client secret, and a
redirect URI that someone registered with that provider first — a ticket for
every deployment, and a secret to hold and rotate. GSF already knows how to sign
a person in, including through SSO, so it does, and this server carries no
credentials at all.

What that buys, concretely:

* Nothing to configure beyond pointing at GSF. Clients register themselves
  through dynamic client registration, which GSF supports and most corporate
  providers do not, so no redirect URI is ever registered by hand.
* No secret on disk here, and no secret to rotate.
* The registrations and grants live in GSF's database, so replicas share them
  and a restart does not sign anyone out.
* Revocation is GSF's: deleting the grant, banning the user, or changing their
  role takes effect on the next call.

The token a caller presents is one GSF minted, so verifying it is a question
only GSF can answer — hence the round trip below. There is no local signature to
check: these tokens are opaque by design, which also means they leak nothing if
logged. The same token is then forwarded upstream unchanged, where
``frontend/auth/resolve-user.ts`` resolves it to its owner.
"""

from __future__ import annotations

import httpx
from fastmcp.server.auth.auth import AccessToken, RemoteAuthProvider, TokenVerifier
from fastmcp.utilities.logging import get_logger
from pydantic import AnyHttpUrl

from gsf_mcp.config import Settings

logger = get_logger(__name__)

# Better Auth's `mcp` plugin exposes this; it looks a grant up by its access
# token and refuses an expired one. The `userinfo_endpoint` its own discovery
# document advertises is not implemented, so this is the only way to ask.
SESSION_PATH = "/api/auth/mcp/get-session"


class GsfTokenVerifier(TokenVerifier):
    """Validates a token by asking GSF whether it still stands.

    A rejected token returns ``None`` so the caller is told to sign in again.
    A GSF that cannot be reached raises instead: an outage is not an expired
    token, and answering "unauthenticated" would send everyone through a
    sign-in that cannot succeed either.
    """

    def __init__(
        self,
        api_url: str,
        timeout_s: float,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        super().__init__()
        self._api_url = api_url
        self._timeout_s = timeout_s
        # A short-lived client of its own rather than the one the tools share:
        # that one rewrites the Authorization header to the caller's credential,
        # which is the very thing being checked here. `transport` is a seam for
        # tests.
        self._transport = transport

    async def verify_token(self, token: str) -> AccessToken | None:
        async with httpx.AsyncClient(
            base_url=self._api_url,
            timeout=self._timeout_s,
            transport=self._transport,
        ) as client:
            response = await client.get(
                SESSION_PATH, headers={"Authorization": f"Bearer {token}"}
            )

        if response.status_code >= 500:
            response.raise_for_status()

        # An unknown or expired token is answered with a literal `null` body and
        # a 200, not an error status. Anything unparseable is treated the same
        # way: a body we cannot read is not a grant, and raising here would turn
        # a refusal into a server error.
        try:
            grant = response.json() if response.status_code == 200 else None
        except ValueError:
            grant = None

        if not isinstance(grant, dict) or not grant.get("userId"):
            logger.debug("GSF rejected an access token")
            return None

        # Space-separated per RFC 6749, and absent when the grant carries none.
        scopes = str(grant.get("scopes") or "").split()

        return AccessToken(
            token=token,
            # The client that was granted the token, which FastMCP uses to
            # attribute the session. Registration guarantees it.
            client_id=str(grant.get("clientId") or "unknown"),
            scopes=scopes,
            subject=str(grant["userId"]),
        )


def build_gsf_auth(settings: Settings) -> RemoteAuthProvider:
    """Advertise GSF as the authorization server for this MCP server.

    Publishing GSF's URL here is the whole handshake: a client that gets a 401
    reads this server's protected-resource metadata, finds GSF, discovers GSF's
    endpoints, registers itself, and sends the user to GSF's ordinary login
    page — which already knows how to do SSO.
    """
    return RemoteAuthProvider(
        token_verifier=GsfTokenVerifier(settings.api_url, timeout_s=settings.timeout_s),
        authorization_servers=[AnyHttpUrl(settings.api_url)],
        base_url=settings.public_url,
        resource_name="GSF",
    )


def gsf_access_token() -> str | None:
    """The token this request was authenticated with, to forward to GSF.

    Read back from FastMCP rather than off the incoming header so that what is
    forwarded is the token that actually passed verification.
    """

    # Imported here because it resolves request state: at module scope it would
    # bind at import time, outside any request.
    from fastmcp.server.dependencies import get_access_token

    try:
        token = get_access_token()
    except Exception:
        # Raised when there is no authenticated request in scope, which is a
        # normal state rather than a fault.
        return None

    return getattr(token, "token", None) or None


__all__ = [
    "GsfTokenVerifier",
    "SESSION_PATH",
    "build_gsf_auth",
    "gsf_access_token",
]
