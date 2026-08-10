# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exchange a caller's SSO token for a Databricks OAuth access token.

When SSO is configured, chat queries run as the *end user* rather than under the
connection's stored personal access token. The SSO JWT itself is not a Databricks
credential — it was minted for GSF's (or AI-Q's) OAuth client, so its audience is
wrong. Databricks identity federation provides the bridge: the workspace token
endpoint accepts the external JWT as a ``subject_token`` and returns a short-lived
Databricks access token carrying the user's own Unity Catalog privileges.

Per-user auth is opt-in per connection ("Authenticate as signed-in user" in
Settings → Connections). It requires a federation policy in the Databricks
account that trusts the SSO issuer. Without one the exchange returns 401 and, per the
fail-closed policy, the chat request is rejected rather than silently falling
back to the stored PAT.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Mapping

import httpx

logger = logging.getLogger(__name__)

_GRANT_TYPE = "urn:ietf:params:oauth:grant-type:token-exchange"
_SUBJECT_TOKEN_TYPE = "urn:ietf:params:oauth:token-type:jwt"

# Databricks issues workspace tokens under this scope; there is no per-deployment
# reason to vary it.
_SCOPE = "all-apis"

# Refresh a little before real expiry so a token can't lapse mid-query.
_EXPIRY_MARGIN_S = 60.0
_REQUEST_TIMEOUT_S = 15.0


class DatabricksOAuthError(RuntimeError):
    """Raised when a subject token cannot be exchanged for a Databricks token."""


@dataclass(frozen=True)
class _CachedToken:
    access_token: str
    expires_at: float


_cache: dict[tuple[str, str], _CachedToken] = {}
_cache_lock = threading.Lock()

# --- SSO-federation flag cache -----------------------------------------------
# any_connection_uses_sso_federation() queries Neo4j + Vault on every call.
# Cache the result for a short window so the check is cheap on the chat hot
# path. 30 s means a toggled connection is effective within half a minute.
_SSO_FEDERATION_TTL_S = 30.0
_sso_federation_result: bool | None = None
_sso_federation_expires: float = 0.0
_sso_federation_lock = threading.Lock()


def uses_sso_federation(connection: Mapping[str, Any]) -> bool:
    """Whether *connection* runs chat queries as the signed-in user.

    Set per connection in Settings → Connections. JSON round-trips give a real
    bool, but string forms are accepted so hand-edited connections behave.
    """
    value = connection.get("sso_federation")
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in ("1", "true", "yes")


def any_connection_uses_sso_federation() -> bool:
    """Whether any configured connection requires a caller SSO token.

    Drives the chat endpoint's fail-closed check: when nothing is federated the
    request proceeds without a token, exactly as before this feature existed.
    Lookup failures return False so a storage blip degrades to the stored access
    token rather than taking chat down.

    The result is cached for :data:`_SSO_FEDERATION_TTL_S` seconds to avoid a
    Neo4j + Vault round-trip on every chat completion request.
    """
    global _sso_federation_result, _sso_federation_expires

    now = time.monotonic()
    with _sso_federation_lock:
        if _sso_federation_result is not None and now < _sso_federation_expires:
            return _sso_federation_result

    try:
        from gsf.dal.connections import list_connections

        result = any(uses_sso_federation(conn) for conn in list_connections())
    except Exception:
        logger.exception("Failed to check connections for SSO federation")
        result = False

    with _sso_federation_lock:
        _sso_federation_result = result
        _sso_federation_expires = now + _SSO_FEDERATION_TTL_S

    return result


def invalidate_sso_federation_cache() -> None:
    """Force the next call to re-query. Call after toggling a connection's SSO setting."""
    global _sso_federation_result
    with _sso_federation_lock:
        _sso_federation_result = None


def _cache_key(host: str, subject_token: str) -> tuple[str, str]:
    """Cache per (workspace, caller) without keeping raw tokens as dict keys."""
    digest = hashlib.sha256(subject_token.encode("utf-8")).hexdigest()
    return (host, digest)


def clear_cache() -> None:
    """Drop all cached exchanged tokens (used by tests and on config change)."""
    with _cache_lock:
        _cache.clear()


def _purge_expired(now: float) -> None:
    """Drop entries whose tokens have expired. Caller must hold ``_cache_lock``.

    Entries are keyed by a hash of the *subject token*, so a caller whose SSO token
    rotates lands on a new key and strands the old one. Nothing else removes entries —
    an expired one is ignored and overwritten, never deleted — so without this the cache
    grows with (workspaces x users x token rotations) for the life of the process.

    Linear in the cache size, which is fine: exchanges happen about once per user per
    token lifetime, and pruning keeps the size proportional to *active* callers.
    """
    expired = [key for key, token in _cache.items() if token.expires_at <= now]
    for key in expired:
        del _cache[key]
    if expired:
        logger.debug("Purged %d expired Databricks token(s) from cache", len(expired))


def exchange_subject_token(host: str, subject_token: str) -> str:
    """Return a Databricks access token for *subject_token* on *host*.

    Results are cached per (workspace host, subject token) until shortly before
    expiry, so a chat turn that touches several tables performs one exchange.

    Raises:
        DatabricksOAuthError: the token endpoint rejected the exchange or
            returned a malformed payload.
    """
    host = host.strip().rstrip("/")
    if host.startswith(("https://", "http://")):
        host = host.split("://", 1)[1]
    if not host:
        raise DatabricksOAuthError("Databricks host is required for token exchange")
    if not subject_token:
        raise DatabricksOAuthError("A subject token is required for token exchange")

    key = _cache_key(host, subject_token)
    now = time.monotonic()

    with _cache_lock:
        cached = _cache.get(key)
        if cached is not None and cached.expires_at > now:
            return cached.access_token

    url = f"https://{host}/oidc/v1/token"
    try:
        response = httpx.post(
            url,
            data={
                "grant_type": _GRANT_TYPE,
                "subject_token": subject_token,
                "subject_token_type": _SUBJECT_TOKEN_TYPE,
                "scope": _SCOPE,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=_REQUEST_TIMEOUT_S,
        )
    except httpx.HTTPError as exc:
        raise DatabricksOAuthError(
            f"Databricks token exchange to {url} failed: {exc}"
        ) from exc

    if response.status_code != 200:
        # The body can echo the subject token back in an error description, so
        # only the status and Databricks' short error code are logged/raised.
        detail = ""
        try:
            detail = str(response.json().get("error", ""))
        except Exception:  # noqa: BLE001 — error body may not be JSON
            pass
        raise DatabricksOAuthError(
            f"Databricks token exchange failed with HTTP {response.status_code}"
            + (f" ({detail})" if detail else "")
        )

    try:
        payload = response.json()
        access_token = str(payload["access_token"])
        expires_in = float(payload.get("expires_in", 3600))
    except Exception as exc:  # noqa: BLE001 — malformed success body
        raise DatabricksOAuthError(
            "Databricks token exchange returned a malformed response"
        ) from exc

    if not access_token:
        raise DatabricksOAuthError("Databricks token exchange returned an empty token")

    ttl = max(expires_in - _EXPIRY_MARGIN_S, 0.0)
    # The exchange itself took real time, so measure against the clock now rather than
    # the reading taken before the request went out.
    stored_at = time.monotonic()
    with _cache_lock:
        _purge_expired(stored_at)
        _cache[key] = _CachedToken(access_token, stored_at + ttl)

    logger.info(
        "Exchanged SSO token for Databricks access token on %s (ttl %.0fs)", host, ttl
    )
    return access_token
