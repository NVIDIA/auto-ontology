# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import httpx
import pytest
from pytest import MonkeyPatch

from auto_ontology.connectors import databricks_oauth
from auto_ontology.connectors.databricks_oauth import (
    DatabricksOAuthError,
    exchange_subject_token,
)


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    databricks_oauth.clear_cache()
    # `any_connection_uses_sso_federation` memoises for a TTL, so without this a
    # result cached by one test decides the answer in the next.
    databricks_oauth.invalidate_sso_federation_cache()


def _response(status: int, payload: dict[str, Any]) -> httpx.Response:
    return httpx.Response(
        status_code=status,
        json=payload,
        request=httpx.Request("POST", "https://example.databricks.com/oidc/v1/token"),
    )


def test_exchange_posts_token_exchange_grant(monkeypatch: MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        captured["url"] = url
        captured["data"] = kwargs["data"]
        return _response(200, {"access_token": "dbx-token", "expires_in": 3600})

    monkeypatch.setattr(httpx, "post", fake_post)

    token = exchange_subject_token("example.databricks.com", "sso-jwt")

    assert token == "dbx-token"
    assert captured["url"] == "https://example.databricks.com/oidc/v1/token"
    assert captured["data"]["grant_type"] == (
        "urn:ietf:params:oauth:grant-type:token-exchange"
    )
    assert captured["data"]["subject_token"] == "sso-jwt"
    assert captured["data"]["subject_token_type"] == (
        "urn:ietf:params:oauth:token-type:jwt"
    )


def test_exchange_strips_scheme_from_host(monkeypatch: MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        captured["url"] = url
        return _response(200, {"access_token": "t", "expires_in": 3600})

    monkeypatch.setattr(httpx, "post", fake_post)
    exchange_subject_token("https://example.databricks.com/", "jwt")

    assert captured["url"] == "https://example.databricks.com/oidc/v1/token"


def test_exchange_caches_per_subject_token(monkeypatch: MonkeyPatch) -> None:
    calls = {"n": 0}

    def fake_post(_url: str, **_kwargs: Any) -> httpx.Response:
        calls["n"] += 1
        return _response(200, {"access_token": f"t{calls['n']}", "expires_in": 3600})

    monkeypatch.setattr(httpx, "post", fake_post)

    assert exchange_subject_token("host.databricks.com", "jwt-a") == "t1"
    assert exchange_subject_token("host.databricks.com", "jwt-a") == "t1"
    assert calls["n"] == 1

    # A different caller must not receive the first caller's token.
    assert exchange_subject_token("host.databricks.com", "jwt-b") == "t2"
    assert calls["n"] == 2


def test_expired_cache_entry_is_refetched(monkeypatch: MonkeyPatch) -> None:
    calls = {"n": 0}

    def fake_post(_url: str, **_kwargs: Any) -> httpx.Response:
        calls["n"] += 1
        # expires_in below the safety margin => cached with ttl 0.
        return _response(200, {"access_token": f"t{calls['n']}", "expires_in": 1})

    monkeypatch.setattr(httpx, "post", fake_post)

    assert exchange_subject_token("host.databricks.com", "jwt") == "t1"
    assert exchange_subject_token("host.databricks.com", "jwt") == "t2"
    assert calls["n"] == 2


def test_rejected_exchange_raises(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(
        httpx,
        "post",
        lambda *_a, **_k: _response(401, {"error": "invalid_request"}),
    )

    with pytest.raises(DatabricksOAuthError) as excinfo:
        exchange_subject_token("host.databricks.com", "jwt")

    assert "401" in str(excinfo.value)
    assert "invalid_request" in str(excinfo.value)


def test_error_message_does_not_leak_subject_token(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(
        httpx,
        "post",
        lambda *_a, **_k: _response(
            400, {"error": "bad", "error_description": "token super-secret-jwt bad"}
        ),
    )

    with pytest.raises(DatabricksOAuthError) as excinfo:
        exchange_subject_token("host.databricks.com", "super-secret-jwt")

    assert "super-secret-jwt" not in str(excinfo.value)


def test_missing_subject_token_raises() -> None:
    with pytest.raises(DatabricksOAuthError):
        exchange_subject_token("host.databricks.com", "")


def test_uses_sso_federation_accepts_bool_and_string_forms() -> None:
    from auto_ontology.connectors.databricks_oauth import uses_sso_federation

    assert uses_sso_federation({"sso_federation": True}) is True
    assert uses_sso_federation({"sso_federation": "true"}) is True
    assert uses_sso_federation({"sso_federation": "1"}) is True

    assert uses_sso_federation({"sso_federation": False}) is False
    assert uses_sso_federation({"sso_federation": "false"}) is False
    assert uses_sso_federation({"sso_federation": ""}) is False
    assert uses_sso_federation({}) is False


def test_any_connection_uses_sso_federation(monkeypatch: MonkeyPatch) -> None:
    import auto_ontology.dal.connections as dal

    monkeypatch.setattr(dal, "list_connections", lambda: [{"type": "databricks"}])
    assert databricks_oauth.any_connection_uses_sso_federation() is False

    monkeypatch.setattr(
        dal,
        "list_connections",
        lambda: [
            {"type": "postgresql"},
            {"type": "databricks", "sso_federation": True},
        ],
    )
    # The answer is cached for a TTL, so changing the stored connections only
    # takes effect after an invalidation — which is exactly what
    # `service.set_sso_federation` does when the flag is toggled.
    databricks_oauth.invalidate_sso_federation_cache()
    assert databricks_oauth.any_connection_uses_sso_federation() is True


def test_any_connection_fails_soft_when_lookup_errors(monkeypatch: MonkeyPatch) -> None:
    """A storage blip degrades to the stored PAT rather than breaking chat."""
    import auto_ontology.dal.connections as dal

    def boom() -> None:
        raise RuntimeError("store down")

    monkeypatch.setattr(dal, "list_connections", boom)
    assert databricks_oauth.any_connection_uses_sso_federation() is False


def test_exchange_requests_all_apis_scope(monkeypatch: MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_post(_url: str, **kwargs: Any) -> httpx.Response:
        captured["data"] = kwargs["data"]
        return _response(200, {"access_token": "t", "expires_in": 3600})

    monkeypatch.setattr(httpx, "post", fake_post)
    exchange_subject_token("host.databricks.com", "jwt")

    assert captured["data"]["scope"] == "all-apis"


def test_connection_string_prefers_exchanged_token() -> None:
    from auto_ontology.connectors.connection_string_factory import (
        build_connection_string,
    )

    connection_string = build_connection_string(
        {
            "type": "databricks",
            "host": "example.databricks.com",
            "http_path": "/sql/1.0/warehouses/w",
            "password": "stored-pat",
            "database": "main",
            "access_token_override": "exchanged-token",
        }
    )

    assert "exchanged-token" in connection_string
    assert "stored-pat" not in connection_string


def test_set_sso_federation_updates_only_the_flag(monkeypatch: MonkeyPatch) -> None:
    """Toggling must not require (or disturb) the stored credentials."""
    import json

    from auto_ontology.server.connections import service

    stored = {
        "type": "databricks",
        "host": "h.databricks.com",
        "http_path": "/sql/1.0/warehouses/w",
        "password": "STORED-PAT",
        "database": "main",
        "schemas": ["sales"],
    }
    captured: dict[str, Any] = {}

    monkeypatch.setattr(service, "list_connections", lambda: [stored])
    monkeypatch.setattr(service, "is_vault_configured", lambda: False)
    monkeypatch.setattr(
        service, "insert_connection", lambda **kwargs: captured.update(kwargs)
    )
    monkeypatch.setattr(service, "invalidate_connectors_cache", lambda: None)
    monkeypatch.setattr(service, "refresh_chat_workers", lambda: None)

    result = service.set_sso_federation(database_name="main", enabled=True)

    assert result == {"database_name": "main", "sso_federation": True}
    written = json.loads(captured["connection"])
    assert written["sso_federation"] is True
    # Everything else survives untouched.
    assert written["password"] == "STORED-PAT"
    assert written["schemas"] == ["sales"]


def test_set_sso_federation_unknown_database_raises(monkeypatch: MonkeyPatch) -> None:
    from auto_ontology.server.connections import service

    monkeypatch.setattr(service, "list_connections", lambda: [])

    with pytest.raises(ValueError, match="No connection found"):
        service.set_sso_federation(database_name="nope", enabled=True)


def test_rotated_subject_token_does_not_strand_the_old_entry(
    monkeypatch: MonkeyPatch,
) -> None:
    """The cache is keyed by a hash of the subject token, so a rotation lands on a new
    key. Without pruning, the superseded entry would live for the process's lifetime and
    the cache would grow with every rotation."""

    def fake_post(_url: str, **_kwargs: Any) -> httpx.Response:
        # Already expired once the margin is subtracted, so the next write prunes it.
        return _response(200, {"access_token": "db-token", "expires_in": 0})

    monkeypatch.setattr(httpx, "post", fake_post)

    databricks_oauth.exchange_subject_token("example.databricks.com", "sso-token-v1")
    assert len(databricks_oauth._cache) == 1

    # The same caller comes back with a rotated SSO token.
    databricks_oauth.exchange_subject_token("example.databricks.com", "sso-token-v2")

    assert len(databricks_oauth._cache) == 1, (
        "the superseded entry should have been purged, not accumulated"
    )


def test_live_entries_survive_a_purge(monkeypatch: MonkeyPatch) -> None:
    """Pruning must only drop expired entries — evicting live ones would force a
    needless exchange on every query."""

    def fake_post(_url: str, **_kwargs: Any) -> httpx.Response:
        return _response(200, {"access_token": "db-token", "expires_in": 3600})

    monkeypatch.setattr(httpx, "post", fake_post)

    databricks_oauth.exchange_subject_token("example.databricks.com", "caller-a")
    databricks_oauth.exchange_subject_token("example.databricks.com", "caller-b")

    # Two distinct, unexpired callers both remain cached.
    assert len(databricks_oauth._cache) == 2


def test_purge_expired_removes_only_lapsed_tokens() -> None:
    databricks_oauth.clear_cache()
    databricks_oauth._cache[("host", "live")] = databricks_oauth._CachedToken(
        "t1", expires_at=1_000.0
    )
    databricks_oauth._cache[("host", "lapsed")] = databricks_oauth._CachedToken(
        "t2", expires_at=10.0
    )

    databricks_oauth._purge_expired(now=100.0)

    assert set(databricks_oauth._cache) == {("host", "live")}
