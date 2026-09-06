# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Settings reject an unusable environment at startup, not on first request."""

from __future__ import annotations

import pytest
from pytest import MonkeyPatch

from gsf_mcp.config import (
    DEFAULT_CHAT_TIMEOUT_S,
    DEFAULT_PORT,
    DEFAULT_SPEC_PATH,
    DEFAULT_TIMEOUT_S,
    ConfigError,
    Settings,
    load_settings,
)

_VARS = (
    "GSF_API_URL",
    "GSF_API_TOKEN",
    "GSF_OPENAPI_SPEC",
    "GSF_MCP_HOST",
    "GSF_MCP_PORT",
    "GSF_MCP_TIMEOUT_S",
    "GSF_MCP_CHAT_TIMEOUT_S",
    "GSF_MCP_PUBLIC_URL",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: MonkeyPatch) -> None:
    """Ignore whatever the developer happens to have exported."""
    for name in _VARS:
        monkeypatch.delenv(name, raising=False)


def test_starts_with_nothing_configured() -> None:
    # Every caller signs in for themselves, so there is no credential to supply
    # and a bare `gsf-mcp` against a local GSF is a complete configuration.
    settings = load_settings()

    assert settings.api_url == "http://localhost:3000"
    assert settings.port == DEFAULT_PORT
    assert settings.timeout_s == DEFAULT_TIMEOUT_S
    assert settings.chat_timeout_s == DEFAULT_CHAT_TIMEOUT_S
    assert settings.spec_path == DEFAULT_SPEC_PATH


def test_holds_no_credential_of_its_own() -> None:
    # Guards the invariant rather than any one variable: a field for a
    # process-wide token is what would let one identity speak for everybody.
    assert not hasattr(load_settings(), "api_token")
    assert "token" not in " ".join(Settings.__dataclass_fields__)


def test_an_exported_api_token_is_ignored(monkeypatch: MonkeyPatch) -> None:
    # Anyone who used an earlier build may still have this exported. It must
    # neither be adopted as an identity nor refused at startup.
    monkeypatch.setenv("GSF_API_TOKEN", "gsf_abc")

    assert load_settings().api_url == "http://localhost:3000"


def test_strips_trailing_slash_from_api_url(monkeypatch: MonkeyPatch) -> None:
    # httpx joins base_url with a leading-slash path, so a trailing slash here
    # would produce '//api/...' and 404 against the Next.js router.
    monkeypatch.setenv("GSF_API_URL", "https://gsf.example.com/")

    assert load_settings().api_url == "https://gsf.example.com"


def test_rejects_missing_spec(monkeypatch: MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("GSF_OPENAPI_SPEC", str(tmp_path / "nope.json"))

    with pytest.raises(ConfigError, match="OpenAPI spec not found"):
        load_settings()


@pytest.mark.parametrize("value", ["0", "-5", "not-a-number"])
def test_rejects_nonsense_timeouts(monkeypatch: MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("GSF_MCP_TIMEOUT_S", value)

    with pytest.raises(ConfigError, match="GSF_MCP_TIMEOUT_S"):
        load_settings()


@pytest.mark.parametrize("value", ["0", "70000", "http"])
def test_rejects_invalid_port(monkeypatch: MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("GSF_MCP_PORT", value)

    with pytest.raises(ConfigError, match="GSF_MCP_PORT"):
        load_settings()


def test_blank_values_fall_back_to_defaults(monkeypatch: MonkeyPatch) -> None:
    # Unset and empty-string are the same thing to a shell export, so an empty
    # value must not be read as "0" or as a literal blank URL.
    monkeypatch.setenv("GSF_MCP_PORT", "")
    monkeypatch.setenv("GSF_MCP_TIMEOUT_S", "  ")

    settings = load_settings()

    assert settings.port == DEFAULT_PORT
    assert settings.timeout_s == DEFAULT_TIMEOUT_S


def test_public_url_is_derived_when_unset(monkeypatch: MonkeyPatch) -> None:
    # 0.0.0.0 is the default bind address and no use to a browser.
    monkeypatch.setenv("GSF_MCP_PORT", "9999")

    assert load_settings().public_url == "http://localhost:9999"


def test_public_url_follows_an_explicit_host(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("GSF_MCP_HOST", "127.0.0.1")
    monkeypatch.setenv("GSF_MCP_PORT", "3003")

    # Clients compare this against the address they dialled, and '127.0.0.1' is
    # not the string 'localhost' even though it is the same interface.
    assert load_settings().public_url == "http://127.0.0.1:3003"


def test_public_url_can_be_set_explicitly(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("GSF_MCP_PUBLIC_URL", "https://mcp.example/")

    assert load_settings().public_url == "https://mcp.example"
