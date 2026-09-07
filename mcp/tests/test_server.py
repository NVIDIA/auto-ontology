# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The assembled server publishes the curated surface and nothing else."""

from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError
from fastmcp.server.http import set_http_request
from starlette.requests import Request

from gsf_mcp import server
from gsf_mcp.config import DEFAULT_SPEC_PATH, ConfigError, Settings
from gsf_mcp.server import (
    ICON_PATH,
    INSTRUCTIONS,
    CallerAuth,
    build_client,
    build_server,
    load_icons,
    load_spec,
)
from gsf_mcp.tools import CURATED
from gsf_mcp import get_version


def _settings(spec_path: Path = DEFAULT_SPEC_PATH) -> Settings:
    return Settings(
        api_url="http://gsf.test",
        spec_path=spec_path,
        host="127.0.0.1",
        port=3003,
        timeout_s=30.0,
        chat_timeout_s=900.0,
        public_url="http://127.0.0.1:3003",
    )


def _tools() -> list[Any]:
    mcp, client = build_server(_settings())

    async def run() -> list[Any]:
        try:
            async with Client(mcp) as session:
                return await session.list_tools()
        finally:
            await client.aclose()

    return asyncio.run(run())


def test_publishes_exactly_the_curated_tools_plus_the_handwritten_ones() -> None:
    names = {tool.name for tool in _tools()}

    assert names == {spec.name for spec in CURATED} | {
        "ask_question",
        "check_readiness",
    }


def test_every_tool_carries_a_description() -> None:
    # An undescribed tool is close to unusable: selection is driven almost
    # entirely by this text.
    undescribed = [
        tool.name for tool in _tools() if not (tool.description or "").strip()
    ]

    assert undescribed == []


def test_generated_descriptions_are_replaced_end_to_end() -> None:
    # test_tools.py checks the override function in isolation; this checks it
    # was actually handed to FastMCP and ran over the real spec.
    leaked = [tool.name for tool in _tools() if "Api." in (tool.description or "")]

    assert leaked == []


def test_ask_question_reports_a_structured_answer() -> None:
    ask = next(tool for tool in _tools() if tool.name == "ask_question")

    assert ask.outputSchema is not None
    assert set(ask.outputSchema["properties"]) >= {"answer", "sql", "rows"}


def test_handshake_advertises_the_gsf_version() -> None:
    # Not FastMCP's, which is what gets reported if the version is left unset
    # and reads as a GSF version in a client's server list.
    mcp, client = build_server(_settings())

    async def run() -> Any:
        try:
            async with Client(mcp) as session:
                return session.initialize_result.serverInfo
        finally:
            await client.aclose()

    info = asyncio.run(run())

    assert info.name == "gsf"
    assert info.version == get_version()


def test_handshake_advertises_the_icon() -> None:
    mcp, client = build_server(_settings())

    async def run() -> Any:
        try:
            async with Client(mcp) as session:
                return session.initialize_result.serverInfo
        finally:
            await client.aclose()

    icons = asyncio.run(run()).icons or []

    assert len(icons) == 1
    assert icons[0].mimeType == "image/svg+xml"
    # Inline, so a client can draw it without reaching the network or GSF.
    assert icons[0].src.startswith("data:image/svg+xml;base64,")


def test_the_icon_decodes_back_to_the_svg() -> None:
    src = load_icons()[0].src
    payload = base64.b64decode(src.removeprefix("data:image/svg+xml;base64,"))

    assert payload == ICON_PATH.read_bytes()
    assert payload.startswith(b"<svg")


def test_a_missing_icon_does_not_stop_the_server(monkeypatch: Any) -> None:
    # Cosmetic, so it must never be the reason a working server fails to start.
    monkeypatch.setattr(server, "ICON_PATH", Path("/nonexistent/nvidia-mark.svg"))

    assert load_icons() == []


def test_instructions_point_at_the_primary_tool() -> None:
    # Sequencing guidance lives here rather than in any one tool description,
    # so a host that shows only this still knows where to start.
    assert "ask_question" in INSTRUCTIONS
    assert "check_answerable" in INSTRUCTIONS
    assert "check_readiness" in INSTRUCTIONS


def _sent_headers(
    auth: CallerAuth, incoming: dict[str, str] | None = None
) -> httpx.Headers:
    """Run one request through *auth* and return the headers it went out with.

    The incoming request is faked with FastMCP's own HTTP request context, so
    this exercises the same lookup a live HTTP transport would.
    """
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={})

    async def run() -> None:
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            auth=auth,
            base_url="http://gsf.test",
        )
        try:
            await client.get("/api/terms")
        finally:
            await client.aclose()

    if incoming is None:
        asyncio.run(run())
    else:
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/mcp",
            "headers": [(k.lower().encode(), v.encode()) for k, v in incoming.items()],
        }
        with set_http_request(Request(scope)):
            asyncio.run(run())

    return captured[0].headers


def test_no_credential_is_attached_to_the_client_itself() -> None:
    """A shared client must not carry one caller's identity for all callers."""
    client = build_client(_settings())
    try:
        assert "x-api-key" not in client.headers
    finally:
        asyncio.run(client.aclose())


def test_the_signed_in_callers_token_is_forwarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GSF issued it, so it goes upstream untouched — no exchange, no unwrapping."""
    monkeypatch.setattr(server, "gsf_access_token", lambda: "gsf-issued")

    headers = _sent_headers(CallerAuth())

    assert headers["authorization"] == "Bearer gsf-issued"


def test_a_smuggled_api_key_cannot_override_the_signed_in_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FastMCP copies the caller's headers onto the outbound request, and GSF
    resolves x-api-key ahead of a bearer token. Left in place, a caller could
    sign in as one user and then call GSF as the owner of some other token.
    """
    monkeypatch.setattr(server, "gsf_access_token", lambda: "gsf-issued")

    headers = _sent_headers(CallerAuth(), {"x-api-key": "gsf_someone_else"})

    assert headers["authorization"] == "Bearer gsf-issued"
    assert "x-api-key" not in headers


def test_a_smuggled_bearer_token_cannot_override_it_either(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server, "gsf_access_token", lambda: "gsf-issued")

    headers = _sent_headers(CallerAuth(), {"authorization": "Bearer someone-else"})

    assert headers["authorization"] == "Bearer gsf-issued"


def test_a_request_with_no_signed_in_session_is_an_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server, "gsf_access_token", lambda: None)

    with pytest.raises(ToolError, match="Sign in again"):
        _sent_headers(CallerAuth())


def test_rejects_a_spec_that_is_not_json(tmp_path: Path) -> None:
    bad = tmp_path / "spec.json"
    bad.write_text("{ not json", encoding="utf-8")

    with pytest.raises(ConfigError, match="not valid JSON"):
        load_spec(_settings(bad))


def test_rejects_a_spec_with_no_paths(tmp_path: Path) -> None:
    empty = tmp_path / "spec.json"
    empty.write_text(json.dumps({"openapi": "3.1.0"}), encoding="utf-8")

    with pytest.raises(ConfigError, match="no paths"):
        load_spec(_settings(empty))


def test_refuses_to_start_when_a_curated_endpoint_disappeared(tmp_path: Path) -> None:
    # Starting anyway would produce a healthy-looking server missing a tool.
    spec = json.loads(DEFAULT_SPEC_PATH.read_text(encoding="utf-8"))
    del spec["paths"]["/api/terms"]
    trimmed = tmp_path / "spec.json"
    trimmed.write_text(json.dumps(spec), encoding="utf-8")

    with pytest.raises(ConfigError, match="GET /api/terms"):
        load_spec(_settings(trimmed))
