# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``check_readiness`` reports what the deployment can actually do."""

from __future__ import annotations

import asyncio
from typing import Any, Callable

import httpx
import pytest
from fastmcp import Client, FastMCP
from fastmcp.exceptions import ToolError

from auto_ontology_mcp import readiness
from auto_ontology_mcp.config import DEFAULT_SPEC_PATH, Settings
from auto_ontology_mcp.readiness import (
    CONNECTIONS_PATH,
    DATABASES_PATH,
    STATUS_PATH,
    Readiness,
)

# The shape that broke a real run: the glossary is compiled and the catalog
# still lists the database it was ingested from, but the connection behind it is
# gone, so nothing can execute.
_COMPILED = {"calculated": True}
_NO_CONNECTIONS: dict[str, Any] = {"data": [], "count": 0}
_ONE_CONNECTION = {"data": [{"database_name": "dw"}], "count": 1}
_ONE_DATABASE = {"data": [{"id": "d1", "name": "dw"}], "count": 1}


def _settings() -> Settings:
    return Settings(
        api_url="http://auto_ontology.test",
        spec_path=DEFAULT_SPEC_PATH,
        host="127.0.0.1",
        port=3003,
        timeout_s=30.0,
        chat_timeout_s=900.0,
    )


def _routed(**bodies: Any) -> Callable[[httpx.Request], httpx.Response]:
    """Answer each probe from *bodies*, keyed by path, defaulting to 200 JSON.

    A value may be an ``int`` status to fail that one probe, or ``None`` to make
    the request raise as a transport error would.
    """
    routes = {
        STATUS_PATH: bodies.get("status", _COMPILED),
        CONNECTIONS_PATH: bodies.get("connections", _ONE_CONNECTION),
        DATABASES_PATH: bodies.get("databases", _ONE_DATABASE),
    }

    def handler(request: httpx.Request) -> httpx.Response:
        body = routes[request.url.path]
        if body is None:
            raise httpx.ConnectError("boom", request=request)
        if isinstance(body, int):
            return httpx.Response(body, json={"detail": "nope"})
        return httpx.Response(200, json=body)

    return handler


def _check(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    database_name: str | None = None,
) -> Readiness:
    settings = _settings()
    client = httpx.AsyncClient(
        base_url=settings.api_url, transport=httpx.MockTransport(handler)
    )
    mcp: FastMCP = FastMCP(name="test")
    readiness.register(mcp, settings, client)

    async def run() -> Readiness:
        try:
            async with Client(mcp) as session:
                arguments = (
                    {} if database_name is None else {"database_name": database_name}
                )
                return (await session.call_tool("check_readiness", arguments)).data
        finally:
            await client.aclose()

    return asyncio.run(run())


def test_a_configured_deployment_is_ready() -> None:
    result = _check(_routed())

    assert result.ready is True
    assert result.semantic_layer_built is True
    assert result.can_execute_sql is True
    assert result.catalog_present is True
    assert result.databases == ["dw"]
    assert result.blockers == []
    assert not hasattr(result, "semantic_searchable")


def test_a_compiled_layer_with_no_connection_is_not_ready() -> None:
    """The silent failure this tool exists for: everything reads, nothing runs."""
    result = _check(_routed(connections=_NO_CONNECTIONS))

    assert result.ready is False
    assert result.semantic_layer_built is True
    assert result.can_execute_sql is False
    # The catalog outlives the connection, so naming the database is not proof
    # that a question can be answered.
    assert result.databases == ["dw"]
    assert any("No database connection" in blocker for blocker in result.blockers)


def test_an_uncompiled_layer_says_to_compile_it() -> None:
    result = _check(_routed(status={"calculated": False}))

    assert result.ready is False
    assert result.semantic_layer_built is False
    assert any("has not been compiled" in blocker for blocker in result.blockers)


def test_both_problems_are_reported_together() -> None:
    # One call should be enough to learn everything that is wrong, rather than
    # sending the caller round the loop once per blocker.
    result = _check(_routed(status={"calculated": False}, connections=_NO_CONNECTIONS))

    assert len(result.blockers) == 2


def test_an_unreadable_probe_degrades_instead_of_failing() -> None:
    result = _check(_routed(status=503))

    assert result.ready is False
    assert result.semantic_layer_built is False
    assert any("Could not read" in blocker for blocker in result.blockers)


def test_a_transport_error_degrades_too() -> None:
    result = _check(_routed(connections=None))

    assert result.ready is False
    assert any(
        "Could not read the configured connections" in blocker
        for blocker in result.blockers
    )


def test_an_empty_catalog_is_not_ready() -> None:
    """A connection can be saved and compiled over before its ingest lands."""
    result = _check(_routed(databases={"data": [], "count": 0}))

    assert result.ready is False
    assert result.catalog_present is False
    assert result.databases == []
    assert any("No database has been ingested" in b for b in result.blockers)


def test_an_unreadable_catalog_degrades_instead_of_passing() -> None:
    # Silence here used to read as "fine": an unreadable catalog left the
    # verdict ready, because only two of the three probes could block.
    result = _check(_routed(databases=503))

    assert result.ready is False
    assert result.catalog_present is False
    assert any("Could not read the catalog" in b for b in result.blockers)


def test_a_rejected_credential_is_an_error_not_a_verdict() -> None:
    """Reporting "not ready" would send the caller off fixing the wrong thing."""
    with pytest.raises(ToolError, match="rejected the credentials"):
        _check(_routed(status=401))


def test_connections_a_caller_may_not_read_do_not_make_it_unready() -> None:
    """Reading connections is admin-only on some deployments.

    A viewer who cannot see them can still ask questions, so the fact goes in
    unverified and the verdict stands on what could be checked.
    """
    result = _check(_routed(connections=403))

    assert result.ready is True
    assert result.can_execute_sql is None
    assert result.blockers == []
    assert any("not permitted to read" in note for note in result.unverified)


def test_a_catalog_the_caller_may_not_read_does_not_make_it_unready() -> None:
    result = _check(_routed(databases=403))

    assert result.ready is True
    assert result.catalog_present is None
    assert result.blockers == []
    assert any(DATABASES_PATH in note for note in result.unverified)


def test_a_caller_without_chat_access_is_not_ready() -> None:
    """Unlike the other two, this permission is the one being asked about."""
    result = _check(_routed(status=403))

    assert result.ready is False
    assert result.semantic_layer_built is None
    assert any("not permitted to use chat" in b for b in result.blockers)


@pytest.mark.parametrize("compiled", [True, False])
def test_readiness_reports_search_and_compilation_for_one_database(
    compiled: bool,
) -> None:
    name = "Sales & Stock"

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == STATUS_PATH
        assert dict(request.url.params) == {"database_name": name}
        return httpx.Response(
            200,
            json={
                "calculated": compiled,
                "database": {
                    "name": name,
                    "ingested": True,
                    "connected": True,
                    "compiled": compiled,
                    "searchable": compiled,
                },
            },
        )

    result = _check(handler, database_name=name)
    assert result.database_name == name
    assert result.ready is compiled
    assert result.databases == [name]
    assert result.catalog_present is True
    assert result.can_execute_sql is True
    assert result.semantic_layer_built is compiled
    assert result.semantic_searchable is compiled
    assert ("Compile its semantic layer." in result.blockers) is not compiled
    assert bool(result.blockers) is not compiled


@pytest.mark.parametrize("status", [_COMPILED, 403, 503])
def test_scoped_readiness_does_not_fall_back_to_a_global_verdict(status: Any) -> None:
    result = _check(_routed(status=status), database_name="stock")
    assert result.ready is False
    assert result.semantic_searchable is None
    assert result.blockers == ["Could not verify readiness for database 'stock'."]
