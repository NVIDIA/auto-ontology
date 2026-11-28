# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``describe_table`` accepts a table name, not only a catalog id."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest
from fastmcp import Client, FastMCP
from fastmcp.exceptions import ToolError

from auto_ontology_mcp import tables
from auto_ontology_mcp.config import DEFAULT_SPEC_PATH, Settings
from auto_ontology_mcp.tables import DETAILS_PATH, SEARCH_PATH

_ORDERS_ID = "08bfb564-bbca-4ab8-bd80-e8deb61b757a"
_ORDER_ITEMS_ID = "766fa956-99e0-4a42-9112-f2e775d483f1"

_DETAILS = {
    "data": {
        "queries": [{"id": "q1", "name": "revenue by channel"}],
        "terms": [{"id": "t1", "name": "Orders", "relationship_types": ["REPRESENTS"]}],
        "terms_total": 1,
    }
}


def _settings() -> Settings:
    return Settings(
        api_url="http://auto_ontology.test",
        spec_path=DEFAULT_SPEC_PATH,
        host="127.0.0.1",
        port=3003,
        timeout_s=180.0,
        chat_timeout_s=900.0,
    )


def _client(
    hits: list[dict[str, Any]] | None = None,
    *,
    search_status: int = 200,
    details: Any = _DETAILS,
    seen: list[str] | None = None,
) -> httpx.AsyncClient:
    """Answer search and details from *hits*, recording the paths called."""

    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request.url.path)
        if request.url.path == SEARCH_PATH:
            if search_status != 200:
                return httpx.Response(search_status, json={"detail": "nope"})
            body = hits or []
            return httpx.Response(200, json={"data": body, "count": len(body)})
        return httpx.Response(200, json=details)

    return httpx.AsyncClient(
        base_url="http://auto_ontology.test", transport=httpx.MockTransport(handler)
    )


def _describe(client: httpx.AsyncClient, table: str) -> Any:
    settings = _settings()
    mcp: FastMCP = FastMCP(name="test")
    tables.register(mcp, settings, client)

    async def run() -> Any:
        try:
            async with Client(mcp) as session:
                return (
                    await session.call_tool("describe_table", {"table": table})
                ).data
        finally:
            await client.aclose()

    return asyncio.run(run())


def test_a_table_name_resolves_and_is_described() -> None:
    seen: list[str] = []
    client = _client([{"type": "Table", "name": "orders", "id": _ORDERS_ID}], seen=seen)

    result = _describe(client, "orders")

    assert seen == [SEARCH_PATH, DETAILS_PATH.format(table_id=_ORDERS_ID)]
    assert result.table == "orders"
    assert result.table_id == _ORDERS_ID
    assert result.terms_total == 1
    assert result.terms[0]["name"] == "Orders"


def test_an_exact_match_wins_over_a_substring() -> None:
    # Searching "orders" also hits "order_items"; describing the wrong table is
    # worse than describing none.
    client = _client(
        [
            {"type": "Table", "name": "order_items", "id": _ORDER_ITEMS_ID},
            {"type": "Table", "name": "orders", "id": _ORDERS_ID},
        ]
    )

    assert _describe(client, "orders").table_id == _ORDERS_ID


def test_an_ambiguous_name_lists_the_candidates() -> None:
    client = _client(
        [
            {"type": "Table", "name": "orders_2024", "id": _ORDERS_ID},
            {"type": "Table", "name": "orders_2025", "id": _ORDER_ITEMS_ID},
        ]
    )

    with pytest.raises(ToolError) as excinfo:
        _describe(client, "orders")

    assert "matches 2 tables" in str(excinfo.value)


def test_an_unknown_table_says_where_to_look() -> None:
    with pytest.raises(ToolError) as excinfo:
        _describe(_client([]), "nothing_like_this")

    assert "No table matches" in str(excinfo.value)


def test_a_table_id_skips_the_search() -> None:
    seen: list[str] = []
    client = _client([], seen=seen)

    result = _describe(client, _ORDERS_ID)

    # An id is already an answer; searching for it would be a wasted read.
    assert seen == [DETAILS_PATH.format(table_id=_ORDERS_ID)]
    assert result.table_id == _ORDERS_ID
    assert result.table is None


def test_a_rejected_session_is_reported_as_such() -> None:
    # A stale token otherwise reads as a search miss, which sends the caller
    # looking for a table that is right there.
    with pytest.raises(ToolError) as excinfo:
        _describe(_client([], search_status=401), "orders")

    assert "did not accept this session" in str(excinfo.value)


def test_details_without_terms_still_describe_the_table() -> None:
    client = _client(
        [{"type": "Table", "name": "staging_tmp", "id": _ORDERS_ID}],
        details={"data": {}},
    )

    result = _describe(client, "staging_tmp")

    assert result.terms == []
    assert result.queries == []
    assert result.terms_total == 0
