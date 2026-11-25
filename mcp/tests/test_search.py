# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``search_terms`` searches Terms by name, description, and synonym."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Callable

import httpx
import pytest
from fastmcp import Client, FastMCP
from fastmcp.exceptions import ToolError

from auto_ontology_mcp import search
from auto_ontology_mcp.config import DEFAULT_SPEC_PATH, Settings
from auto_ontology_mcp.search import MAX_TERMS, SEARCH_PATH, TermSearch


def _settings() -> Settings:
    return Settings(
        api_url="http://auto_ontology.test",
        spec_path=DEFAULT_SPEC_PATH,
        host="127.0.0.1",
        port=3003,
        timeout_s=30.0,
        chat_timeout_s=900.0,
    )


def _hit(index: int, **extra: Any) -> dict[str, Any]:
    return {"id": f"t{index}", "name": f"Term {index}", "type": "Term", **extra}


def _search(
    handler: Callable[[httpx.Request], httpx.Response], query: str = "revenue"
) -> TermSearch:
    settings = _settings()
    client = httpx.AsyncClient(
        base_url=settings.api_url, transport=httpx.MockTransport(handler)
    )
    mcp: FastMCP = FastMCP(name="test")
    search.register(mcp, settings, client)

    async def run() -> TermSearch:
        try:
            async with Client(mcp) as session:
                result = await session.call_tool("search_terms", {"query": query})
                return TermSearch.model_validate(result.structured_content)
        finally:
            await client.aclose()

    return asyncio.run(run())


def test_searches_terms_by_description_and_synonym_not_only_name() -> None:
    """The regression: a name-only filter returned nothing for 'revenue'."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        hit = _hit(1, description="Invoice line; extended price is revenue.")
        return httpx.Response(200, json={"data": [hit], "count": 1})

    result = _search(handler)

    assert seen[0].method == "POST"
    assert seen[0].url.path == SEARCH_PATH
    assert json.loads(seen[0].content) == {
        "search_term": "revenue",
        "filters": {"objects": ["Term"], "description": True},
    }
    assert [term.id for term in result.terms] == ["t1"]
    assert result.terms[0].description == "Invoice line; extended price is revenue."
    assert result.total == 1
    assert result.truncated is False


def test_a_long_result_is_cut_to_the_best_matches_and_says_so() -> None:
    hits = [_hit(i) for i in range(MAX_TERMS + 5)]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": hits, "count": len(hits)})

    result = _search(handler)

    assert [term.id for term in result.terms] == [f"t{i}" for i in range(MAX_TERMS)]
    assert result.total == MAX_TERMS + 5
    assert result.truncated is True


def test_a_refused_search_is_an_actionable_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"detail": "Forbidden"})

    with pytest.raises(ToolError, match="HTTP 403"):
        _search(handler)
