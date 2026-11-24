# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``describe_table`` — one table's glossary terms, by name or by id.

Hand-written rather than generated, because the spec's parameter is a
``table_id`` and nothing else on this surface hands one out. Every table tool an
agent can reach starts from a term, whose column attributes carry the id; the
generated tool therefore only worked for a caller that had already walked term →
column attributes → ``table_id`` — or that could read the catalog database,
which is exactly what this server exists to avoid.

Resolving a name first makes the tool reachable from how a question actually
arrives ("what does orders mean"), while an id still works for callers that
have one. The lookup is a global search narrowed to tables, so it costs one
extra read and never lists the physical catalog.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

import httpx
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from pydantic import BaseModel, Field

from auto_ontology_mcp.config import Settings

logger = logging.getLogger(__name__)

SEARCH_PATH = "/api/search/global-search"
DETAILS_PATH = "/api/exploration/tables/{table_id}/details"
TABLE_OBJECT = "Table"


class TableDescription(BaseModel):
    """What one table means: the terms it answers to, and its stored queries."""

    table: str | None = Field(
        default=None,
        description=(
            "The table's name. Absent when the caller passed an id, so nothing "
            "was needed from the search index to find it."
        ),
    )
    table_id: str = Field(description="The catalog id, reusable in later calls.")
    terms: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Glossary terms mapped to the table, with the relationship that "
            "maps them (HAS_ATTRIBUTE, REPRESENTS, SEMANTIC_FK)."
        ),
    )
    queries: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Stored queries from query history that reference the table.",
    )
    terms_total: int = Field(default=0, description="How many terms are listed.")


def _as_uuid(value: str) -> str | None:
    """Return *value* as a canonical id, or ``None`` when it is not one."""
    try:
        return str(uuid.UUID(value))
    except ValueError:
        return None


async def _resolve(client: httpx.AsyncClient, table: str) -> tuple[str, str | None]:
    """Return ``(table_id, table_name)`` for a name or an id.

    An exact name match wins over a substring match, so ``orders`` does not
    resolve to ``order_items`` merely because both are hits. Two equally good
    matches are an error rather than a guess: describing the wrong table is
    confident misinformation, and the candidates are cheap to hand back.
    """
    if (table_id := _as_uuid(table)) is not None:
        return table_id, None

    response = await client.post(
        SEARCH_PATH,
        json={"search_term": table, "filters": {"objects": [TABLE_OBJECT]}},
    )
    if response.status_code == 401:
        raise ToolError(
            "The deployment did not accept this session. Sign in again through "
            "the Auto Ontology deployment this server is pointed at."
        )
    response.raise_for_status()
    hits = [
        hit
        for hit in (response.json().get("data") or [])
        if str(hit.get("type") or "").casefold() == TABLE_OBJECT.casefold()
    ]

    exact = [
        hit for hit in hits if str(hit.get("name") or "").casefold() == table.casefold()
    ]
    matches = exact or hits
    if not matches:
        raise ToolError(
            f"No table matches {table!r}. Search terms (search_terms) or a term's "
            "column attributes are the usual ways to find one."
        )
    if len(matches) > 1:
        candidates = ", ".join(
            f"{hit.get('name')} ({hit.get('id')})" for hit in matches[:10]
        )
        raise ToolError(f"{table!r} matches {len(matches)} tables: {candidates}")
    return str(matches[0]["id"]), str(matches[0].get("name") or table)


def register(mcp: FastMCP, settings: Settings, client: httpx.AsyncClient) -> None:
    """Attach ``describe_table`` to *mcp*."""

    @mcp.tool(
        name="describe_table",
        description=(
            "Describe what one table means: the glossary terms mapped to it and "
            "the stored queries that reference it. Takes the table's name, or "
            "its catalog id when a previous call returned one. Use it to check "
            "whether a table is already modelled before asking a question about "
            "it, and to see which terms a question should be phrased in."
        ),
    )
    async def describe_table(table: str) -> TableDescription:
        """Describe the table named (or identified) by ``table``."""
        table_id, name = await _resolve(client, table)
        response = await client.get(DETAILS_PATH.format(table_id=table_id))
        if response.status_code == 401:
            raise ToolError(
                "The deployment did not accept this session. Sign in again "
                "through the Auto Ontology deployment this server is pointed at."
            )
        response.raise_for_status()
        details = (response.json() or {}).get("data") or {}

        return TableDescription(
            table=name,
            table_id=table_id,
            terms=details.get("terms") or [],
            queries=details.get("queries") or [],
            terms_total=int(details.get("terms_total") or 0),
        )


__all__ = [
    "DETAILS_PATH",
    "SEARCH_PATH",
    "TABLE_OBJECT",
    "TableDescription",
    "register",
]
