# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``search_terms`` — find the glossary terms a business word refers to.

Hand-written rather than generated. ``GET /api/terms?query=`` is the obvious
route, but it filters on the term *name* only: it backs the Terms page's
list filter, and a name filter is what that page wants. An agent asking what
"revenue" means here needs the term whose description or synonyms say so, and
a name filter returns nothing for it — which reads as "not in this data".

``POST /api/search/global-search`` already matches name, description, and
whole-word synonyms, ranked in that order. Generating a tool from it would hand
the agent its whole filter surface (tables, columns, tags, databases) and leave
it to remember to narrow to Terms, so this pins the request to Terms with
descriptions on and exposes only the query.
"""

from __future__ import annotations

from typing import Any

import httpx
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from pydantic import BaseModel, Field

from auto_ontology_mcp.config import Settings

SEARCH_PATH = "/api/search/global-search"

# The server caps a search at 200 hits, already ranked best-first. An agent
# needs the top of that list, not all of it in its context.
MAX_TERMS = 25


class TermMatch(BaseModel):
    """One glossary term that matched."""

    id: str = Field(
        description="Pass to get_term, get_term_columns, or get_term_sql_attributes."
    )
    name: str | None = None
    description: str | None = None
    synonyms: list[str] = Field(
        default_factory=list,
        description="The synonyms that matched the query, when one is why it matched.",
    )
    certified: bool | str | None = Field(
        default=None, description="Whether a person has reviewed this term."
    )


class TermSearch(BaseModel):
    """Structured result of one ``search_terms`` call."""

    terms: list[TermMatch] = Field(
        default_factory=list,
        description=f"Best matches first, at most {MAX_TERMS} of them.",
    )
    total: int = Field(default=0, description="Terms that matched before truncation.")
    truncated: bool = Field(
        default=False, description=f"True when more than {MAX_TERMS} terms matched."
    )


def _build_result(payload: Any) -> TermSearch:
    hits = payload.get("data") if isinstance(payload, dict) else None
    terms = [TermMatch.model_validate(hit) for hit in hits or []]
    return TermSearch(
        terms=terms[:MAX_TERMS],
        total=len(terms),
        truncated=len(terms) > MAX_TERMS,
    )


def register(mcp: FastMCP, settings: Settings, client: httpx.AsyncClient) -> None:
    """Attach ``search_terms`` to *mcp*."""

    @mcp.tool(
        name="search_terms",
        description=(
            "Search the business glossary. Terms are the human-meaningful "
            "concepts Auto Ontology has compiled over the connected databases — "
            '"Customer", "Open Order", "Net Revenue" — each mapped to real '
            "columns and SQL expressions. A term matches when every word of the "
            "query appears in its name or in its description, or as whole words "
            "in one of its synonyms; name matches rank first. Use this first to "
            "learn what a business word actually means in this deployment "
            "before asking a question that depends on it.\n\n"
            "It matches text, not meaning: an empty result means no term uses "
            "those words, not that the data cannot answer. Try a synonym, or "
            "use check_answerable, which matches by meaning."
        ),
    )
    async def search_terms(query: str) -> TermSearch:
        """Return the glossary terms whose text matches *query*.

        Args:
            query: A business word or phrase, at least two characters.
        """
        body = {
            "search_term": query,
            "filters": {"objects": ["Term"], "description": True},
        }
        try:
            response = await client.post(SEARCH_PATH, json=body)
        except httpx.HTTPError as exc:
            raise ToolError(
                f"Could not reach Auto Ontology at {settings.api_url}: {exc}"
            ) from exc

        if response.status_code != 200:
            raise ToolError(
                f"Auto Ontology could not search the glossary "
                f"(HTTP {response.status_code}): {response.text.strip()[:300]}"
            )
        return _build_result(response.json())


__all__ = ["MAX_TERMS", "SEARCH_PATH", "TermMatch", "TermSearch", "register"]
