# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""API route handlers for global search."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from gsf.server.responses import (
    GlobalSearchCountResponse,
    GlobalSearchListResponse,
)
from gsf.server.search import service as search_service
from gsf.server.search.constants import TEXT_MATCH_CONTAINS

router = APIRouter()


class GlobalSearchFilters(BaseModel):
    """What a query is matched against, and which kinds of object it may hit.

    The two text flags widen the match beyond an object's name: ``description``
    adds its description, ``synonyms`` adds a Term's aliases.

    Their defaults differ, which is deliberate rather than an oversight.
    ``description`` is off because matching prose is the narrower, more
    surprising behaviour to opt into. ``synonyms`` is on because alias matching
    is what this endpoint did for its whole life before the flag existed, and
    defaulting it off would quietly take Terms away from every caller that does
    not send the flag.
    """

    description: bool = False
    synonyms: bool = True
    objects: list[str] | None = None


class GlobalSearchRequest(BaseModel):
    search_term: str = Field(min_length=0)
    text_match_option: str = TEXT_MATCH_CONTAINS
    filters: GlobalSearchFilters = Field(default_factory=GlobalSearchFilters)


def _run(
    payload: GlobalSearchRequest,
    *,
    count: bool,
) -> dict:
    try:
        kwargs = {
            "search_term": payload.search_term,
            "text_match_option": payload.text_match_option,
            "objects": payload.filters.objects,
            "include_description": payload.filters.description,
            "include_synonyms": payload.filters.synonyms,
        }
        if count:
            return search_service.global_search_count(**kwargs)
        return search_service.global_search(**kwargs)
    except search_service.SearchValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/search/global-search", response_model=GlobalSearchListResponse)
def post_global_search(payload: GlobalSearchRequest) -> dict:
    """Fulltext global search across catalog and semantic nodes.

    Requires at least two characters after trimming. ``contains`` is the only
    match option. Results are capped at 200 and ranked so names that contain
    every token sort first, then Terms an alias matched.

    ``filters.synonyms`` turns that alias matching off, which drops the Terms
    reached only by one and empties the ``synonyms`` reported on every hit --
    with it off, no alias caused a hit, so there is none to name.
    """
    return _run(payload, count=False)


@router.post("/search/global-search/count", response_model=GlobalSearchCountResponse)
def post_global_search_count(payload: GlobalSearchRequest) -> dict:
    """Hit counts by type for the same query as ``/search/global-search``."""
    return _run(payload, count=True)
