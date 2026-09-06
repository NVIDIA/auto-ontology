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
    description: bool = False
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
    every token sort first.
    """
    return _run(payload, count=False)


@router.post("/search/global-search/count", response_model=GlobalSearchCountResponse)
def post_global_search_count(payload: GlobalSearchRequest) -> dict:
    """Hit counts by type for the same query as ``/search/global-search``."""
    return _run(payload, count=True)
