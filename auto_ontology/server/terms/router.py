# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""API route handlers for semantic Terms."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Path, Query
from pydantic import BaseModel, Field

from auto_ontology.dal import sql_attributes as sql_attr_dal
from auto_ontology.dal import terms as terms_dal
from auto_ontology.server.pagination import LIMIT_QUERY, SKIP_QUERY
from auto_ontology.server.terms import service as term_service
from auto_ontology.server.responses import (
    ColumnAttributePageResponse,
    ColumnAttributePatchResponse,
    SqlAttributePageResponse,
    TermDetailResponse,
    TermResponse,
    TermsPageResponse,
)

router = APIRouter()
logger = logging.getLogger(__name__)


class TermUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1)
    description: str | None = None
    name_certified: bool | None = None
    description_certified: bool | None = None


class ColumnAttributeUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1)
    description: str | None = None
    sample_values: list[str] | None = None
    certified: bool | None = None


@router.get("/terms", response_model=TermsPageResponse)
def list_terms(
    query: str | None = Query(
        default=None,
        description="Case-insensitive substring filter on the term name.",
    ),
    skip: int = SKIP_QUERY,
    limit: int | None = LIMIT_QUERY,
) -> dict:
    """Return Term nodes.

    *query*, when given, filters to terms whose name contains it
    (case-insensitive).

    Terms come back ordered by name, and *skip*/*limit* select one page of
    that order; ``total`` reports how many match in full, so a caller knows
    when to stop asking. Omitting *limit* returns every matching term.

    Each returned term carries its resolved ``zones``, so the Terms list
    and the Exploration graph can render Zone chips from this single
    response without a separate per-page zones request.

    ``column_attribute_counts``, ``sql_attribute_counts`` and
    ``related_counts`` are per-term breakdowns (``[{term_id, count}, ...]``)
    that let the Terms list render per-card badges without an N+1 fetch, and
    cover the terms on this page (see ``service.list_terms_page``). The
    underlying ColumnAttribute/SqlAttribute nodes are fetched separately, only
    for the focused Term, via ``/terms/{term_id}/column-attributes`` and
    ``/terms/{term_id}/sql-attributes`` below — this endpoint stays
    counts-only so the Terms list never pulls attribute nodes it won't show.
    """
    return term_service.list_terms_page(
        zone_ids=None, search=query, skip=skip, limit=limit
    )


@router.get(
    "/terms/{term_id}/column-attributes", response_model=ColumnAttributePageResponse
)
def list_term_column_attributes_by_id(
    term_id: str,
    skip: int = SKIP_QUERY,
    limit: int | None = LIMIT_QUERY,
) -> dict:
    """Return ColumnAttribute nodes for a single Term.

    Attributes come back ordered by name, and *skip*/*limit* select one page
    of that order. ``count`` is the length of ``data`` (this page), ``total``
    the number of attributes the term has in full.

    Each attribute includes ``primary_column`` and ``referenced_columns``
    (catalog path ids + names) for navigation from the detail page.
    """
    attrs = terms_dal.fetch_column_attributes_by_term_id(
        term_id, zone_ids=None, skip=skip, limit=limit
    )
    # `len(attrs)` only equals the term's full count when neither paging
    # argument was given — a `skip` alone (no `limit`) still returns a
    # partial read, so it needs the same separate count query.
    total = (
        terms_dal.count_column_attributes_by_term_id(term_id, zone_ids=None)
        if skip or limit is not None
        else len(attrs)
    )
    return {"data": attrs, "count": len(attrs), "total": total}


@router.patch(
    "/terms/{term_id}/column-attributes/{attr_id}",
    response_model=ColumnAttributePatchResponse,
)
def update_column_attribute(
    term_id: str,
    body: ColumnAttributeUpdate,
    attr_id: str = Path(description="ColumnAttribute id, not a SqlAttribute id."),
) -> dict:
    """Update ColumnAttribute name/description/sample_values and refresh embeddings.

    ``sample_values`` are persisted on the owning Column and also refresh that
    Column's data-VDB embedding (same behaviour as catalog column edit).
    """
    patch = body.model_dump(exclude_unset=True)
    if not patch:
        raise HTTPException(
            status_code=422, detail="No ColumnAttribute fields to update"
        )

    name = patch.get("name")
    if isinstance(name, str):
        name = name.strip()
    if "name" in patch and not name:
        raise HTTPException(
            status_code=422, detail="Column attribute name cannot be blank"
        )

    try:
        row = term_service.update_column_attribute(
            term_id,
            attr_id,
            name=name if isinstance(name, str) else None,
            description=patch.get("description"),
            sample_values=patch.get("sample_values"),
            certified=patch.get("certified"),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if row is None:
        raise HTTPException(status_code=404, detail="ColumnAttribute not found")

    return {
        "data": {
            "id": row["id"],
            "name": row["name"],
            "description": row.get("description"),
            "sample_values": row.get("sample_values"),
            "certified": row.get("certified", False),
        },
        # The owning Term's aggregate badge depends on this attribute's flag,
        # so hand back the recomputed value rather than making the client
        # re-derive it (see terms_dal.get_term_certification).
        "term_certification": terms_dal.get_term_certification(term_id),
    }


@router.get("/terms/{term_id}/sql-attributes", response_model=SqlAttributePageResponse)
def list_term_sql_attributes_by_id(
    term_id: str,
    skip: int = SKIP_QUERY,
    limit: int | None = LIMIT_QUERY,
) -> dict:
    """Return SqlAttribute nodes for a single Term.

    Attributes come back ordered by name, and *skip*/*limit* select one page
    of that order. ``count`` is the length of ``data`` (this page), ``total``
    the number of attributes the term has in full.
    """
    attrs = sql_attr_dal.fetch_sql_attributes_by_term_id(
        term_id, zone_ids=None, skip=skip, limit=limit
    )
    # `len(attrs)` only equals the term's full count when neither paging
    # argument was given — a `skip` alone (no `limit`) still returns a
    # partial read, so it needs the same separate count query.
    total = (
        sql_attr_dal.count_sql_attributes_by_term_id(term_id, zone_ids=None)
        if skip or limit is not None
        else len(attrs)
    )
    return {"data": attrs, "count": len(attrs), "total": total}


@router.patch("/terms/{term_id}", response_model=TermResponse)
def update_term(term_id: str, body: TermUpdate) -> dict:
    """Update a Term and invalidate dependent SqlAttribute suggestions."""
    patch = body.model_dump(exclude_unset=True)
    if not patch:
        raise HTTPException(status_code=422, detail="No Term fields to update")

    name = patch.get("name")
    if isinstance(name, str):
        name = name.strip()
    if "name" in patch and not name:
        raise HTTPException(status_code=422, detail="Term name cannot be blank")

    row = terms_dal.update_term(
        term_id,
        name=name if isinstance(name, str) else None,
        description=patch.get("description"),
        name_certified=patch.get("name_certified"),
        description_certified=patch.get("description_certified"),
    )
    if row is None:
        raise HTTPException(status_code=404, detail="Term not found")

    # Certification flags carry no embedding content; only name/description
    # changes need suggestion invalidation and a VDB refresh.
    content_changed = "name" in patch or "description" in patch
    name_changed = (
        content_changed and (row.get("old_name") or "").strip() != row["name"].strip()
    )
    if name_changed:
        sql_attr_dal.clear_sql_attribute_description_suggestions_for_term(term_id)

    if content_changed:
        try:
            term_service.refresh_term_embeddings(
                term_id, refresh_dependent_attrs=name_changed
            )
        except Exception:
            logger.warning(
                "Failed to refresh Term embeddings for %r", term_id, exc_info=True
            )

    return {
        "data": {
            "id": row["id"],
            "name": row["name"],
            "description": row.get("description"),
            "name_certified": row.get("name_certified", False),
            "description_certified": row.get("description_certified", False),
            "certification": terms_dal.get_term_certification(term_id),
        }
    }


@router.get("/terms/{term_id}", response_model=TermDetailResponse)
def get_term(term_id: str) -> dict:
    """Return a single Term node by id, including zones and related terms."""
    term = terms_dal.get_full_term_by_id(term_id, zone_ids=None)
    if term is None:
        raise HTTPException(status_code=404, detail="Term not found")
    term["related_terms"] = terms_dal.fetch_related_terms(term_id, zone_ids=None)
    return {"data": term}
