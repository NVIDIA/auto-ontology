# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""API route handlers for semantic Terms."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from gsf.dal import sql_attributes as sql_attr_dal
from gsf.dal import terms as terms_dal
from gsf.server.terms import service as term_service

router = APIRouter()
logger = logging.getLogger(__name__)


class TermUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1)
    description: str | None = None


@router.get("/terms")
def list_terms(
    zone_ids: list[str] | None = Query(default=None),
    q: str | None = Query(default=None),
) -> dict:
    """Return Term nodes zone-scoped to the provided zones.

    ``None`` (param absent) → no filter, return all (admin callers).
    ``[]`` (empty list) → viewer with no zone access, returns empty.
    ``[id, ...]`` → filter to terms reachable through those zones.

    *q*, when given, additionally filters to terms whose name contains it
    (case-insensitive).
    """
    terms, _attrs = terms_dal.fetch_all_terms_and_attributes(
        zone_ids=zone_ids, search=q
    )
    return {"data": terms, "count": len(terms)}


@router.get("/terms/column-attributes")
def list_term_column_attributes(
    zone_ids: list[str] | None = Query(default=None),
) -> dict:
    """Return ColumnAttribute nodes (zone-scoped when zone_ids provided)."""
    attrs = terms_dal.fetch_column_attributes(zone_ids=zone_ids)
    return {"data": attrs, "count": len(attrs)}


@router.get("/terms/sql-attributes")
def list_term_sql_attributes(
    zone_ids: list[str] | None = Query(default=None),
) -> dict:
    """Return SqlAttribute nodes (zone-scoped when zone_ids provided)."""
    attrs = sql_attr_dal.fetch_sql_attributes(zone_ids=zone_ids)
    return {"data": attrs, "count": len(attrs)}


@router.get("/terms/related-counts")
def list_related_terms_counts(
    zone_ids: list[str] | None = Query(default=None),
) -> dict:
    """Return per-term related-term counts, zone-scoped when zone_ids are provided."""
    counts = terms_dal.fetch_related_terms_counts(zone_ids=zone_ids)
    return {"data": counts, "count": len(counts)}


@router.get("/terms/{term_id}/column-attributes")
def list_term_column_attributes_by_id(
    term_id: str,
    zone_ids: list[str] | None = Query(default=None),
) -> dict:
    """Return ColumnAttribute nodes for a single Term with FK count.

    Zone-scoped when zone_ids are provided, so a viewer can't see attributes
    of out-of-zone tables just because they belong to a term they can see.
    """
    attrs = terms_dal.fetch_column_attributes_by_term_id(term_id, zone_ids=zone_ids)
    return {"data": attrs, "count": len(attrs)}


@router.get("/terms/{term_id}/sql-attributes")
def list_term_sql_attributes_by_id(
    term_id: str,
    zone_ids: list[str] | None = Query(default=None),
) -> dict:
    """Return SqlAttribute nodes for a single Term.

    Zone-scoped when zone_ids are provided, matching the term visibility
    rules used by the single-term detail endpoint.
    """
    attrs = sql_attr_dal.fetch_sql_attributes_by_term_id(term_id, zone_ids=zone_ids)
    return {"data": attrs, "count": len(attrs)}


@router.get("/terms/{term_id}/related-terms")
def list_related_terms(
    term_id: str,
    zone_ids: list[str] | None = Query(default=None),
) -> dict:
    """Return Term nodes related to the given term via SEMANTIC_FK join paths.

    Zone-scoped when zone_ids are provided, so related terms outside the
    caller's zones are never surfaced as clickable chips.
    """
    related = terms_dal.fetch_related_terms(term_id, zone_ids=zone_ids)
    return {"data": related, "count": len(related)}


@router.patch("/terms/{term_id}")
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
    )
    if row is None:
        raise HTTPException(status_code=404, detail="Term not found")

    name_changed = (row.get("old_name") or "").strip() != row["name"].strip()
    if name_changed:
        sql_attr_dal.clear_sql_attribute_description_suggestions_for_term(term_id)

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
        }
    }


@router.get("/terms/{term_id}")
def get_term(
    term_id: str,
    zone_ids: list[str] | None = Query(default=None),
) -> dict:
    """Return a single Term node by id, including its resolved zones.

    Zone-scoped when zone_ids are provided: a term that also represents an
    out-of-zone table is treated as not found, so a viewer can't bypass the
    ``/terms`` list's zone scoping by requesting a term directly by id.
    """
    term = terms_dal.get_full_term_by_id(term_id, zone_ids=zone_ids)
    if term is None:
        raise HTTPException(status_code=404, detail="Term not found")
    return {"data": term}
