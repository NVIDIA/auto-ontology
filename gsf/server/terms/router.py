# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""API route handlers for semantic Terms."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from gsf.dal import terms as neo4j_dal

router = APIRouter()


@router.get("/terms")
def list_terms(zone_ids: list[str] | None = Query(default=None)) -> dict:
    """Return Term nodes zone-scoped to the provided zones.

    ``None`` (param absent) → no filter, return all (admin callers).
    ``[]`` (empty list) → viewer with no zone access, returns empty.
    ``[id, ...]`` → filter to terms reachable through those zones.
    """
    terms, _attrs = neo4j_dal.fetch_all_terms_and_attributes(zone_ids=zone_ids)
    return {"data": terms, "count": len(terms)}


@router.get("/terms/column-attributes")
def list_term_column_attributes(
    zone_ids: list[str] | None = Query(default=None),
) -> dict:
    """Return ColumnAttribute nodes with FK count (zone-scoped when zone_ids provided)."""
    attrs = neo4j_dal.fetch_column_attributes_with_fk_count(zone_ids=zone_ids)
    return {"data": attrs, "count": len(attrs)}


@router.get("/terms/related-counts")
def list_related_terms_counts(
    zone_ids: list[str] | None = Query(default=None),
) -> dict:
    """Return per-term related-term counts, zone-scoped when zone_ids are provided."""
    counts = neo4j_dal.fetch_related_terms_counts(zone_ids=zone_ids)
    return {"data": counts, "count": len(counts)}


@router.get("/terms/{term_id}/column-attributes")
def list_term_column_attributes_by_id(term_id: str) -> dict:
    """Return ColumnAttribute nodes for a single Term with FK count."""
    attrs = neo4j_dal.fetch_column_attributes_by_term_id(term_id)
    return {"data": attrs, "count": len(attrs)}


@router.get("/terms/{term_id}/related-terms")
def list_related_terms(term_id: str) -> dict:
    """Return Term nodes related to the given term via SEMANTIC_FK join paths."""
    related = neo4j_dal.fetch_related_terms(term_id)
    return {"data": related, "count": len(related)}


@router.get("/terms/{term_id}")
def get_term(term_id: str) -> dict:
    """Return a single Term node by id, including its resolved zones."""
    term = neo4j_dal.fetch_term_by_id(term_id)
    if term is None:
        raise HTTPException(status_code=404, detail="Term not found")
    return {"data": term}
