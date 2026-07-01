# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""API route handlers for semantic Terms."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from gsf.dal import terms as neo4j_dal

router = APIRouter()


@router.get("/terms")
def list_terms() -> dict:
    """Return all Term nodes from Neo4j."""
    terms, _attrs = neo4j_dal.fetch_all_terms_and_attributes()
    return {"data": terms, "count": len(terms)}


@router.get("/terms/column-attributes")
def list_term_column_attributes() -> dict:
    """Return all ColumnAttribute nodes with FK count and primary-key flag."""
    attrs = neo4j_dal.fetch_column_attributes_with_fk_count()
    return {"data": attrs, "count": len(attrs)}


@router.get("/terms/related-counts")
def list_related_terms_counts() -> dict:
    """Return per-term related-term counts derived from SEMANTIC_FK join paths."""
    counts = neo4j_dal.fetch_related_terms_counts()
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
