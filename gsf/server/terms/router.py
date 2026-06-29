# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""API route handlers for semantic Terms."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from gsf.semantic import neo4j_dal

router = APIRouter()


@router.get("/terms")
def list_terms() -> dict:
    """Return all Term nodes from Neo4j."""
    terms, _attrs = neo4j_dal.fetch_all_terms_and_attributes()
    return {"data": terms, "count": len(terms)}


@router.get("/terms/attributes")
def list_term_attributes() -> dict:
    """Return all ColumnAttribute nodes with FK count and primary-key flag."""
    attrs = neo4j_dal.fetch_column_attributes_with_fk_count()
    return {"data": attrs, "count": len(attrs)}


@router.get("/terms/{term_id}")
def get_term(term_id: str) -> dict:
    """Return a single Term node by id."""
    term = neo4j_dal.fetch_term_by_id(term_id)
    if term is None:
        raise HTTPException(status_code=404, detail="Term not found")
    return {"data": term}
