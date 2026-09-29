# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""API route handlers for SqlAttribute CRUD."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from auto_ontology.dal import terms as terms_dal
from auto_ontology.server.sql_attributes import service as dal
from auto_ontology.server.responses import (
    DescriptionSuggestionResponse,
    IdResponse,
    SqlAttributeListResponse,
    SqlAttributeResponse,
    SqlAttributePatchResponse,
    SqlExpressionValidationResponse,
)

router = APIRouter()


class SqlAttributeValidate(BaseModel):
    expression: str = Field(..., min_length=1)
    term_id: str | None = None
    attribute_id: str | None = None


class SqlAttributeCreate(BaseModel):
    name: str = Field(..., min_length=1)
    description: str
    expression: str = Field(..., min_length=1)
    term_id: str
    source: str = "manual"


class SqlAttributeUpdate(BaseModel):
    name: str = Field(..., min_length=1)
    description: str
    expression: str = Field(..., min_length=1)
    term_id: str
    source: str = "manual"


class SqlAttributeMetadataPatch(BaseModel):
    name: str | None = Field(default=None, min_length=1)
    description: str | None = None
    certified: bool | None = None


@router.get("/sql-attributes", response_model=SqlAttributeListResponse)
def list_sql_attributes() -> dict:
    """All SqlAttribute nodes with their linked Term."""
    rows = dal.list_sql_attributes()
    return {"data": rows, "count": len(rows)}


@router.get("/sql-attributes/{attr_id}", response_model=SqlAttributeResponse)
def get_sql_attribute(attr_id: str) -> dict:
    """One SqlAttribute by id, including its resolved zones."""
    row = dal.get_full_sql_attribute_by_id(attr_id, zone_ids=None)
    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"SqlAttribute {attr_id!r} not found",
        )
    return {"data": row}


@router.get(
    "/sql-attributes/{attr_id}/description-suggestion",
    response_model=DescriptionSuggestionResponse,
)
def get_sql_attribute_description_suggestion(attr_id: str) -> dict:
    """LLM-generated (or cached) description suggestion for a SqlAttribute.

    Returns 404 when no SqlAttribute with that id exists. ``data`` is
    ``null`` when a suggestion could not be produced (e.g. the LLM call
    failed) — that is not treated as an error.
    """
    if dal.get_sql_attribute_by_id(attr_id) is None:
        raise HTTPException(
            status_code=404,
            detail=f"SqlAttribute {attr_id!r} not found",
        )
    return {"data": dal.suggest_sql_attribute_description(attr_id)}


@router.post("/sql-attributes/validate", response_model=SqlExpressionValidationResponse)
def validate_sql_attribute(body: SqlAttributeValidate) -> dict:
    """Validate a SQL expression against the catalog.

    The connector is resolved server-side (never chosen by the client —
    see ``auto_ontology/connectors/registry.py``). Does not create a SqlAttribute.
    Returns 422 when the SQL can't be parsed or doesn't resolve to a
    known table.
    """
    try:
        result = dal.validate_sql_attribute(
            expression=body.expression,
            term_id=body.term_id,
            attribute_id=body.attribute_id,
        )
    except (dal.SqlAttributeExpressionConflict, dal.SqlAttributeSqlError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"data": result}


@router.post("/sql-attributes", status_code=201, response_model=SqlAttributeResponse)
def create_sql_attribute(body: SqlAttributeCreate) -> dict:
    """Create a SqlAttribute with its Sql node, linked to a Term.

    The connector is resolved server-side (never chosen by the client —
    see ``auto_ontology/connectors/registry.py``). Returns 409 when the name is
    already taken, 422 when the target Term does not exist or the SQL
    can't be parsed.
    """
    try:
        row = dal.create_sql_attribute(
            name=body.name,
            description=body.description,
            expression=body.expression,
            term_id=body.term_id,
            source=body.source,
        )
    except dal.SqlAttributeNameConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except dal.SqlAttributeExpressionConflict as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (ValueError, dal.SqlAttributeSqlError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"data": row}


@router.put("/sql-attributes/{attr_id}", response_model=SqlAttributeResponse)
def update_sql_attribute(attr_id: str, body: SqlAttributeUpdate) -> dict:
    """Replace a SqlAttribute, re-parse SQL, and re-link to a Term.

    The connector is resolved server-side (never chosen by the client —
    see ``auto_ontology/connectors/registry.py``). Returns 404 when no SqlAttribute
    with that id exists, 409 on name conflict, 422 when the Term doesn't
    exist or SQL can't be parsed.
    """
    try:
        row = dal.update_sql_attribute(
            attr_id=attr_id,
            name=body.name,
            description=body.description,
            expression=body.expression,
            term_id=body.term_id,
            source=body.source,
        )
    except dal.SqlAttributeNameConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except dal.SqlAttributeExpressionConflict as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (ValueError, dal.SqlAttributeSqlError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"SqlAttribute {attr_id!r} not found",
        )
    return {"data": row}


@router.patch("/sql-attributes/{attr_id}", response_model=SqlAttributePatchResponse)
def patch_sql_attribute(attr_id: str, body: SqlAttributeMetadataPatch) -> dict:
    """Patch SqlAttribute name/description without touching SQL expression."""
    patch = body.model_dump(exclude_unset=True)
    if not patch:
        raise HTTPException(status_code=422, detail="No SqlAttribute fields to update")

    name = patch.get("name")
    if isinstance(name, str):
        name = name.strip()
    if "name" in patch and not name:
        raise HTTPException(
            status_code=422, detail="SQL attribute name cannot be blank"
        )

    try:
        row = dal.update_sql_attribute(
            attr_id=attr_id,
            name=name if isinstance(name, str) else None,
            description=patch.get("description"),
            certified=patch.get("certified"),
        )
    except dal.SqlAttributeNameConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"SqlAttribute {attr_id!r} not found",
        )
    term_id = row.get("term_id")
    return {
        "data": row,
        # The owning Term's aggregate badge depends on this attribute's flag,
        # so hand back the recomputed value rather than making the client
        # re-derive it (see terms_dal.get_term_certification).
        "term_certification": (
            terms_dal.get_term_certification(term_id) if term_id else None
        ),
    }


@router.delete("/sql-attributes/{attr_id}", response_model=IdResponse)
def delete_sql_attribute(attr_id: str) -> dict:
    """Delete a SqlAttribute and its edges."""
    row = dal.delete_sql_attribute(attr_id)
    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"SqlAttribute {attr_id!r} not found",
        )
    return {"data": row}
