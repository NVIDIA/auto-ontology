# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""API route handlers for SqlAttribute CRUD."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from auto_ontology.dal import terms as terms_dal
from auto_ontology.dal.owners import CATALOG_EDIT, edit_subject
from auto_ontology.server.identity import resolve_internal_user
from auto_ontology.server.owners import enforce_edit, owner_from_body
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
    owner_id: str | None = None


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
    owner_id: str | None = None


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
def get_sql_attribute_description_suggestion(request: Request, attr_id: str) -> dict:
    """LLM-generated (or cached) description suggestion for a SqlAttribute.

    Returns 404 when no SqlAttribute with that id exists. ``data`` is
    ``null`` when a suggestion could not be produced (e.g. the LLM call
    failed) — that is not treated as an error. Returns 403 when the caller
    may not edit this attribute. A suggestion is cached on the attribute,
    so the same edit check as a save applies.
    """
    subject = edit_subject(attr_id)
    if subject is not None:
        enforce_edit(request, subject)
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
def create_sql_attribute(request: Request, body: SqlAttributeCreate) -> dict:
    """Create a SqlAttribute with its Sql node, linked to a Term.

    The connector is resolved server-side (never chosen by the client —
    see ``auto_ontology/connectors/registry.py``). ``owner_id``, when set,
    names a user or a group that can edit the catalog; omit it to leave the
    attribute unassigned. The caller is stored as ``created_by`` so the side
    panel can name them. Returns 400 when that id cannot be assigned, 409
    when the name is already taken, 422 when the target Term does not exist
    or the SQL can't be parsed.
    """
    try:
        row = dal.create_sql_attribute(
            name=body.name,
            description=body.description,
            expression=body.expression,
            term_id=body.term_id,
            source=body.source,
            owner_id=owner_from_body(body.owner_id, CATALOG_EDIT),
            created_by=resolve_internal_user(request, required=False),
        )
    except dal.SqlAttributeNameConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except dal.SqlAttributeExpressionConflict as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (ValueError, dal.SqlAttributeSqlError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"data": row}


@router.put("/sql-attributes/{attr_id}", response_model=SqlAttributeResponse)
def update_sql_attribute(
    request: Request, attr_id: str, body: SqlAttributeUpdate
) -> dict:
    """Replace a SqlAttribute, re-parse SQL, and re-link to a Term.

    The connector is resolved server-side (never chosen by the client —
    see ``auto_ontology/connectors/registry.py``). Returns 404 when no SqlAttribute
    with that id exists, 409 on name conflict, 422 when the Term doesn't
    exist or SQL can't be parsed. Returns 403 when the caller may not edit
    this attribute, the same check as the metadata patch.
    """
    subject = edit_subject(attr_id)
    if subject is not None:
        enforce_edit(request, subject)
    try:
        row = dal.update_sql_attribute(
            attr_id=attr_id,
            name=body.name,
            description=body.description,
            expression=body.expression,
            term_id=body.term_id,
            source=body.source,
            modified_by=resolve_internal_user(request, required=False),
            touch=True,
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
def patch_sql_attribute(
    request: Request, attr_id: str, body: SqlAttributeMetadataPatch
) -> dict:
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

    subject = edit_subject(attr_id)
    if subject is not None:
        enforce_edit(request, subject)
    set_owner = "owner_id" in patch
    owner_id = owner_from_body(body.owner_id, CATALOG_EDIT) if set_owner else None

    try:
        row = dal.update_sql_attribute(
            attr_id=attr_id,
            name=name if isinstance(name, str) else None,
            description=patch.get("description"),
            certified=patch.get("certified"),
            owner_id=owner_id,
            set_owner=set_owner,
            modified_by=resolve_internal_user(request, required=False),
            touch=True,
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
def delete_sql_attribute(request: Request, attr_id: str) -> dict:
    """Delete a SqlAttribute and its edges."""
    subject = edit_subject(attr_id)
    if subject is not None:
        enforce_edit(request, subject)
    row = dal.delete_sql_attribute(attr_id)
    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"SqlAttribute {attr_id!r} not found",
        )
    return {"data": row}
