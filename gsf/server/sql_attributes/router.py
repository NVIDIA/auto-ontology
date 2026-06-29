# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""API route handlers for SqlAttribute CRUD."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from gsf.server.sql_attributes import dal

router = APIRouter()


class SqlAttributeCreate(BaseModel):
    name: str
    description: str
    expression: str
    term_id: str
    source: str = "manual"


class SqlAttributeUpdate(BaseModel):
    name: str
    description: str
    expression: str
    term_id: str
    source: str = "manual"


@router.get("/sql-attributes")
def list_sql_attributes() -> dict:
    """All SqlAttribute nodes with their linked Term."""
    rows = dal.list_sql_attributes()
    return {"data": rows, "count": len(rows)}


@router.get("/sql-attributes/{attr_id}")
def get_sql_attribute(attr_id: str) -> dict:
    """One SqlAttribute by id."""
    row = dal.get_sql_attribute(attr_id)
    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"SqlAttribute {attr_id!r} not found",
        )
    return {"data": row}


@router.post("/sql-attributes", status_code=201)
def create_sql_attribute(body: SqlAttributeCreate) -> dict:
    """Create a SqlAttribute with its Sql node, linked to a Term.

    Returns 409 when the name is already taken, 422 when the target
    Term does not exist or the SQL can't be parsed.
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
    except (ValueError, dal.SqlAttributeSqlError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"data": row}


@router.put("/sql-attributes/{attr_id}")
def update_sql_attribute(attr_id: str, body: SqlAttributeUpdate) -> dict:
    """Replace a SqlAttribute, re-parse SQL, and re-link to a Term.

    Returns 404 when no SqlAttribute with that id exists, 409 on name
    conflict, 422 when the Term doesn't exist or SQL can't be parsed.
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
    except (ValueError, dal.SqlAttributeSqlError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"SqlAttribute {attr_id!r} not found",
        )
    return {"data": row}


@router.delete("/sql-attributes/{attr_id}")
def delete_sql_attribute(attr_id: str) -> dict:
    """Delete a SqlAttribute and its edges."""
    row = dal.delete_sql_attribute(attr_id)
    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"SqlAttribute {attr_id!r} not found",
        )
    return {"data": row}
