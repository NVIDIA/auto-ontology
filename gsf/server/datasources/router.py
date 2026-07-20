# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""API route handlers for datasources and connectors."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from gsf.server.custom_analyses import service as custom_analyses_dal
from gsf.server.datasources import service as dal
from gsf.server.pql_analyses import service as pql_analyses_dal


class NodeUpdate(BaseModel):
    description: str | None = None
    sample_values: list[str] | None = None


class CustomAnalysisValidate(BaseModel):
    sql: str = Field(..., min_length=1)


class CustomAnalysisCreate(BaseModel):
    name: str = Field(..., min_length=1)
    description: str
    sql: str = Field(..., min_length=1)


class PqlAnalysisCreate(BaseModel):
    name: str = Field(..., min_length=1)
    description: str
    pql: str = Field(..., min_length=1)


router = APIRouter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _count_payload(data: object) -> dict:
    if isinstance(data, list):
        return {"data": data, "count": len(data)}
    return {"data": data, "count": 1}


# ---------------------------------------------------------------------------
# Catalog lazy tree (/api/schemas, /api/tables, /api/columns)
# ---------------------------------------------------------------------------


@router.get("/schemas/{db_id}")
def list_schemas_by_database(
    db_id: str,
    zone_ids: list[str] | None = Query(default=None),
) -> dict:
    """Schemas for a database, zone-scoped when zone_ids are provided."""
    result = dal.fetch_schemas_for_database(db_id, zone_ids=zone_ids)
    return result


@router.get("/tables/{schema_id}")
def list_tables_by_schema(
    schema_id: str,
    database_name: str | None = None,
    zone_ids: list[str] | None = Query(default=None),
) -> dict:
    """Tables under a schema (lazy tree), zone-scoped when zone_ids are provided."""
    rows = dal.fetch_tables_for_schema(
        schema_id, database_name=database_name, zone_ids=zone_ids
    )
    return _count_payload(rows)


@router.get("/columns/{table_id}")
def list_columns_by_table(table_id: str) -> dict:
    """Columns for a table."""
    result = dal.fetch_columns_for_table(table_id)
    return _count_payload(result)


# ---------------------------------------------------------------------------
# Datasource routes (/api/datasources)
# ---------------------------------------------------------------------------


@router.get("/datasources/dbs")
def list_databases(zone_ids: list[str] | None = Query(default=None)) -> dict:
    """Databases visible via the given zones (all when zone_ids is absent)."""
    rows = dal.fetch_databases(zone_ids=zone_ids)
    return _count_payload(rows)


# ---------------------------------------------------------------------------
# Custom analyses (/api/custom-analyses)
# ---------------------------------------------------------------------------


@router.get("/custom-analyses")
def list_custom_analyses(zone_ids: list[str] | None = Query(default=None)) -> dict:
    """CustomAnalysis nodes joined with their HAS_SQL neighbour, zone-scoped when zone_ids are provided."""
    rows = custom_analyses_dal.list_custom_analyses(zone_ids=zone_ids)
    return _count_payload(rows)


@router.post("/custom-analyses/validate")
def validate_custom_analysis_sql(body: CustomAnalysisValidate) -> dict:
    """Validate a SQL expression against the full catalog.

    Does not create a CustomAnalysis. Returns 422 when the SQL can't be
    parsed or doesn't resolve to a known table.
    """
    try:
        result = custom_analyses_dal.validate_custom_analysis_sql(body.sql)
    except custom_analyses_dal.CustomAnalysisSqlError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"data": result}


@router.post("/custom-analyses", status_code=201)
def create_custom_analysis(body: CustomAnalysisCreate) -> dict:
    """Create a new CustomAnalysis with its Sql node.

    Strict insert: input shape is enforced by :class:`CustomAnalysisCreate`
    and the frontend is responsible for trimming and rejecting blank
    values before sending. Updating an existing analysis goes through
    ``PUT /custom-analyses/{analysis_id}``.

    Returns 409 when ``name`` or ``sql`` is already used by another
    CustomAnalysis (both are unique natural keys), and 422 when the SQL
    can't be parsed against the current catalog (no recognised tables) —
    without those references the analysis would be invisible to
    retrieval.
    """
    try:
        row = custom_analyses_dal.create_custom_analysis(
            name=body.name,
            description=body.description,
            sql=body.sql,
        )
    except (
        custom_analyses_dal.CustomAnalysisNameConflict,
        custom_analyses_dal.CustomAnalysisSqlConflict,
    ) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except custom_analyses_dal.CustomAnalysisSqlError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"data": row}


@router.put("/custom-analyses/{analysis_id}")
def update_custom_analysis(analysis_id: str, body: CustomAnalysisCreate) -> dict:
    """Replace a CustomAnalysis (matched by id) and re-link its Sql node.

    Returns 404 when no CustomAnalysis with ``analysis_id`` exists, 409
    when ``name`` or ``sql`` is already taken by a different
    CustomAnalysis, and 422 when the SQL can't be parsed against the
    current catalog — all surface the failure to the UI without
    producing an inconsistent graph.
    """
    try:
        row = custom_analyses_dal.update_custom_analysis(
            analysis_id=analysis_id,
            name=body.name,
            description=body.description,
            sql=body.sql,
        )
    except (
        custom_analyses_dal.CustomAnalysisNameConflict,
        custom_analyses_dal.CustomAnalysisSqlConflict,
    ) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except custom_analyses_dal.CustomAnalysisSqlError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"CustomAnalysis {analysis_id!r} not found",
        )
    return {"data": row}


@router.delete("/custom-analyses/{analysis_id}")
def delete_custom_analysis(analysis_id: str) -> dict:
    """Delete a CustomAnalysis (with its Sql node and VDB embedding).

    Returns 404 when no CustomAnalysis with ``analysis_id`` exists.
    On success the deleted ``{"id": ...}`` is echoed so the UI can
    confirm the targeted record was removed.
    """
    row = custom_analyses_dal.delete_custom_analysis(analysis_id)
    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"CustomAnalysis {analysis_id!r} not found",
        )
    return {"data": row}


# ---------------------------------------------------------------------------
# PQL analyses (/api/pql-analyses) — verified PQL few-shots for prediction
# ---------------------------------------------------------------------------


@router.get("/pql-analyses")
def list_pql_analyses() -> dict:
    """All PqlAnalysis nodes ``{id, name, description, pql}``."""
    return _count_payload(pql_analyses_dal.list_pql_analyses())


@router.post("/pql-analyses", status_code=201)
def create_pql_analysis(body: PqlAnalysisCreate) -> dict:
    """Create a new PqlAnalysis (verified PQL example) and embed it.

    Returns 409 when ``name`` or ``pql`` is already used by another PqlAnalysis.
    The PQL text is stored as-is — it is validated against the prediction graph at
    predict time, not here.
    """
    try:
        row = pql_analyses_dal.create_pql_analysis(
            name=body.name,
            description=body.description,
            pql=body.pql,
        )
    except (
        pql_analyses_dal.PqlAnalysisNameConflict,
        pql_analyses_dal.PqlAnalysisPqlConflict,
    ) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"data": row}


@router.put("/pql-analyses/{analysis_id}")
def update_pql_analysis(analysis_id: str, body: PqlAnalysisCreate) -> dict:
    """Replace a PqlAnalysis (matched by id) and re-embed it.

    Returns 404 when no PqlAnalysis with ``analysis_id`` exists, and 409 when
    ``name`` or ``pql`` is already taken by a different PqlAnalysis.
    """
    try:
        row = pql_analyses_dal.update_pql_analysis(
            analysis_id=analysis_id,
            name=body.name,
            description=body.description,
            pql=body.pql,
        )
    except (
        pql_analyses_dal.PqlAnalysisNameConflict,
        pql_analyses_dal.PqlAnalysisPqlConflict,
    ) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"PqlAnalysis {analysis_id!r} not found",
        )
    return {"data": row}


@router.delete("/pql-analyses/{analysis_id}")
def delete_pql_analysis(analysis_id: str) -> dict:
    """Delete a PqlAnalysis (with its VDB embedding).

    Returns 404 when no PqlAnalysis with ``analysis_id`` exists.
    """
    row = pql_analyses_dal.delete_pql_analysis(analysis_id)
    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"PqlAnalysis {analysis_id!r} not found",
        )
    return {"data": row}


# ---------------------------------------------------------------------------
# Node property update (/api/nodes/{node_id})
# ---------------------------------------------------------------------------


@router.patch("/nodes/{node_id}")
def update_node(node_id: str, body: NodeUpdate) -> dict:
    """Update mutable properties of any catalog node."""
    props = body.model_dump(exclude_none=True)
    result = dal.update_node_properties(node_id, props)
    return result
