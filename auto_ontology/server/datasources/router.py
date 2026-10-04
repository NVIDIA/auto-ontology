# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""API route handlers for datasources and connectors."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Path, Query
from pydantic import BaseModel, Field

from auto_ontology.server.custom_analyses import service as custom_analyses_dal
from auto_ontology.server.datasources import service as dal
from auto_ontology.server.pagination import LIMIT_QUERY, SKIP_QUERY
from auto_ontology.server.pql_analyses import service as pql_analyses_dal
from auto_ontology.server.models import NodeUpdateResult
from auto_ontology.server.responses import (
    CustomAnalysisPageResponse,
    CustomAnalysisResponse,
    DatabaseListResponse,
    IdResponse,
    PqlAnalysisPageResponse,
    PqlAnalysisResponse,
    SchemasPayload,
    SqlValidationResponse,
    TableColumnsPageResponse,
    TableListResponse,
)


class NodeUpdate(BaseModel):
    description: str | None = None
    sample_values: list[str] | None = None
    description_certified: bool | None = None


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


@router.get("/schemas/{db_id}", response_model=SchemasPayload | None)
def list_schemas_by_database(db_id: str) -> dict:
    """Schemas for a database."""
    result = dal.fetch_schemas_for_database(db_id, zone_ids=None)
    return result


@router.get("/tables/{schema_id}", response_model=TableListResponse)
def list_tables_by_schema(
    schema_id: str,
    database_name: str | None = Query(
        default=None,
        description=(
            "Accepted for API compatibility and ignored — schema ids are "
            "globally unique, so the database is already implied."
        ),
    ),
) -> dict:
    """Tables under a schema (lazy tree)."""
    rows = dal.fetch_tables_for_schema(
        schema_id, database_name=database_name, zone_ids=None
    )
    return _count_payload(rows)


@router.get("/columns/{table_id}", response_model=TableColumnsPageResponse)
def list_columns_by_table(
    table_id: str,
    skip: int = SKIP_QUERY,
    limit: int | None = LIMIT_QUERY,
) -> dict:
    """Columns for a table.

    Columns come back ordered by ordinal position, and *skip*/*limit* select
    one page of that order; ``total`` reports how many the table has in full,
    so a pager knows when to stop asking. Omitting *limit* returns every
    column, which is what the lazy catalog tree loads.
    """
    # `fetch_columns_for_table` returns None only when the table is missing —
    # falling back to an empty envelope keeps the response shape
    # `{data: {columns: [...]}, count: 1}` instead of `data: null`, which
    # crashes callers (e.g. `frontend/api/datasources.ts`) that read
    # `envelope.columns` unconditionally.
    result = dal.fetch_columns_for_table(table_id, skip=skip, limit=limit) or {
        "columns": []
    }
    columns = result.get("columns") or []
    # `len(columns)` only equals the table's full count when neither paging
    # argument was given — a `skip` alone (no `limit`) still returns a
    # partial read, so it needs the same separate count query.
    total = (
        dal.count_columns_for_table(table_id)
        if skip or limit is not None
        else len(columns)
    )
    # `columns_count` describes the table, not the page: the envelope would
    # otherwise carry two numbers for the same thing that disagree as soon as
    # a page is requested (see `mergeTable` in
    # `frontend/lib/data/datasource-tree-merge.ts`, which maxes them together).
    return {**_count_payload({**result, "columns_count": total}), "total": total}


# ---------------------------------------------------------------------------
# Datasource routes (/api/datasources)
# ---------------------------------------------------------------------------


@router.get("/datasources/dbs", response_model=DatabaseListResponse)
def list_databases() -> dict:
    """Databases in the catalog."""
    rows = dal.fetch_databases(zone_ids=None)
    return _count_payload(rows)


# ---------------------------------------------------------------------------
# Custom analyses (/api/custom-analyses)
# ---------------------------------------------------------------------------


@router.get("/custom-analyses", response_model=CustomAnalysisPageResponse)
def list_custom_analyses(
    query: str | None = Query(
        default=None,
        description="Case-insensitive substring filter on the analysis name.",
    ),
    skip: int = SKIP_QUERY,
    limit: int | None = LIMIT_QUERY,
) -> dict:
    """CustomAnalysis nodes joined with their HAS_SQL neighbour.

    *query*, when given, keeps the analyses whose name contains it
    (case-insensitive).

    Analyses come back ordered by name, and *skip*/*limit* select one page of
    that order; ``total`` counts every match, so a caller knows when to stop
    asking. Omitting *limit* returns every matching analysis.
    """
    rows = custom_analyses_dal.list_custom_analyses(
        zone_ids=None, search=query, skip=skip, limit=limit
    )
    # The count is a second read, so it is worth skipping for the request that
    # asked for everything: a whole unpaged list already is its own total.
    total = (
        custom_analyses_dal.count_custom_analyses(zone_ids=None, search=query)
        if skip or limit is not None
        else len(rows)
    )
    return {**_count_payload(rows), "total": total}


@router.post("/custom-analyses/validate", response_model=SqlValidationResponse)
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


@router.post("/custom-analyses", status_code=201, response_model=CustomAnalysisResponse)
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


@router.put("/custom-analyses/{analysis_id}", response_model=CustomAnalysisResponse)
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


@router.delete("/custom-analyses/{analysis_id}", response_model=IdResponse)
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


@router.get("/pql-analyses", response_model=PqlAnalysisPageResponse)
def list_pql_analyses(
    query: str | None = Query(
        default=None,
        description="Case-insensitive substring filter on the analysis name.",
    ),
    skip: int = SKIP_QUERY,
    limit: int | None = LIMIT_QUERY,
) -> dict:
    """PqlAnalysis nodes ``{id, name, description, pql}``.

    *query*, ``skip`` and ``limit`` behave as they do on ``/custom-analyses``:
    a name filter over a page of the name order, with ``total`` counting
    every match rather than the rows on this page.
    """
    rows = pql_analyses_dal.list_pql_analyses(search=query, skip=skip, limit=limit)
    total = (
        pql_analyses_dal.count_pql_analyses(search=query)
        if skip or limit is not None
        else len(rows)
    )
    return {**_count_payload(rows), "total": total}


@router.post("/pql-analyses", status_code=201, response_model=PqlAnalysisResponse)
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


@router.put("/pql-analyses/{analysis_id}", response_model=PqlAnalysisResponse)
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


@router.delete("/pql-analyses/{analysis_id}", response_model=IdResponse)
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


@router.patch(
    "/nodes/{node_id}",
    response_model=NodeUpdateResult | None,
    # The body echoes only the properties the patch actually sent, so the
    # model's other fields must not be filled in with nulls that the handler
    # never returned — this keeps the response byte-identical to before it was
    # typed.
    response_model_exclude_unset=True,
)
def update_node(
    node_id: str = Path(
        description="Id of any catalog node — a Database, Schema, Table or Column."
    ),
    body: NodeUpdate = ...,
) -> dict:
    """Update mutable properties of any catalog node."""
    props = body.model_dump(exclude_none=True)
    try:
        result = dal.update_node_properties(node_id, props)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return result
