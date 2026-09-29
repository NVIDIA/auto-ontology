# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP routes for native Auto Ontology and Apache Ossie model YAML export/import."""

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import Response
from ossie_nvidia_gsf import GSFConversionError
from pydantic import ValidationError

from auto_ontology.dal.model_interchange import (
    ModelImportValidationError,
    UnknownDatabaseIdsError,
)
from auto_ontology.server.model_interchange import service
from auto_ontology.server.model_interchange.schemas import ExportRequest
from auto_ontology.server.responses import ModelImportResponse

router = APIRouter()


class YamlResponse(Response):
    """Declares the export's media type so the spec doesn't also claim JSON."""

    media_type = "application/x-yaml"


@router.post(
    "/model/export",
    response_class=YamlResponse,
    responses={
        200: {
            "content": {"application/x-yaml": {"schema": {"type": "string"}}},
            "description": "Auto Ontology model YAML, sent as a file attachment",
        }
    },
)
def export_model(body: ExportRequest) -> Response:
    """Export the scoped catalog + semantic model as a YAML attachment."""
    try:
        yaml_text = service.export_model(body)
    except UnknownDatabaseIdsError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except GSFConversionError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Cannot express this model as Apache Ossie YAML: {exc}",
        ) from exc
    return Response(
        content=yaml_text,
        media_type="application/x-yaml",
        headers={
            "Content-Disposition": (
                f'attachment; filename="{body.format.value}-model.yaml"'
            ),
        },
    )


@router.post("/model/import", response_model=ModelImportResponse)
async def import_model(
    request: Request,
    file: UploadFile | None = File(default=None),
    replace: bool = Query(default=True),
    embed: bool = Query(default=True),
) -> dict:
    """Import a native Auto Ontology or Apache Ossie model YAML file or raw YAML body."""
    if file is not None:
        raw = await file.read()
    else:
        raw = await request.body()

    if not raw:
        raise HTTPException(
            status_code=422,
            detail="Provide YAML as a multipart 'file' upload or raw request body",
        )

    try:
        yaml_text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=422, detail="YAML must be UTF-8") from exc

    try:
        summary = service.import_model(yaml_text, replace=replace, embed=embed)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc
    except GSFConversionError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Cannot read this file as Apache Ossie YAML: {exc}",
        ) from exc
    except ModelImportValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except UnknownDatabaseIdsError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return {"success": True, "summary": summary}
