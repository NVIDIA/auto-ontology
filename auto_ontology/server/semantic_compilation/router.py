# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""API route for triggering semantic compilation on the ingestion service.

The settings UI stores the on/off flag directly in Postgres (via Prisma); these
routes only relay the "run it now" and "reset the layer" requests to the
ingestion service, mirroring how connection changes trigger ingest (see
``connections/service.py``).
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Query

from auto_ontology.connectors.registry import has_connector_for_database
from auto_ontology.dal.datasources import database_catalog_status
from auto_ontology.dal.terms import semantic_layer_calculated
from auto_ontology.ingestion_service.history import (
    get_last_failure_if_most_recent,
    get_last_successful_run,
)
from auto_ontology.server.ingestion.proxy import (
    is_semantic_compilation_running,
    trigger_semantic_compile,
    trigger_semantic_reset,
)
from auto_ontology.server.responses import (
    DatabaseSemanticStatus,
    SemanticStatusResponse,
    StatusResponse,
)
from auto_ontology.vdb import SEMANTIC_VDB_COLLECTION, shared_vdb

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/semantic-compilation/trigger", status_code=202, response_model=StatusResponse
)
async def trigger() -> dict[str, str]:
    """Ask the ingestion service to run (and schedule) semantic compilation."""
    trigger_semantic_compile()
    return {"status": "accepted"}


@router.post(
    "/semantic-compilation/reset", status_code=202, response_model=StatusResponse
)
def reset(
    database_name: str | None = Query(
        default=None,
        description=(
            "Reset only this database's semantic layer. Omit to reset every database."
        ),
    ),
) -> dict[str, str]:
    """Ask the ingestion service to rebuild the semantic layer.

    The service deletes the semantic nodes and embeddings and compiles them
    again, so the layer is rebuilt without waiting for the next scheduled run.
    With ``database_name`` only that database's layer is deleted; without it,
    every database's.
    """
    trigger_semantic_reset(database_name)
    return {"status": "accepted"}


@router.get(
    "/semantic-compilation/status",
    response_model=SemanticStatusResponse,
    response_model_exclude_unset=True,
)
def semantic_status(
    database_name: str | None = Query(
        default=None,
        min_length=1,
        description="Report readiness for this database's exact connector name.",
    ),
) -> SemanticStatusResponse:
    """Report semantic compilation status, optionally for one database.

    With a database name, report whether it has catalog tables, a configured
    connector, any table representing a semantic term, and stored semantic
    vectors. ``calculated`` then reflects that database's compilation state.
    Otherwise, ``calculated`` reports whether any semantic term exists.

    ``running`` and the last success and failure times describe the deployment's
    compilation scheduler in both cases. History remains available from Postgres
    when the ingestion process is unreachable.
    """
    scheduler = {
        "running": is_semantic_compilation_running(),
        "last_success_at": get_last_successful_run(),
        "last_failure_at": get_last_failure_if_most_recent(),
    }
    if database_name is None:
        return SemanticStatusResponse(
            calculated=semantic_layer_calculated(), **scheduler
        )

    ingested, compiled = database_catalog_status(database_name)
    database = DatabaseSemanticStatus(
        name=database_name,
        ingested=ingested,
        connected=has_connector_for_database(database_name),
        compiled=compiled,
        searchable=shared_vdb(SEMANTIC_VDB_COLLECTION).has_embeddings_for_database(
            database_name
        ),
    )
    return SemanticStatusResponse(calculated=compiled, database=database, **scheduler)
