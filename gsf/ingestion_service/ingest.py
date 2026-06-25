# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import threading
from typing import Any

from gsf.connectors.connection_string_factory import build_connection_string
from gsf.utils import get_embed_params
from nemo_retriever.graph import Graph
from nemo_retriever.tabular_data.operators.tabular_schema_extract_operator import (
    TabularSchemaExtractOp,
)
from nemo_retriever.tabular_data.operators.tabular_fetch_embeddings_operator import (
    TabularFetchEmbeddingsOp,
)
from nemo_retriever.operators.embed.operators import _BatchEmbedActor
from nemo_retriever.operators.vdb import IngestVdbOperator
from nemo_retriever.common.params.models import TabularExtractParams
from gsf.vdb import get_data_vdb, get_semantic_vdb
from gsf.connectors.registry import create_connector
from gsf.server.connections.dal import delete_database_subgraph
from gsf.semantic.compile import run_semantic_compilation

logger = logging.getLogger("ingestion_service.ingest")


def run_ingest(connection_string: str) -> None:
    TABULAR_PARAMS = TabularExtractParams(
        connector=create_connector(connection_string),
    )

    if not TABULAR_PARAMS.connector:
        raise ValueError("Connector is not set")

    try:
        database_name = TABULAR_PARAMS.connector.database_name
        embed_params = get_embed_params()

        graph = (
            Graph()
            >> TabularSchemaExtractOp(tabular_params=TABULAR_PARAMS)
            >> TabularFetchEmbeddingsOp(database_name=database_name)
            >> _BatchEmbedActor(params=embed_params)
        )

        results = graph.execute(None)
        result_df = results[0] if results else None

        if result_df is not None and not result_df.empty:
            data_vdb = get_data_vdb(database_name=database_name, reset=True)
            ingest_op = IngestVdbOperator(vdb=data_vdb)
            ingest_op(result_df.to_dict(orient="records"))
            logger.info(
                f"Tabular ingest result: {len(result_df)} rows written to pgvector",
            )
        else:
            logger.info("Tabular ingest result: no rows produced")

        logger.info("Starting semantic compilation")
        run_semantic_compilation(database_name)

    finally:
        TABULAR_PARAMS.connector.close()


def trigger_ingest(connection: dict[str, Any]) -> None:
    """Run ingest for a new connection without blocking the caller."""
    connection_string = build_connection_string(connection)
    database_name = str(connection.get("database") or "")

    def _run() -> None:
        try:
            run_ingest(connection_string)
        except Exception:
            logger.exception(
                "Background ingest failed for database %s",
                database_name,
            )

    threading.Thread(
        target=_run,
        daemon=True,
        name=f"ingest-{database_name}",
    ).start()


def trigger_ingest_delete(database_name: str) -> None:
    """Remove ingested database graph and embeddings without blocking the caller."""

    def _run() -> None:
        try:
            run_ingest_delete(database_name)
        except Exception:
            logger.exception(
                "Background ingest delete failed for database %s",
                database_name,
            )

    threading.Thread(
        target=_run,
        daemon=True,
        name=f"ingest-delete-{database_name}",
    ).start()


def run_ingest_delete(database_name: str) -> None:
    """Remove ingested database graph and embeddings for a database."""
    database_name = database_name.strip()
    if not database_name:
        raise ValueError("Database name is required")

    delete_database_subgraph(database_name)

    vdb = get_data_vdb()
    deleted_data_objects = vdb.delete_by_database(database_name)

    semantic_vdb = get_semantic_vdb()
    deleted_semantic = semantic_vdb.delete_by_database(database_name)
    logger.info(
        f"Tabular and semantic ingest delete: removed {len(deleted_data_objects) + len(deleted_semantic)} pgvector rows "
        f"for database {database_name}",
    )
