# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import threading
from typing import Any

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
from gsf.connectors.registry import get_connectors, invalidate_connectors_cache
from gsf.dal.connections import delete_database_subgraph
from nemo_retriever.tabular_data.sql_database import SQLDatabase

logger = logging.getLogger("ingestion_service.ingest")


def run_ingest(connector: SQLDatabase) -> None:
    """Extract, embed, and persist a connector's tabular catalog.

    The connector's lifecycle is owned by the caller — ``run_ingest`` does not
    close it, since scheduled runs reuse cached connectors from
    :func:`gsf.connectors.registry.get_connectors`.
    """
    TABULAR_PARAMS = TabularExtractParams(connector=connector)

    if not TABULAR_PARAMS.connector:
        raise ValueError("Connector is not set")

    database_name = connector.database_name
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


def trigger_ingest(connection: dict[str, Any]) -> None:
    """Run ingest for a newly created connection without blocking the caller."""
    database_name = str(connection.get("database") or "")

    def _run() -> None:
        try:
            # The connection was just created, so reload the connector cache to
            # pick it up, then ingest via the same connectors the scheduler uses
            # (schema filter included).
            invalidate_connectors_cache()
            key = database_name.strip().casefold()
            connector = next(
                (
                    c
                    for c in get_connectors()
                    if str(getattr(c, "database_name", "")).casefold() == key
                ),
                None,
            )
            if connector is None:
                logger.error(
                    "Ingest skipped: no loaded connector for database %s",
                    database_name,
                )
                return
            run_ingest(connector)
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

    data_vdb = get_data_vdb()
    deleted_data_objects = data_vdb.delete_by_database(database_name)

    semantic_vdb = get_semantic_vdb()
    deleted_semantic = semantic_vdb.delete_by_database(database_name)
    logger.info(
        f"Tabular and semantic ingest delete: removed {len(deleted_data_objects) + len(deleted_semantic)} pgvector rows "
        f"for database {database_name}",
    )
