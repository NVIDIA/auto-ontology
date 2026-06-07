# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

from gsf.utils import get_embed_params
from nemo_retriever.graph import Graph
from nemo_retriever.graph.tabular_schema_extract_operator import TabularSchemaExtractOp
from nemo_retriever.graph.tabular_fetch_embeddings_operator import (
    TabularFetchEmbeddingsOp,
)
from nemo_retriever.text_embed.operators import _BatchEmbedActor
from nemo_retriever.vdb import IngestVdbOperator
from nemo_retriever.params import TabularExtractParams
from gsf.vdb import get_vdb
from gsf.connectors.registry import create_connector

logger = logging.getLogger("ingestion_service.ingest")


def run_ingest(connection_string: str) -> None:
    TABULAR_PARAMS = TabularExtractParams(
        connector=create_connector(connection_string),
    )

    if not TABULAR_PARAMS.connector:
        raise ValueError("Connector is not set")

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
        ingest_op = IngestVdbOperator(vdb=get_vdb(database_name=database_name))
        ingest_op(result_df.to_dict(orient="records"))
        logger.info(
            f"Tabular ingest result: {len(result_df)} rows written to pgvector",
        )
    else:
        logger.info("Tabular ingest result: no rows produced")
