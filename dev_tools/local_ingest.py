# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local ingest: docker-compose Postgres source → pgvector embeddings store.

Run after ``docker compose up -d`` and ``dev_tools.seed_local_postgres``.

Usage::

    uv run --no-sync python -m dev_tools.local_ingest
"""

from __future__ import annotations

import logging
import os

from nemo_retriever.tabular_data.operators.tabular_fetch_embeddings_operator import (
    TabularFetchEmbeddingsOp,
)
from nemo_retriever.operators.vdb import IngestVdbOperator
from gsf.catalog import ingest_catalog
from gsf.utils import get_embed_params
from gsf.utils.embedding import batch_embed
from gsf.vdb import get_data_vdb
from gsf.connectors.registry import create_connector

logger = logging.getLogger("dev_tools.local_ingest")


def run_ingest(connection_string: str) -> None:
    """Ingest the catalog, embed it, and write the embeddings to pgvector."""
    connector = create_connector(connection_string)
    database_name = connector.database_name

    schema_data = ingest_catalog(connector)

    embed_rows = TabularFetchEmbeddingsOp(database_name=database_name)(schema_data)
    result_df = batch_embed(embed_rows, get_embed_params())

    # PostgresVDB.__init__ wipes existing rows for `database_name` before we
    # re-ingest, so the store starts clean for this database.
    vdb = get_data_vdb(database_name=database_name)

    if result_df is not None and not result_df.empty:
        ingest_op = IngestVdbOperator(vdb=vdb)
        ingest_op(result_df.to_dict(orient="records"))
        logger.info("Tabular ingest: %d rows written to pgvector.", len(result_df))
    else:
        logger.info("Tabular ingest: no rows produced.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # Remote source DB to extract tabular schema/embeddings from. Kept separate
    # from the local POSTGRES_* vars (which point at the pgvector store).
    _CONNECTION_STRINGS = os.environ.get("CONNECTION_STRINGS", "")
    connection_string = _CONNECTION_STRINGS.split(",")[0].strip()
    if not connection_string:
        raise EnvironmentError(
            "CONNECTION_STRINGS is not set. Add it to your .env, e.g.:\n\n"
            "    CONNECTION_STRINGS=postgresql://user:password@host:5432/dbname"
        )
    try:
        run_ingest(connection_string)
    except KeyboardInterrupt:
        logger.info("local_ingest: shutting down")
        raise SystemExit(0)
