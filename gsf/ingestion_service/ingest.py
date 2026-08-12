# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from typing import Any, Iterator

from gsf.utils import get_embed_params
from gsf.utils.embedding import batch_embed
from gsf.utils.embedding_rows import (
    CatalogEmbeddingRowsOp,
)
from nemo_retriever.operators.vdb import IngestVdbOperator
from gsf.connectors.base import SQLDatabase

from gsf.catalog import ingest_catalog
from gsf.vdb import get_data_vdb
from gsf.connectors.registry import get_connectors, invalidate_connectors_cache
from gsf.dal.reset import delete_all_data, delete_semantic_layer

logger = logging.getLogger("ingestion_service.ingest")


@contextmanager
def _shared_connection(connector: SQLDatabase) -> Iterator[None]:
    """Hold one connection open for the whole extraction, where supported.

    Introspection issues a statement per table, and on Databricks opening a connection
    is both the slowest step (~0.9s) and the flakiest — measured hanging for minutes,
    and sometimes never completing. Paying that once per run instead of once per
    statement removes the dominant cost and the dominant failure mode.

    Connectors without the hook are used unchanged.
    """
    reuse = getattr(connector, "reuse_connection", None)
    if reuse is None:
        yield
        return
    with reuse():
        yield


def run_ingest(connector: SQLDatabase) -> None:
    """Extract, embed, and persist a connector's tabular catalog.

    The connector's lifecycle is owned by the caller — ``run_ingest`` does not
    close it, since scheduled runs reuse cached connectors from
    :func:`gsf.connectors.registry.get_connectors`.

    Three straight-line steps, formerly a ``Graph()`` chain of three operators.
    Only the middle one is still a library operator; the catalog write is
    GSF-owned (:func:`gsf.catalog.ingest_catalog`) and the embed step goes
    through :func:`gsf.utils.embedding.batch_embed`.
    """
    if connector is None:
        raise ValueError("Connector is not set")

    database_name = connector.database_name
    embed_params = get_embed_params()

    with _shared_connection(connector):
        tables_df, columns_df = ingest_catalog(connector)

    embed_rows = CatalogEmbeddingRowsOp(database_name=database_name)(
        (tables_df, columns_df)
    )
    result_df = batch_embed(embed_rows, embed_params)

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


def trigger_delete_ingest(database_name: str | None = None) -> None:
    """Delete a database's ingested graph and embeddings without blocking."""

    def _run() -> None:
        try:
            delete_all_data(database_name)
        except Exception:
            logger.exception(
                "Background delete-ingest failed for database %s",
                database_name,
            )

    threading.Thread(
        target=_run,
        daemon=True,
        name=f"delete-ingest-{database_name}",
    ).start()


def trigger_reset_semantic(database_name: str | None = None) -> None:
    """Delete a database's semantic layer without blocking.

    Passing ``None`` resets the semantic layer of every database.
    """

    def _run() -> None:
        try:
            delete_semantic_layer(database_name)
        except Exception:
            logger.exception(
                "Background reset-semantic failed for database %s",
                database_name or "<all>",
            )

    threading.Thread(
        target=_run,
        daemon=True,
        name=f"reset-semantic-{database_name or 'all'}",
    ).start()
