# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from typing import Any, Iterator

from gsf.utils import get_embed_params
from gsf.utils.embedding import batch_embed_chunks
from gsf.utils.embedding_rows import (
    CatalogEmbeddingRowsOp,
)
from nemo_retriever.common.vdb.records import to_client_vdb_records
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


def _row_count(frame: Any) -> int:
    """Row count for a possibly-``None`` DataFrame.

    Kept defensive on purpose: this only feeds a log line, and a completion log
    that raises would turn a successful ingest into a failed one.
    """
    try:
        return 0 if frame is None else len(frame)
    except TypeError:
        return 0


def run_ingest(connector: SQLDatabase) -> None:
    """Extract, embed, and persist a connector's tabular catalog.

    The connector's lifecycle is owned by the caller — ``run_ingest`` does not
    close it, since scheduled runs reuse cached connectors from
    :func:`gsf.connectors.registry.get_connectors`.

    Three straight-line steps, formerly a ``Graph()`` chain of three operators.
    Only the middle one is still a library operator; the catalog write is
    GSF-owned (:func:`gsf.catalog.ingest_catalog`) and the embed step goes
    through :func:`gsf.utils.embedding.batch_embed_chunks`.

    Embedding is the long pole — minutes of remote round trips against tens of
    seconds for everything else — so its chunks are written to pgvector as they
    arrive rather than buffered into one DataFrame. That keeps peak memory to a
    chunk, overlaps the write with the next chunk's embed, and means a failure
    three minutes in has already persisted most of the catalog.
    """
    if connector is None:
        raise ValueError("Connector is not set")

    database_name = connector.database_name
    embed_params = get_embed_params()

    started = time.monotonic()
    logger.info("Data ingestion started for database %s", database_name)

    with _shared_connection(connector):
        tables_df, columns_df = ingest_catalog(connector)

    embed_rows = CatalogEmbeddingRowsOp(database_name=database_name)(
        (tables_df, columns_df)
    )

    data_vdb = None
    rows_written = 0
    for chunk in batch_embed_chunks(embed_rows, embed_params, label=database_name):
        records = to_client_vdb_records(chunk)
        if not records:
            # Every row in the chunk came back without an embedding. Already
            # logged as a warning by the embed step; nothing to write.
            continue
        if data_vdb is None:
            # Reset lazily, on the first chunk that actually has something to
            # write. Resetting up front would delete the live catalog and then
            # spend minutes embedding, so a failed embed — a bad key, a dead
            # endpoint — would leave retrieval with nothing instead of with the
            # previous run's rows.
            data_vdb = get_data_vdb(database_name=database_name, reset=True)
        rows_written += data_vdb.run(records)

    if rows_written:
        logger.info("Tabular ingest result: %d rows written to pgvector", rows_written)
    else:
        # Previously silent, which read exactly like a successful embed. An empty
        # result is normal for a catalog that has not changed, but it is also what
        # a broken embed step produces, so it is worth a line either way.
        logger.info(
            "No embedding rows produced for database %s — nothing written to "
            "pgvector (expected when the catalog is unchanged)",
            database_name,
        )

    # Only reached when every step above succeeded: each one raises rather than
    # returning a failure, and nothing here catches. So this line means the run
    # is genuinely complete, not merely over.
    logger.info(
        "Data ingestion finished successfully for database %s — "
        "%d table(s), %d column(s), %d embedding row(s) in %.1fs",
        database_name,
        _row_count(tables_df),
        _row_count(columns_df),
        rows_written,
        time.monotonic() - started,
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
