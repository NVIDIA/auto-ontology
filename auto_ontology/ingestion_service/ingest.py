# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import logging
import threading
import time
from contextlib import contextmanager
from typing import Any, Iterator

from auto_ontology.utils import get_embed_params
from auto_ontology.utils.embedding import batch_embed_chunks
from auto_ontology.utils.embedding_rows import (
    CatalogEmbeddingRowsOp,
)
from nemo_retriever.common.vdb.records import to_client_vdb_records
from auto_ontology.connectors.base import SQLDatabase

from auto_ontology.catalog import ingest_catalog
from auto_ontology.vdb import get_data_vdb
from auto_ontology.connectors.registry import (
    get_connectors,
    invalidate_connectors_cache,
)
from auto_ontology.dal.reset import delete_all_data, delete_semantic_layer
from auto_ontology.ingestion_service.rules import replay_rules

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
    :func:`auto_ontology.connectors.registry.get_connectors`.

    Three straight-line steps, formerly a ``Graph()`` chain of three operators.
    Only the middle one is still a library operator; the catalog write is
    Auto Ontology-owned (:func:`auto_ontology.catalog.ingest_catalog`) and the embed step goes
    through :func:`auto_ontology.utils.embedding.batch_embed_chunks`.

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
            # Unconditional, unlike the ingest pass's own replay: neither
            # scheduler covers this path. It is a bare thread, and creating a
            # connection asks for an ingest without asking for a compilation
            # (see ``server/connections/service.py``), so nothing would label
            # the database that was just ingested until a scheduler tick hours
            # later. Runs even when compilation is on, which at worst repeats
            # work the next semantic pass would have done -- the replay is a
            # diff and writes nothing where the labels are already right.
            replay_rules("ingest")
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


# How long to wait for an in-flight compilation pass to stop before resetting
# anyway. A pass is cut short at a database boundary and the database in flight
# cannot be interrupted, so this has to exceed the slowest single-database
# compile; resetting underneath one would reintroduce the race this ordering
# exists to remove.
_ABORT_DRAIN_TIMEOUT_S = 1800
_ABORT_POLL_S = 0.5

# create_task keeps only a weak reference, so a bare task can be garbage
# collected mid-flight. Holding it here until it finishes is the documented
# workaround.
_background_tasks: set[asyncio.Task[None]] = set()

# One reset at a time. Each call runs as its own task, and the pause is a single
# boolean, so two overlapping resets would have the first to finish resume the
# scheduler while the second is still deleting — reopening the window the pause
# exists to close. Serialising also stops them deleting and rebuilding on top of
# each other: the second reset cancels the first's rebuild and starts its own,
# which is what asking for a reset twice should mean.
_reset_lock = asyncio.Lock()


async def _wait_until_idle(scheduler: Any) -> bool:
    """Block until *scheduler* is not mid-pass. False if it never stopped."""
    deadline = time.monotonic() + _ABORT_DRAIN_TIMEOUT_S
    while scheduler.running:
        if time.monotonic() > deadline:
            return False
        await asyncio.sleep(_ABORT_POLL_S)
    return True


async def reset_semantic_layer_and_recompile(
    scheduler: Any, database_name: str | None = None
) -> None:
    """Delete a semantic layer and rebuild it — strictly in that order.

    The ordering *is* the feature. These three steps used to overlap: the
    deletion ran on its own thread while a compilation pass ran on the
    scheduler's task, and both races lost data.

    A pass asks which tables still lack a Term *once*, at its start. Begin a
    pass while the delete is still in flight and it snapshots the pre-delete
    answer — every table still has a Term, so it compiles nothing, and the
    delete lands milliseconds later. Observed on a live store: the first
    database of a reset pass computed its work list 11ms before the delete and
    reported "0 table(s) processed", leaving that database wiped and not
    rebuilt. Symmetrically, an *old* pass still running during the delete keeps
    writing Terms that the delete has already swept past.

    So: stop the running pass, wait for it to actually stop, delete, and only
    then start the rebuild. Callers run this as a background task, which keeps
    the endpoint's non-blocking contract — the wait can be minutes, since the
    database in flight cannot be interrupted.

    Passing ``None`` resets the semantic layer of every database.
    """
    label = database_name or "<all>"
    async with _reset_lock:
        await _reset_semantic_layer_and_recompile(scheduler, database_name, label)


async def _reset_semantic_layer_and_recompile(
    scheduler: Any, database_name: str | None, label: str
) -> None:
    """Body of :func:`reset_semantic_layer_and_recompile`, one caller at a time."""
    # Paused for the whole stop-and-delete window, not just aborted. abort()
    # ends the current pass but leaves the loop ticking, and a pass started by a
    # timer tick or /semantic/compile between the drain and the end of the
    # delete would read the pre-delete catalog, find every table already
    # carrying a Term, compile nothing, and leave the layer deleted but not
    # rebuilt — the exact failure this ordering exists to prevent.
    scheduler.pause()
    try:
        scheduler.abort()
        if not await _wait_until_idle(scheduler):
            logger.warning(
                "reset-semantic %s: a compilation pass was still running after "
                "%ds; resetting anyway",
                label,
                _ABORT_DRAIN_TIMEOUT_S,
            )
        await asyncio.to_thread(delete_semantic_layer, database_name)
    except Exception:
        # Deliberately no rebuild on failure: a pass started now would compile
        # against a half-deleted layer and the result would be neither the old
        # one nor a clean one.
        logger.exception("Background reset-semantic failed for database %s", label)
        return
    finally:
        # Always, including the failure path — a paused scheduler that is never
        # resumed silently stops compiling for good.
        scheduler.resume()

    if not scheduler.start():
        scheduler.trigger()


def trigger_reset_semantic(scheduler: Any, database_name: str | None = None) -> None:
    """Schedule :func:`reset_semantic_layer_and_recompile` without blocking.

    Must be called from the event loop thread — ``scheduler.start()`` and
    ``trigger()`` touch asyncio primitives owned by that loop.
    """
    task = asyncio.create_task(
        reset_semantic_layer_and_recompile(scheduler, database_name)
    )
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
