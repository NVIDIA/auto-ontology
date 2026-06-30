# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ingestion service.

Serves a FastAPI app exposing ingest and delete-ingest endpoints, and runs
``ingest()`` once at startup, then every 24h measured from the previous run —
independent of how long each run takes. The recurring sweep runs as a
background task started by the app lifespan.

Reloads connections from Neo4j on every pass so newly added connections are
picked up without restarting this process.

Usage::

    uv run --no-sync python -m gsf.ingestion_service
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

import uvicorn
from fastapi import FastAPI

from gsf.env import load_env

load_env()

from gsf.connectors.connection_string_factory import (  # noqa: E402
    build_connection_string,
)
from gsf.ingestion_service.ingest import run_ingest  # noqa: E402
from gsf.ingestion_service.router import router  # noqa: E402
from gsf.dal.connections import list_connections  # noqa: E402

logger = logging.getLogger(__name__)

# Recurring sweep cadence; every 24 hours, measured from the previous run
# rather than a fixed wall-clock time.
INGEST_INTERVAL = timedelta(hours=24)


async def ingest() -> None:
    """Run one ingestion pass for all configured connections."""
    raw = os.environ.get("CONNECTION_STRINGS", "")
    connections = [cs.strip() for cs in raw.split(",") if cs.strip()]

    if connections:
        logger.info("ingest: using connections from CONNECTION_STRINGS")
    else:
        try:
            connections = [build_connection_string(conn) for conn in list_connections()]
        except Exception:
            logger.exception("ingest: failed to load connections from Neo4j")
            connections = []

    if not connections:
        logger.info(
            "ingest: no connections configured. "
            "Add a connection in Settings → Connections or set CONNECTION_STRINGS in your .env."
        )
        return

    logger.info("ingest: starting (%s connection(s))", len(connections))
    for connection_string in connections:
        try:
            await asyncio.to_thread(run_ingest, connection_string)
        except Exception:
            logger.exception("ingest: failed for connection %s", connection_string)
    logger.info("ingest: finished")


async def _wait_until(next_run: datetime, stop: asyncio.Event) -> None:
    """Sleep until ``next_run`` in bounded chunks, returning early if stopped."""
    _WAIT_CHUNK_SECONDS = 500
    while not stop.is_set():
        remaining = (next_run - datetime.now(timezone.utc)).total_seconds()
        if remaining <= 0:
            return
        try:
            await asyncio.wait_for(
                stop.wait(), timeout=min(remaining, _WAIT_CHUNK_SECONDS)
            )
        except asyncio.TimeoutError:
            continue


async def _run_forever(stop: asyncio.Event) -> None:
    # Run once at startup so a fresh deploy refreshes existing connections
    # without waiting for the next scheduled slot.
    try:
        await ingest()
    except Exception:
        logger.exception("ingest: unhandled error; will retry on next tick")

    while not stop.is_set():
        # Schedule the next run one interval out from "now", so a long ingest
        # naturally pushes the next one back instead of burst-firing to catch up.
        next_run = datetime.now(timezone.utc) + INGEST_INTERVAL
        logger.info("ingest: next run at %s", next_run.isoformat())
        await _wait_until(next_run, stop)
        if stop.is_set():
            break

        try:
            await ingest()
        except Exception:
            logger.exception("ingest: unhandled error; will retry on next tick")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # uvicorn owns SIGINT/SIGTERM; the lifespan drives the scheduler's stop
    # event so an in-flight wait exits promptly on shutdown.
    stop = asyncio.Event()
    task = asyncio.create_task(_run_forever(stop))
    try:
        yield
    finally:
        stop.set()
        await task
        logger.info("ingestion_service: shutting down")


app = FastAPI(title="GSF Ingestion Service", lifespan=lifespan)
app.include_router(router)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("INGESTION_PORT", "3002")),
        log_config=None,
    )


if __name__ == "__main__":
    main()
