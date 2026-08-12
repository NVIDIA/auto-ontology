# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ingestion service.

Serves a FastAPI app exposing ingest and delete-ingest endpoints, and runs two
background schedulers via the app lifespan:

* :class:`DataScheduler` — data ingestion (which also runs semantic compilation
  at the end of each ingest).
* :class:`SemanticScheduler` — semantic compilation on its own, triggerable via
  ``POST /semantic/compile``.

Each runs once at startup, then every 24h measured from the previous run —
independent of how long each run takes. Connections are reloaded from the catalog on
every pass so newly added connections are picked up without restarting.

Usage::

    uv run --no-sync python -m gsf.ingestion_service
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from gsf.env import load_env

load_env()

from gsf.ingestion_service.config import (  # noqa: E402
    is_semantic_compilation_enabled,
)
from gsf.ingestion_service.data_scheduler import DataScheduler  # noqa: E402
from gsf.ingestion_service.router import router  # noqa: E402
from gsf.ingestion_service.semantic_scheduler import SemanticScheduler  # noqa: E402
from gsf.version import get_app_version  # noqa: E402

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # uvicorn owns SIGINT/SIGTERM; each scheduler exposes a stop() that unwinds
    # its in-flight wait promptly on shutdown. Schedulers are stored on
    # app.state so the router can trigger them on demand.
    data_scheduler = DataScheduler()
    semantic_scheduler = SemanticScheduler()
    app.state.data_scheduler = data_scheduler
    app.state.semantic_scheduler = semantic_scheduler

    data_scheduler.start()
    # Only start semantic compilation if it's enabled in settings. When a user
    # enables it later, the POST /semantic/compile trigger starts the scheduler
    # on demand, so no restart is needed.
    if is_semantic_compilation_enabled():
        semantic_scheduler.start()
        logger.info("semantic: compilation enabled; scheduler started")
    else:
        logger.info(
            "semantic: compilation disabled; scheduler idle "
            "(enable it in Settings → Semantic Compilation)"
        )
    try:
        yield
    finally:
        await data_scheduler.stop()
        await semantic_scheduler.stop()
        logger.info("ingestion_service: shutting down")


app = FastAPI(title="GSF Ingestion Service", lifespan=lifespan)
app.include_router(router)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger.info("Starting GSF Ingestion Service — app version %s", get_app_version())
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("INGESTION_PORT", "3002")),
        log_config=None,
    )


if __name__ == "__main__":
    main()
