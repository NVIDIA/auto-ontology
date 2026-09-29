# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ingestion service.

Serves a FastAPI app exposing ingest and delete-ingest endpoints, and runs two
background schedulers via the app lifespan:

* :class:`DataScheduler` — data ingestion, triggerable via ``POST /ingest``.
* :class:`SemanticScheduler` — semantic compilation over the catalog that
  :class:`DataScheduler` writes to the store, triggerable via
  ``POST /semantic/compile``. Its first pass waits for
  :class:`DataScheduler`'s first pass to finish (see ``semantic_scheduler.py``)
  so the two, which both start at the same moment on boot, can't race.

Each runs once at startup, then every 24h measured from the previous run —
independent of how long each run takes. Connections are reloaded from the
catalog on every pass so newly added ones are picked up without restarting.

Usage::

    uv run --no-sync python -m auto_ontology.ingestion_service
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from auto_ontology.env import load_env

load_env()

from auto_ontology.dal.schema_version import require_current_schema  # noqa: E402
from auto_ontology.ingestion_service.config import (  # noqa: E402
    is_semantic_compilation_enabled,
)
from auto_ontology.ingestion_service.data_scheduler import DataScheduler  # noqa: E402
from auto_ontology.ingestion_service.router import router  # noqa: E402
from auto_ontology.ingestion_service.semantic_scheduler import SemanticScheduler  # noqa: E402
from auto_ontology.version import get_app_version  # noqa: E402

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Same gate as the API server: this service writes the catalog, so against a
    # schema it does not expect every ingest fails partway through instead of
    # not starting. Worse here than there, because a scheduler runs unattended —
    # nobody is watching a request come back 500, so the failure surfaces as an
    # empty catalog hours later.
    require_current_schema()

    # uvicorn owns SIGINT/SIGTERM; each scheduler exposes a stop() that unwinds
    # its in-flight wait promptly on shutdown. Schedulers are stored on
    # app.state so the router can trigger them on demand.
    data_scheduler = DataScheduler()
    # Waits for the ingest scheduler's first pass before compiling (see
    # semantic_scheduler.py) so the two starting at the same moment on boot
    # can't race — otherwise the very first compile could run against a
    # catalog that ingest hasn't written yet.
    semantic_scheduler = SemanticScheduler(depends_on=data_scheduler)
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


app = FastAPI(title="Auto Ontology Ingestion Service", lifespan=lifespan)
app.include_router(router)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger.info(
        "Starting Auto Ontology Ingestion Service — app version %s", get_app_version()
    )
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("INGESTION_PORT", "3002")),
        log_config=None,
    )


if __name__ == "__main__":
    main()
