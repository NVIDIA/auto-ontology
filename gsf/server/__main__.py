# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FastAPI application entry-point — middleware, routers, lifespan."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from nemo_retriever.tabular_data.neo4j import neo4j_connection
import uvicorn
from gsf.env import load_env

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

load_env()

from gsf.server.chat.router import router as chat_router  # noqa: E402
from gsf.server.chat.worker import get_pool, shutdown_pool  # noqa: E402
from gsf.server.connections.router import router as connections_router  # noqa: E402
from gsf.server.datasources.router import router as datasources_router  # noqa: E402
from gsf.server.health.router import router as health_router  # noqa: E402
from gsf.server.zones.router import router as zones_router  # noqa: E402


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Kick off the chat worker pool's initial warm spawn at boot. The pool
    # itself is lazy on first use, but eagerly initialising here means the
    # very first chat request doesn't pay a cold start.
    get_pool()
    try:
        yield
    finally:
        # Tear down warm subprocesses before exiting so we don't leak them.
        shutdown_pool()
        if neo4j_connection._conn is not None:
            neo4j_connection._conn.close()
            neo4j_connection._conn = None


def main() -> None:
    app = FastAPI(title="GSF API", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://127.0.0.1:3000",
            "http://localhost:3000",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(
        datasources_router, prefix="/api", tags=["datasources", "connectors"]
    )
    app.include_router(connections_router, prefix="/api", tags=["connections"])
    app.include_router(chat_router, prefix="/api", tags=["chat"])
    app.include_router(health_router, prefix="/api", tags=["health"])
    app.include_router(zones_router, prefix="/api", tags=["zones"])

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=3001,
        log_config=None,
    )


if __name__ == "__main__":
    main()
