# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FastAPI application entry-point — middleware, routers, lifespan."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from auto_ontology.env import load_env

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

load_env()

logger = logging.getLogger(__name__)

from auto_ontology.version import get_app_version  # noqa: E402
from auto_ontology.dal import close_store  # noqa: E402
from auto_ontology.dal.schema_version import require_current_schema  # noqa: E402
from auto_ontology.server.chat.router import router as chat_router  # noqa: E402
from auto_ontology.server.chat.worker import get_pool, shutdown_pool  # noqa: E402
from auto_ontology.server.connections.router import router as connections_router  # noqa: E402
from auto_ontology.server.datasources.router import router as datasources_router  # noqa: E402
from auto_ontology.server.exploration.router import router as exploration_router  # noqa: E402
from auto_ontology.server.health.router import router as health_router  # noqa: E402
from auto_ontology.server.metadata.router import router as metadata_router  # noqa: E402
from auto_ontology.server.rules.router import router as rules_router  # noqa: E402
from auto_ontology.server.semantic_compilation.router import (  # noqa: E402
    router as semantic_compilation_router,
)
from auto_ontology.server.sql_attributes.router import router as sql_attributes_router  # noqa: E402
from auto_ontology.server.zones.router import router as zones_router  # noqa: E402
from auto_ontology.server.tags.router import router as tags_router  # noqa: E402
from auto_ontology.server.terms.router import router as terms_router  # noqa: E402
from auto_ontology.server.model_interchange.router import (  # noqa: E402
    router as model_interchange_router,
)
from auto_ontology.server.search.router import router as search_router  # noqa: E402


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Before anything else: refuse to serve against a schema this build does not
    # expect. Here rather than in `create_app` deliberately — the OpenAPI spec
    # generator builds the app with no database at all (see that docstring), so
    # requiring one to construct it would break the spec check in CI. The
    # lifespan runs only when the app is actually served, which covers both
    # documented commands: `python -m auto_ontology.server` and `uvicorn ... --factory`.
    require_current_schema()

    # Kick off the chat worker pool's initial warm spawn at boot. The pool
    # itself is lazy on first use, but eagerly initialising here means the
    # very first chat request doesn't pay a cold start.
    get_pool()
    try:
        yield
    finally:
        # Tear down warm subprocesses before exiting so we don't leak them.
        shutdown_pool()
        close_store()


def create_app() -> FastAPI:
    """Build the API app without serving it.

    Kept separate from ``main`` so tooling — the OpenAPI spec generator,
    tests — can construct the app and read ``app.openapi()`` without
    starting uvicorn.
    """
    app = FastAPI(title="Auto Ontology API", lifespan=lifespan)

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
    app.include_router(exploration_router, prefix="/api", tags=["exploration"])
    app.include_router(connections_router, prefix="/api", tags=["connections"])
    app.include_router(chat_router, prefix="/api", tags=["chat"])
    app.include_router(metadata_router, prefix="/api", tags=["metadata"])
    app.include_router(health_router, prefix="/api", tags=["health"])
    app.include_router(sql_attributes_router, prefix="/api", tags=["sql-attributes"])
    app.include_router(
        semantic_compilation_router, prefix="/api", tags=["semantic-compilation"]
    )
    app.include_router(zones_router, prefix="/api", tags=["zones"])
    app.include_router(tags_router, prefix="/api", tags=["tags"])
    app.include_router(rules_router, prefix="/api", tags=["rules"])
    app.include_router(terms_router, prefix="/api", tags=["terms"])
    app.include_router(
        model_interchange_router, prefix="/api", tags=["model-interchange"]
    )
    app.include_router(search_router, prefix="/api", tags=["search"])

    return app


def main() -> None:
    logger.info("Starting Auto Ontology API server — app version %s", get_app_version())

    app = create_app()

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=3001,
        log_config=None,
    )


if __name__ == "__main__":
    main()
