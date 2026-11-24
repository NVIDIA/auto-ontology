# SPDX-FileCopyrightText: Copyright (c) 2026, AkilsuryaS.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify import through the real gateway handler, FastAPI, and local Postgres.

Run with POSTGRES_* pointing to a migrated disposable database and frontend
Node dependencies installed. A temporary catalog and local backend listener
are created and cleaned up; no model/embedding credentials are required.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import subprocess
import threading
import time
import uuid

from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from sqlalchemy import select
import uvicorn
import yaml

from auto_ontology.dal import model_interchange as dal
from auto_ontology.dal import schema as s
from auto_ontology.dal.session import store
from auto_ontology.server.model_interchange.router import router
from auto_ontology.server.model_interchange.service import import_model


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    name = f"issue267-gateway-{uuid.uuid4().hex[:8]}"
    payload = {
        "data_layer": {
            "databases": [
                {
                    "id": name,
                    "schemas": [
                        {
                            "id": f"{name}-schema",
                            "name": "public",
                            "database_name": name,
                            "tables": [
                                {
                                    "id": f"{name}-orders",
                                    "name": "orders",
                                    "columns": [
                                        {"id": f"{name}-order-id", "name": "id"},
                                        {
                                            "id": f"{name}-customer-ref",
                                            "name": "customer_id",
                                        },
                                    ],
                                },
                                {
                                    "id": f"{name}-customers",
                                    "name": "customers",
                                    "columns": [
                                        {"id": f"{name}-customer-id", "name": "id"},
                                    ],
                                },
                            ],
                        }
                    ],
                }
            ]
        },
    }
    summary = import_model(yaml.safe_dump(payload), replace=False, embed=False)
    db_ids = summary["database_ids"]
    listener = None
    server = None
    thread = None
    try:
        app = FastAPI()
        app.include_router(router, prefix="/api")

        def fixture_state() -> dict:
            schemas = select(s.catalog_schema.c.id).where(
                s.catalog_schema.c.database_id.in_(db_ids)
            )
            tables = select(s.catalog_table.c.id).where(
                s.catalog_table.c.schema_id.in_(schemas)
            )
            return jsonable_encoder(
                {
                    "databases": store().query_read(
                        select(s.catalog_database)
                        .where(s.catalog_database.c.id.in_(db_ids))
                        .order_by(s.catalog_database.c.id)
                    ),
                    "schemas": store().query_read(
                        select(s.catalog_schema)
                        .where(s.catalog_schema.c.id.in_(schemas))
                        .order_by(s.catalog_schema.c.id)
                    ),
                    "tables": store().query_read(
                        select(s.catalog_table)
                        .where(s.catalog_table.c.id.in_(tables))
                        .order_by(s.catalog_table.c.id)
                    ),
                    "columns": store().query_read(
                        select(s.catalog_column)
                        .where(s.catalog_column.c.table_id.in_(tables))
                        .order_by(s.catalog_column.c.id)
                    ),
                }
            )

        # Loopback-only acceptance server, not a production endpoint. Every 409
        # case checks full row state before the later successful import can mask it.
        app.get("/api/__fixture_state")(fixture_state)
        before = fixture_state()
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="off"))
        thread = threading.Thread(
            target=server.run, kwargs={"sockets": [listener]}, daemon=True
        )
        document = dal.assemble_export_document(
            dal.fetch_export_rows(db_ids),
            dialect_by_db_name={name: "postgres"},
            sql_column_resolver=lambda _sql, _db: [],
        )
        bound_yaml = yaml.safe_dump(document.model_dump())
        for db in document.data_layer.databases:
            db.id = f"foreign-{db.id}"
            for schema in db.schemas:
                schema.id = f"foreign-{schema.id}"
                for table in schema.tables:
                    table.id = f"foreign-{table.id}"
                    for column in table.columns:
                        column.id = f"foreign-{column.id}"
        thread.start()
        deadline = time.monotonic() + 10
        while not server.started:
            if not thread.is_alive() or time.monotonic() > deadline:
                raise RuntimeError("The local acceptance-test backend did not start")
            time.sleep(0.01)
        subprocess.run(
            [
                "node",
                "node_modules/vitest/vitest.mjs",
                "run",
                "--config",
                "tests/model-import.config.ts",
            ],
            cwd=root / "frontend",
            check=True,
            timeout=60,
            env={
                **os.environ,
                "PYTHON_API_URL": f"http://127.0.0.1:{port}",
                "MODEL_IMPORT_CONFLICT_YAML": yaml.safe_dump(document.model_dump()),
                "MODEL_IMPORT_BOUND_YAML": bound_yaml,
                "MODEL_IMPORT_FIXTURE_STATE": json.dumps(before),
            },
        )
        assert (
            len(
                store().query_read(
                    select(s.catalog_database.c.id).where(
                        s.catalog_database.c.id.in_(db_ids)
                    )
                )
            )
            == 1
        )
    finally:
        if server is not None:
            server.should_exit = True
        if thread is not None and thread.is_alive():
            thread.join(timeout=10)
        if listener is not None:
            listener.close()
        store().query_write(
            s.catalog_database.delete().where(s.catalog_database.c.id.in_(db_ids))
        )


if __name__ == "__main__":
    main()
