# SPDX-FileCopyrightText: Copyright (c) 2026, AkilsuryaS.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify import through the real gateway handler, FastAPI, and local Postgres.

Run with POSTGRES_* pointing to a migrated disposable database and frontend
Node dependencies installed. A temporary catalog and local backend listener
are created and cleaned up; no model/embedding credentials are required.
"""

from __future__ import annotations

import os
from pathlib import Path
import socket
import subprocess
import threading
import time
import uuid

from fastapi import FastAPI
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
    app = FastAPI()
    app.include_router(router, prefix="/api")
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="off"))
    thread = threading.Thread(
        target=server.run, kwargs={"sockets": [listener]}, daemon=True
    )
    try:
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
        server.should_exit = True
        if thread.is_alive():
            thread.join(timeout=10)
        listener.close()
        store().query_write(
            s.catalog_database.delete().where(s.catalog_database.c.id.in_(db_ids))
        )


if __name__ == "__main__":
    main()
