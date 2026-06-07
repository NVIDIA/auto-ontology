# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Seed the local docker-compose Postgres with the demo catalog.

Creates the four demo databases if they don't exist, then applies each
database's DDL from ``dev_tools/sql/<db>.sql``. Idempotent — safe to re-run.

Usage::

    docker compose up -d postgres
    uv run --no-sync python -m dev_tools.seed_local_postgres
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import psycopg

logger = logging.getLogger("dev_tools.seed_local_postgres")

DEFAULT_POSTGRES_HOST = "localhost"
DEFAULT_POSTGRES_PORT = 5432
DEFAULT_POSTGRES_ADMIN_DB = "postgres"

SQL_DIR = Path(__file__).resolve().parent / "sql"

DATABASES: tuple[str, ...] = ("testdb",)


def _conn_params(db: str) -> dict[str, str]:
    return {
        "host": os.environ.get("POSTGRES_HOST", DEFAULT_POSTGRES_HOST),
        "port": os.environ.get("POSTGRES_PORT", str(DEFAULT_POSTGRES_PORT)),
        "user": os.environ["POSTGRES_USER"],
        "password": os.environ["POSTGRES_PASSWORD"],
        "dbname": db,
    }


def _ensure_database(admin_conn: psycopg.Connection, db: str) -> None:
    with admin_conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db,))
        if cur.fetchone():
            logger.info("Database %s already exists; skipping CREATE.", db)
            return
        logger.info("Creating database %s.", db)
        cur.execute(f'CREATE DATABASE "{db}"')


def _apply_ddl(db: str) -> None:
    ddl_path = SQL_DIR / f"{db}.sql"
    if not ddl_path.is_file():
        raise FileNotFoundError(f"DDL file missing: {ddl_path}")

    sql = ddl_path.read_text()
    logger.info("Applying DDL to %s from %s", db, ddl_path.name)
    with psycopg.connect(**_conn_params(db)) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)


def seed() -> None:
    admin_params = _conn_params(
        os.environ.get("POSTGRES_ADMIN_DB", DEFAULT_POSTGRES_ADMIN_DB),
    )
    admin_conn = psycopg.connect(**admin_params)
    admin_conn.autocommit = True
    try:
        for db in DATABASES:
            _ensure_database(admin_conn, db)
    finally:
        admin_conn.close()

    for db in DATABASES:
        _apply_ddl(db)

    logger.info("Seed complete.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    seed()
