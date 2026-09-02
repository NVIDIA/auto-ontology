# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Seed the local docker-compose Postgres with the demo catalogs.

Creates each demo database if it doesn't exist, then applies its SQL from
``dev_tools/sql/``. Idempotent — safe to re-run.

``pagila`` is the Postgres fixture: two schemas, views, a materialized view, a
partitioned table, arrays, an enum and a tsvector column — see
``dev_tools/sql/README.md`` for why each of those matters.

The SQLite fixture (Chinook) is built by ``dev_tools.build_sqlite_fixtures``;
run both, or just ``dev_tools.seed_fixtures`` which calls each in turn.

Usage::

    docker compose up -d postgres
    uv run --no-sync python -m dev_tools.seed_local_postgres
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import psycopg

logger = logging.getLogger("dev_tools.seed_local_postgres")

DEFAULT_POSTGRES_HOST = "localhost"
DEFAULT_POSTGRES_PORT = 5432
DEFAULT_POSTGRES_ADMIN_DB = "postgres"

SQL_DIR = Path(__file__).resolve().parent / "sql"


@dataclass(frozen=True)
class Fixture:
    """A demo database and the SQL that builds it.

    ``sentinel`` is a table whose presence means the fixture is already loaded.
    Needed because ``pagila.sql`` is ``pg_dump`` output, which is not
    re-runnable — replaying it against a populated database fails on the first
    ``CREATE TABLE``. Checking a sentinel keeps the script re-runnable without
    dropping data someone may have already ingested against.
    """

    name: str
    sentinel: str
    files: tuple[str, ...] = field(default=())

    @property
    def sql_files(self) -> tuple[Path, ...]:
        return tuple(SQL_DIR / f for f in (self.files or (f"{self.name}.sql",)))


FIXTURES: tuple[Fixture, ...] = (
    Fixture(
        name="pagila",
        sentinel="film",
        files=("pagila.sql", "pagila_analytics.sql"),
    ),
)

# Kept for callers that only want the names.
DATABASES: tuple[str, ...] = tuple(f.name for f in FIXTURES)


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


def _already_seeded(conn: psycopg.Connection, sentinel: str) -> bool:
    """Whether *sentinel* already exists in any user schema of *conn*."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT 1
            FROM information_schema.tables
            WHERE table_name = %s
              AND table_schema NOT IN ('pg_catalog', 'information_schema')
            LIMIT 1
            """,
            (sentinel,),
        )
        return cur.fetchone() is not None


def _apply_sql(fixture: Fixture) -> None:
    """Apply *fixture*'s SQL files in order, unless it is already loaded."""
    for path in fixture.sql_files:
        if not path.is_file():
            raise FileNotFoundError(f"Fixture SQL missing: {path}")

    with psycopg.connect(**_conn_params(fixture.name)) as conn:
        if _already_seeded(conn, fixture.sentinel):
            logger.info(
                "Database %s already has %s; skipping (drop the database to "
                "re-seed from scratch).",
                fixture.name,
                fixture.sentinel,
            )
            return

        for path in fixture.sql_files:
            logger.info("Applying %s to %s", path.name, fixture.name)
            with conn.cursor() as cur:
                cur.execute(path.read_text())
        conn.commit()


def seed() -> None:
    admin_params = _conn_params(
        os.environ.get("POSTGRES_ADMIN_DB", DEFAULT_POSTGRES_ADMIN_DB),
    )
    admin_conn = psycopg.connect(**admin_params)
    admin_conn.autocommit = True
    try:
        for fixture in FIXTURES:
            _ensure_database(admin_conn, fixture.name)
    finally:
        admin_conn.close()

    for fixture in FIXTURES:
        _apply_sql(fixture)

    logger.info("Seed complete: %s", ", ".join(f.name for f in FIXTURES))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    seed()
