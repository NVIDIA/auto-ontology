# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the SQLite fixture databases from their vendored SQL.

Chinook is the second fixture database. It exists to make the fixture set
multi-*database* and multi-*dialect*, which is what several behaviours need in
order to be testable at all: cross-database rejection in ``find_join_path``,
``delete_by_database`` scoping, per-database pgvector collection resets, and
zones spanning databases. It also carries a self-referencing foreign key
(``Employee.ReportsTo``), which Pagila has nowhere.

Only the ``.sql`` is vendored — the ``.sqlite`` binary is generated here, so git
stores diffable text rather than a 1 MB blob that changes wholesale on any edit.
Uses the ``sqlite3`` standard library, so there is no dependency on the
``sqlite3`` CLI being installed.

Usage::

    uv run --no-sync python -m dev_tools.build_sqlite_fixtures
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("dev_tools.build_sqlite_fixtures")

SQL_DIR = Path(__file__).resolve().parent / "sql"


@dataclass(frozen=True)
class SqliteFixture:
    name: str
    sentinel: str


FIXTURES: tuple[SqliteFixture, ...] = (SqliteFixture(name="chinook", sentinel="Track"),)


def _has_table(path: Path, table: str) -> bool:
    if not path.is_file():
        return False
    with sqlite3.connect(path) as con:
        row = con.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
            (table,),
        ).fetchone()
    return row is not None


def build(fixture: SqliteFixture, *, force: bool = False) -> Path:
    """Materialise ``<name>.sqlite`` from ``<name>.sql``. Idempotent."""
    sql_path = SQL_DIR / f"{fixture.name}.sql"
    db_path = SQL_DIR / f"{fixture.name}.sqlite"

    if not sql_path.is_file():
        raise FileNotFoundError(f"Fixture SQL missing: {sql_path}")

    if not force and _has_table(db_path, fixture.sentinel):
        logger.info("%s already built; skipping.", db_path.name)
        return db_path

    db_path.unlink(missing_ok=True)
    logger.info("Building %s from %s", db_path.name, sql_path.name)

    # The upstream script is UTF-8 with a BOM; utf-8-sig strips it so the first
    # statement doesn't arrive with a stray ﻿ and fail to parse.
    script = sql_path.read_text(encoding="utf-8-sig")
    con = sqlite3.connect(db_path)
    try:
        con.executescript(script)
        con.commit()
    finally:
        con.close()

    return db_path


def build_all(*, force: bool = False) -> list[Path]:
    return [build(fixture, force=force) for fixture in FIXTURES]


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    for built in build_all():
        logger.info("Ready: %s (%s bytes)", built, f"{built.stat().st_size:,}")
