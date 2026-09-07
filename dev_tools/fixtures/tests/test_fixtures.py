# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Assert the fixture databases have the shapes the refactor's tests rely on.

Every claim in ``dev_tools/fixtures/sql/README.md`` about what Pagila and Chinook cover
is load-bearing: golden captures, the ``find_join_path`` suite, and the
schema-scoped zone tests are all written against these shapes. Three of the
original assumptions turned out to be wrong when first checked, which is why
they are asserted here rather than described in prose and trusted.

The Postgres half needs a live server and is skipped without one. The SQLite
half needs nothing — it builds the database from vendored SQL in-process.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest

from dev_tools.fixtures import build_sqlite_fixtures

SQL_DIR = Path(__file__).resolve().parents[1] / "sql"


# --------------------------------------------------------------------------
# Chinook (SQLite) — no server required
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def chinook() -> sqlite3.Connection:
    db = build_sqlite_fixtures.build(build_sqlite_fixtures.FIXTURES[0])
    con = sqlite3.connect(db)
    yield con
    con.close()


def test_chinook_builds_from_vendored_sql(chinook: sqlite3.Connection) -> None:
    tables = [
        r[0]
        for r in chinook.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
    ]
    assert len(tables) == 11, tables


def test_chinook_has_a_self_referencing_fk(chinook: sqlite3.Connection) -> None:
    """``Employee.ReportsTo`` — Pagila's ``public`` schema has no self-reference.

    ``find_join_path``'s cycle-termination test needs one that lives in the
    same schema as its target.
    """
    fks = list(chinook.execute('PRAGMA foreign_key_list("Employee")'))
    assert any(fk[2] == "Employee" for fk in fks), fks


def test_chinook_has_enough_rows_for_sample_values(
    chinook: sqlite3.Connection,
) -> None:
    """``store_column_sample_values`` is meaningless on a handful of rows."""
    assert chinook.execute("SELECT count(*) FROM Track").fetchone()[0] > 3000


# --------------------------------------------------------------------------
# Pagila (Postgres) — needs a live server
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pagila():
    psycopg = pytest.importorskip("psycopg")
    if not os.environ.get("POSTGRES_USER"):
        pytest.skip("POSTGRES_* not set; run dev_tools.fixtures.seed_fixtures first")
    try:
        conn = psycopg.connect(
            host=os.environ.get("POSTGRES_HOST", "localhost"),
            port=os.environ.get("POSTGRES_PORT", "5432"),
            user=os.environ["POSTGRES_USER"],
            password=os.environ["POSTGRES_PASSWORD"],
            dbname="pagila",
            connect_timeout=3,
        )
    except Exception as exc:  # noqa: BLE001 — any failure means "no fixture DB"
        pytest.skip(f"pagila not reachable: {exc}")
    yield conn
    conn.close()


def _scalar(conn, sql: str):
    with conn.cursor() as cur:
        cur.execute(sql)
        return cur.fetchone()[0]


def test_pagila_has_views_and_a_materialized_view(pagila) -> None:
    """``table_type`` is only ever exercised as ``base table`` without these."""
    assert (
        _scalar(
            pagila,
            "SELECT count(*) FROM information_schema.views WHERE table_schema='public'",
        )
        == 7
    )
    assert (
        _scalar(pagila, "SELECT count(*) FROM pg_matviews WHERE schemaname='public'")
        == 1
    )


def test_pagila_has_a_partitioned_table(pagila) -> None:
    """``payment`` is partitioned; its children surface as tables."""
    assert _scalar(pagila, "SELECT count(*) FROM pg_class WHERE relkind='p'") >= 1
    assert (
        _scalar(
            pagila,
            "SELECT count(*) FROM information_schema.tables "
            "WHERE table_schema='public' AND table_type='BASE TABLE'",
        )
        == 22
    )


def test_pagila_has_two_schemas(pagila) -> None:
    """The Schema tier and the ``-Schema`` exclusion are untestable with one."""
    with pagila.cursor() as cur:
        cur.execute(
            "SELECT DISTINCT table_schema FROM information_schema.tables "
            "WHERE table_schema IN ('public','analytics') ORDER BY 1"
        )
        assert [r[0] for r in cur.fetchall()] == ["analytics", "public"]


def test_pagila_has_cross_schema_foreign_keys(pagila) -> None:
    """Join paths that cross a schema boundary."""
    assert (
        _scalar(
            pagila,
            """
            SELECT count(*) FROM pg_constraint c
            JOIN pg_class r ON r.oid=c.conrelid
            JOIN pg_namespace rn ON rn.oid=r.relnamespace
            JOIN pg_class f ON f.oid=c.confrelid
            JOIN pg_namespace fn ON fn.oid=f.relnamespace
            WHERE c.contype='f' AND rn.nspname <> fn.nspname
            """,
        )
        == 2
    )


def test_pagila_has_a_self_referencing_fk_outside_public(pagila) -> None:
    """``analytics.category_rollup.parent_rollup_id``.

    Upstream Pagila has no self-referencing FK anywhere — the plan originally
    claimed ``staff.reports_to``, which does not exist.
    """
    assert (
        _scalar(
            pagila,
            "SELECT count(*) FROM pg_constraint WHERE contype='f' "
            "AND conrelid=confrelid",
        )
        == 1
    )
    assert (
        _scalar(
            pagila,
            "SELECT count(*) FROM analytics.category_rollup "
            "WHERE parent_rollup_id IS NOT NULL",
        )
        > 0
    )


def test_pagila_has_non_scalar_column_types(pagila) -> None:
    """Arrays, an enum and a tsvector — ``data_type`` handling beyond scalars."""
    assert (
        _scalar(
            pagila,
            "SELECT count(*) FROM information_schema.columns "
            "WHERE table_schema='public' AND data_type='ARRAY'",
        )
        >= 1
    )
    assert (
        _scalar(
            pagila,
            "SELECT count(*) FROM information_schema.columns "
            "WHERE table_schema='public' AND udt_name='tsvector'",
        )
        >= 1
    )
    assert (
        _scalar(pagila, "SELECT count(*) FROM pg_type WHERE typname='mpaa_rating'") == 1
    )


def test_pagila_multi_hop_join_chain_is_intact(pagila) -> None:
    """``customer -> address -> city -> country``, for ``find_join_path``."""
    with pagila.cursor() as cur:
        cur.execute(
            """
            SELECT conrelid::regclass::text || '->' || confrelid::regclass::text
            FROM pg_constraint
            WHERE contype='f'
              AND conrelid::regclass::text IN ('customer','address','city')
            ORDER BY 1
            """
        )
        edges = {r[0] for r in cur.fetchall()}
    assert {"customer->address", "address->city", "city->country"} <= edges, edges


def test_pagila_trim_left_no_orphans(pagila) -> None:
    """The trim deleted payments before rentals; nothing dangles."""
    assert (
        _scalar(
            pagila,
            "SELECT count(*) FROM payment p LEFT JOIN rental r "
            "ON r.rental_id=p.rental_id WHERE r.rental_id IS NULL",
        )
        == 0
    )
    assert (
        _scalar(
            pagila,
            "SELECT count(*) FROM rental r LEFT JOIN inventory i "
            "ON i.inventory_id=r.inventory_id WHERE i.inventory_id IS NULL",
        )
        == 0
    )


def test_pagila_has_enough_rows_for_uniqueness_detection(pagila) -> None:
    assert _scalar(pagila, "SELECT count(*) FROM film") == 1000
    assert _scalar(pagila, "SELECT count(*) FROM inventory") > 4000
