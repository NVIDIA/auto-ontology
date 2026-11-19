# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A scoped delete can spare the vectors of rows that survive it (#499)."""

from __future__ import annotations

import json
import uuid

import psycopg
import pytest

from auto_ontology.infra.postgres import get_postgres_connection_string
from auto_ontology.vdb.postgres import PostgresVDB


@pytest.fixture
def vdb():
    """A collection with the promoted ``database_name`` column."""
    name = f"scoped_{uuid.uuid4().hex[:8]}"
    try:
        conn = psycopg.connect(get_postgres_connection_string(), connect_timeout=3)
    except Exception as exc:  # noqa: BLE001 -- no database, nothing to assert
        pytest.skip(f"postgres not reachable: {exc}")
    rows = [
        ("t-shared", "shop"),
        ("t-own", "shop"),
        ("a-own", "shop"),
        ("t-elsewhere", "warehouse"),
    ]
    with conn:
        with conn.cursor() as cur:
            cur.execute("CREATE SCHEMA IF NOT EXISTS vdb")
            cur.execute(
                f"CREATE TABLE vdb.{name} ("
                "  langchain_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),"
                "  content text, database_name text, langchain_metadata jsonb)"
            )
            cur.executemany(
                f"INSERT INTO vdb.{name} (content, database_name, langchain_metadata)"
                " VALUES (%s, %s, %s)",
                [(node_id, db, json.dumps({"id": node_id})) for node_id, db in rows],
            )
        conn.commit()
    try:
        yield PostgresVDB(
            connection_string=get_postgres_connection_string(),
            collection_name=name,
            schema_name="vdb",
        )
    finally:
        with psycopg.connect(get_postgres_connection_string()) as cleanup:
            with cleanup.cursor() as cur:
                cur.execute(f"DROP TABLE IF EXISTS vdb.{name}")
            cleanup.commit()


def _left(vdb: PostgresVDB) -> set[str]:
    with psycopg.connect(vdb.connection_string) as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT langchain_metadata ->> 'id' FROM vdb.{vdb.collection_name}"
            )
            return {row[0] for row in cur.fetchall()}


def test_lists_the_ids_a_database_tagged(vdb: PostgresVDB) -> None:
    assert sorted(vdb.metadata_ids_for_database("shop")) == [
        "a-own",
        "t-own",
        "t-shared",
    ]


def test_deletes_a_databases_rows_except_the_kept_ones(vdb: PostgresVDB) -> None:
    deleted = vdb.delete_by_database_keeping("shop", {"t-shared"})

    assert sorted(deleted) == ["a-own", "t-own"]
    assert _left(vdb) == {"t-shared", "t-elsewhere"}
