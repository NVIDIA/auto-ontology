# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A per-call store must give its connections back when the call ends (#553).

A catalog build creates a ``PostgresVDB`` per table, on worker threads, and drops
it. Dropped there, the store's pool was not disposed: 40 such calls from 8
threads left 22 idle connections open after the threads had exited, and two
builds sharing a server exhausted ``max_connections``. Using the store as a
context manager closes it deterministically, so the count returns to where it
started.
"""

from __future__ import annotations

import uuid
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import psycopg
import pytest

from auto_ontology.infra.postgres import get_postgres_connection_string
from auto_ontology.vdb import VDB_SCHEMA
from auto_ontology.vdb.postgres import PostgresVDB
from auto_ontology.vdb.records import write_embedded_rows

_DIMENSIONS = 2048
_CALLS = 24
_WORKERS = 8


def _client_connections(conn: psycopg.Connection) -> int:
    row = conn.execute(
        "SELECT count(*) FROM pg_stat_activity "
        "WHERE datname = current_database() AND backend_type = 'client backend' "
        "AND pid <> pg_backend_pid()"
    ).fetchone()
    return int(row[0]) if row else 0


@pytest.fixture
def probe():
    """A throwaway collection, and a connection to count the others with."""
    try:
        conn = psycopg.connect(
            get_postgres_connection_string(), connect_timeout=3, autocommit=True
        )
    except Exception as exc:  # noqa: BLE001 -- no database, nothing to assert
        pytest.skip(f"postgres not reachable: {exc}")
    collection = f"pool_probe_{uuid.uuid4().hex[:8]}"
    try:
        yield conn, collection
    finally:
        conn.execute(f'DROP TABLE IF EXISTS {VDB_SCHEMA}."{collection}"')
        conn.close()


def test_closing_each_store_returns_its_connections(probe) -> None:
    conn, collection = probe

    def one_call(index: int) -> None:
        with PostgresVDB(
            connection_string=get_postgres_connection_string(),
            collection_name=collection,
            schema_name=VDB_SCHEMA,
        ) as vdb:
            node_id = f"probe-{index}"
            write_embedded_rows(
                vdb,
                [
                    {
                        "text": "pool lifecycle probe",
                        "metadata": {
                            "id": node_id,
                            "label": "Column",
                            "embedding": [0.001] * _DIMENSIONS,
                        },
                    }
                ],
            )
            vdb.delete_by_id(node_id)

    # The first write creates the collection; doing that concurrently races.
    one_call(-1)
    before = _client_connections(conn)
    with ThreadPoolExecutor(_WORKERS) as pool:
        list(pool.map(one_call, range(_CALLS)))

    assert _client_connections(conn) == before


def test_the_context_manager_closes_the_store() -> None:
    vdb = PostgresVDB(connection_string="postgresql://unused")
    vdb.close = MagicMock()  # type: ignore[method-assign]

    with vdb as entered:
        assert entered is vdb
        vdb.close.assert_not_called()

    vdb.close.assert_called_once_with()
