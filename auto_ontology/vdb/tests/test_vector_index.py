# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The collection's HNSW index must actually be reachable from the store's query.

Nothing fails when it isn't: the same rows come back, in the same order, just
slowly. Measured on 51,200 rows the difference was 175ms versus 0.36ms, and the
only place it shows up is a latency graph. So the assertion is on the *plan*.

Two ways to get this wrong, and both look fine until you read a plan:

* **Indexing the cast instead of the column.** pgvector caps HNSW at 2000
  dimensions for ``vector`` and the model emits 2048, so the tempting fix is an
  expression index on ``(embedding::halfvec(2048))``. Postgres matches an
  expression index syntactically, and ``PGVectorStore`` emits the plain
  ``embedding <=> $1`` -- which never matches it. Hence the column itself is
  ``halfvec``; see ``PostgresVDB._create_vector_index``.
* **Assuming the parameter's type has to match.** It does not: pgvector casts
  the *parameter* (``embedding <=> ($1)::halfvec``), which is free, and the
  index is still used.

``enable_seqscan = off`` makes the check independent of table size -- at the few
rows a fixture holds Postgres would prefer a sequential scan either way, and the
test would prove nothing.
"""

from __future__ import annotations

import uuid

import psycopg
import pytest

from auto_ontology.infra.postgres import get_postgres_connection_string

_DIMENSIONS = 2048


@pytest.fixture
def collection():
    """A collection shaped like the one ``init_vectorstore_table`` creates."""
    name = f"probe_{uuid.uuid4().hex[:8]}"
    try:
        conn = psycopg.connect(get_postgres_connection_string(), connect_timeout=3)
    except Exception as exc:  # noqa: BLE001 -- no database, nothing to assert
        pytest.skip(f"postgres not reachable: {exc}")

    with conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute("CREATE SCHEMA IF NOT EXISTS vdb")
            # `vector`, exactly as langchain_postgres creates it -- the ALTER to
            # halfvec is the thing under test, not a precondition of it.
            cur.execute(
                f"CREATE TABLE vdb.{name} ("
                "  langchain_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),"
                "  content text,"
                f" embedding vector({_DIMENSIONS}),"
                "  database_name varchar(100),"
                "  label varchar(100),"
                "  langchain_metadata jsonb)"
            )
        conn.commit()
    try:
        yield name
    finally:
        with psycopg.connect(get_postgres_connection_string()) as cleanup:
            with cleanup.cursor() as cur:
                cur.execute(f"DROP TABLE IF EXISTS vdb.{name}")
            cleanup.commit()


def _index_the_collection(name: str) -> None:
    """What ``PostgresVDB._create_vector_index`` does, against *name*."""
    with psycopg.connect(get_postgres_connection_string()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"ALTER TABLE vdb.{name} "
                f"ALTER COLUMN embedding TYPE halfvec({_DIMENSIONS})"
            )
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS ix_hnsw_{name}_embedding "
                f"ON vdb.{name} USING hnsw (embedding halfvec_cosine_ops)"
            )
        conn.commit()


def _plan(name: str) -> str:
    """The plan for the query ``PGVectorStore`` writes: a vector-typed parameter."""
    vector = "[" + ",".join("0.1" for _ in range(_DIMENSIONS)) + "]"
    with psycopg.connect(get_postgres_connection_string()) as conn:
        with conn.cursor() as cur:
            cur.execute("SET enable_seqscan = off")
            cur.execute(
                f"EXPLAIN SELECT langchain_id FROM vdb.{name} "
                f"ORDER BY embedding <=> '{vector}'::vector({_DIMENSIONS}) LIMIT 5"
            )
            return "\n".join(row[0] for row in cur.fetchall())


def test_the_plain_vector_column_cannot_be_indexed(collection) -> None:
    """The constraint the halfvec column exists to work around."""
    with psycopg.connect(get_postgres_connection_string()) as conn:
        with conn.cursor() as cur:
            with pytest.raises(psycopg.errors.ProgramLimitExceeded):
                cur.execute(
                    f"CREATE INDEX ON vdb.{collection} "
                    "USING hnsw (embedding vector_cosine_ops)"
                )


def test_search_uses_the_index(collection) -> None:
    """The store's own query shape has to reach it, not just some query."""
    assert "Index Scan" not in _plan(collection), "indexed before being indexed"
    _index_the_collection(collection)
    plan = _plan(collection)
    assert "Index Scan" in plan, f"the search fell back to a scan:\n{plan}"
    assert f"ix_hnsw_{collection}_embedding" in plan


def test_the_parameter_is_cast_not_the_column(collection) -> None:
    """Casting the column instead would put the index out of reach again."""
    _index_the_collection(collection)
    plan = _plan(collection)
    # `embedding <=> ($1)::halfvec`, not `(embedding)::halfvec <=> ...`
    assert "(embedding)::halfvec" not in plan, f"the column is being cast:\n{plan}"
