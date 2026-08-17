# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The HNSW indexes must actually be reachable from the search query.

Postgres matches an expression index *syntactically*. The indexes are built on
``(embedding::halfvec(2048)) halfvec_cosine_ops``, so a search that orders by
anything else -- the plain ``vector`` column being the obvious mistake -- plans a
sequential scan instead. Nothing fails when that happens: the same rows come
back, in the same order, just slowly. Measured on 51,200 rows the difference was
175ms versus 0.36ms, and the only place it shows up is a latency graph.

So the assertion is on the *plan*, not the results.

``enable_seqscan = off`` makes the check independent of table size. At the few
hundred rows a test fixture holds, Postgres would correctly prefer a sequential
scan whether or not the index existed, and the test would prove nothing either
way.
"""

from __future__ import annotations

import pytest
from sqlalchemy import text

from gsf.dal.session import store
from gsf.vdb import entity_store as es

pytestmark = pytest.mark.usefixtures("_requires_store")


@pytest.fixture
def _requires_store() -> None:
    try:
        store().query_read(text("SELECT 1 FROM catalog_table LIMIT 1"))
    except Exception as exc:  # noqa: BLE001 -- no database, nothing to assert
        pytest.skip(f"catalog store not reachable: {exc}")


def _plan_for(label: str) -> str:
    """The query plan for a real search against *label*'s table."""
    statement = es.search_statement(
        [0.1] * es.EMBEDDING_DIMENSIONS, label=label, top_k=10, database_name=None
    )
    sql = str(statement.compile(compile_kwargs={"literal_binds": True}))
    store().query_read(text("SET enable_seqscan = off"))
    try:
        rows = store().query_read(text("EXPLAIN " + sql))
        return "\n".join(str(next(iter(row.values()))) for row in rows)
    finally:
        store().query_read(text("SET enable_seqscan = on"))


@pytest.mark.parametrize("label", sorted(es.LABEL_TABLES))
def test_search_can_use_the_hnsw_index(label: str) -> None:
    table = es.LABEL_TABLES[label].name
    index = f"ix_hnsw_{table}_embedding"
    have = store().query_read(
        text("SELECT 1 FROM pg_indexes WHERE indexname = :n"), {"n": index}
    )
    if not have:
        pytest.skip(f"{index} not present (migration not applied)")

    plan = _plan_for(label)
    assert f"Index Scan using {index}" in plan, (
        f"the search for {label} cannot use {index}; it planned:\n{plan}\n\n"
        "The index is on (embedding::halfvec(2048)); the ORDER BY must use that "
        "exact expression or Postgres falls back to a sequential scan."
    )


def test_the_exact_column_does_the_final_ranking() -> None:
    """The shortlist is approximate; the distances handed back must not be.

    Without the outer re-rank the caller would receive fp16 distances, which are
    close enough to look right and wrong enough to reorder near-ties.
    """
    statement = es.search_statement(
        [0.1] * es.EMBEDDING_DIMENSIONS, label="Column", top_k=5, database_name=None
    )
    sql = str(statement.compile(compile_kwargs={"literal_binds": True})).lower()
    outer = sql.rsplit("from", 1)[0]
    assert "halfvec" not in outer, (
        "the outer projection ranks on halfvec; distances returned to callers "
        "must come from the exact vector column"
    )
