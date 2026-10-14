# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Embedded rows must reach the store in the shape retrieval reads back.

Retrieval reads a hit's ``metadata["id"]``, ``label``, ``name`` and friends
from the *top level* of the stored metadata, so a converter that tucked them
away, or a search that returned something else, would make every hit look
empty without raising. The round trip below writes through the real
:class:`PostgresVDB` and reads back through the real :class:`Retriever`; only
the embedding endpoint is replaced.
"""

from __future__ import annotations

import uuid
from typing import Any, Iterator

import psycopg
import pytest

from auto_ontology.infra.postgres import get_postgres_connection_string
from auto_ontology.utils.embedding import EmbedParams
from auto_ontology.utils.embedding_rows import build_embed_row
from auto_ontology.vdb.postgres import PostgresVDB
from auto_ontology.vdb.records import to_vdb_records, write_embedded_rows

_DIMENSIONS = 2048


def _row(node_id: str, embedding: Any, text: str = "orders table") -> dict:
    row = build_embed_row(
        text=text,
        node_id=node_id,
        label="Table",
        name=node_id,
        schema_name="public",
        database_name="shop",
    )
    if embedding is not None:
        row["metadata"]["embedding"] = embedding
    return row


def test_a_record_keeps_the_metadata_retrieval_reads() -> None:
    [[record]] = to_vdb_records([_row("t1", [0.5, 0.25])])

    assert record["document_type"] == "text"
    metadata = record["metadata"]
    assert {k: metadata[k] for k in ("id", "label", "name", "database_name")} == {
        "id": "t1",
        "label": "Table",
        "name": "t1",
        "database_name": "shop",
    }
    assert metadata["embedding"] == [0.5, 0.25]
    assert metadata["content"] == "orders table"
    assert metadata["content_metadata"]["page_number"] == -1
    assert metadata["content_metadata"]["id"] == "t1"
    assert metadata["source_metadata"] == {
        "source_id": "auto_ontology:t1",
        "source_name": "auto_ontology:t1",
    }


def test_the_callers_row_is_not_mutated() -> None:
    row = _row("t1", [0.5])
    to_vdb_records([row])
    assert "content" not in row["metadata"]


@pytest.mark.parametrize(
    "row",
    [
        _row("missing", None),
        # What a failed request leaves: present, but empty.
        _row("empty", []),
        _row("blank", [0.5], text="   "),
    ],
    ids=["missing", "empty", "blank-text"],
)
def test_rows_that_cannot_be_searched_are_dropped(row: dict) -> None:
    assert to_vdb_records([row]) == []


def test_nothing_to_write_is_falsy() -> None:
    """``[[]]`` is truthy, and callers use truthiness to decide whether to write."""
    assert to_vdb_records([]) == []
    assert to_vdb_records([_row("empty", [])]) == []


def test_a_dataframe_is_accepted() -> None:
    import pandas as pd

    [[record]] = to_vdb_records(pd.DataFrame([_row("t1", [0.5])]))
    assert record["metadata"]["id"] == "t1"


# ---------------------------------------------------------------------------
# Round trip through Postgres
# ---------------------------------------------------------------------------


def _unit(axis: int) -> list[float]:
    vector = [0.0] * _DIMENSIONS
    vector[axis] = 1.0
    return vector


@pytest.fixture
def vdb() -> Iterator["PostgresVDB"]:
    connection_string = get_postgres_connection_string()
    try:
        psycopg.connect(connection_string, connect_timeout=3).close()
    except Exception as exc:  # noqa: BLE001 -- no database, nothing to assert
        pytest.skip(f"postgres not reachable: {exc}")

    name = f"probe_{uuid.uuid4().hex[:8]}"
    store = PostgresVDB(
        connection_string=connection_string,
        collection_name=name,
        schema_name="vdb",
    )
    try:
        yield store
    finally:
        store.close()
        with psycopg.connect(connection_string) as conn:
            conn.execute(f"DROP TABLE IF EXISTS vdb.{name}")


def test_written_rows_come_back_from_a_search(
    vdb: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from auto_ontology.utils import retriever as retriever_mod

    written = write_embedded_rows(
        vdb,
        [
            _row("orders", _unit(0), text="orders table"),
            _row("customers", _unit(1), text="customers table"),
            _row("failed", [], text="never embedded"),
        ],
    )
    assert written == 2

    sent: list[tuple[list[str], str, dict]] = []

    def _embed(texts: list[str], params: Any, input_type: str, **kw: Any) -> list:
        sent.append((texts, input_type, kw))
        return [_unit(1)]

    monkeypatch.setattr(retriever_mod, "embed_texts", _embed)
    retriever = retriever_mod.Retriever(vdb, EmbedParams("m", "http://unused"))

    hits = retriever.query(
        "who buys", top_k=5, vdb_kwargs={"where": {"database_name": "shop"}}
    )

    # The caller's _query_with_retry is the only retry on this path, and each
    # attempt is bounded, so a dead endpoint cannot hold a question for long.
    assert sent == [(["who buys"], "query", {"timeout": 60.0, "retries": 0})]
    assert [hit["metadata"]["id"] for hit in hits] == ["customers", "orders"]
    best = hits[0]
    assert best["text"] == "customers table"
    assert best["metadata"]["label"] == "Table"
    assert best["_distance"] == pytest.approx(0.0, abs=1e-3)
    assert "embedding" not in best["metadata"]
    # The filter is applied by the store, not after the fact.
    assert (
        retriever.query("who buys", vdb_kwargs={"where": {"database_name": "x"}}) == []
    )
