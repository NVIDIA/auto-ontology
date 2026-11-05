# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A copied pgvector row must read back from Qdrant like a freshly written one."""

from __future__ import annotations

import pytest
from qdrant_client import QdrantClient

from auto_ontology.vdb.qdrant import QdrantVDB
from dev_tools.copy_vdb_to_qdrant import check_targets, copy_rows, pg_row_to_record

pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes have no effect")


def _embedding(axis: int, dimensions: int = 4) -> list[float]:
    vector = [0.0] * dimensions
    vector[axis] = 1.0
    return vector


def test_the_promoted_columns_return_to_the_metadata() -> None:
    record = pg_row_to_record(
        (
            "orders table",
            "[1,0,0,0]",
            '{"id": "t1", "name": "orders", "document_type": "text"}',
            "shop",
            "Table",
        )
    )
    assert record == {
        "text": "orders table",
        "document_type": "text",
        "metadata": {
            "id": "t1",
            "name": "orders",
            "database_name": "shop",
            "label": "Table",
            "embedding": [1, 0, 0, 0],
        },
    }


def test_copied_rows_are_searchable_and_filterable() -> None:
    client = QdrantClient(":memory:")
    target = QdrantVDB(client=client, collection_name="copy", vector_size=4)
    rows = [
        ("orders table", "[1,0,0,0]", {"id": "orders"}, "shop", "Table"),
        ("customers table", _embedding(1), {"id": "customers"}, "shop", "Table"),
        ("other table", _embedding(1), {"id": "other"}, "elsewhere", "Table"),
    ]

    assert copy_rows(rows, target) == 3

    [hits] = target.retrieval(
        [_embedding(1)], top_k=5, where={"database_name": "shop", "label": "Table"}
    )
    assert [hit["metadata"]["id"] for hit in hits] == ["customers", "orders"]
    assert hits[0]["text"] == "customers table"
    assert hits[0]["metadata"]["document_type"] == "text"
    assert target.delete_by_id("orders") == 1


def test_an_occupied_target_stops_the_copy_before_it_starts() -> None:
    client = QdrantClient(":memory:")
    empty = QdrantVDB(client=client, collection_name="empty", vector_size=4)
    occupied = QdrantVDB(client=client, collection_name="occupied", vector_size=4)
    copy_rows([("t", _embedding(0), {"id": "t"}, "shop", "Table")], occupied)

    with pytest.raises(SystemExit, match="occupied"):
        check_targets([empty, occupied], replace=False)
    check_targets([empty, occupied], replace=True)
