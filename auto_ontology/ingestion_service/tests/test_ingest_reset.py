# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A failed embed must never reset the live catalog.

``run_ingest`` resets the data VDB lazily, on the first chunk with something
to write, so that a dead endpoint or a bad key leaves the previous run's rows
in place. A chunk whose every request failed still comes back as rows, each
without an embedding -- or, from older code paths, with an empty one -- and
those must count as "nothing to write" -- otherwise the reset fires and every row it was meant to
replace is then skipped on write.
"""

from __future__ import annotations

from typing import Any, Iterator

import pandas as pd
import pytest


class _Connector:
    database_name = "pagila"


class _RecordingVdb:
    def __init__(self) -> None:
        self.written: list[Any] = []

    def run(self, records: list) -> int:
        self.written.extend(records)
        return sum(len(batch) for batch in records)


def _row(node_id: str, embedding: list[float] | None) -> dict:
    metadata: dict[str, Any] = {
        "id": node_id,
        "label": "Table",
        "name": node_id,
        "source_path": f"auto_ontology:{node_id}",
        "database_name": "pagila",
        "content_metadata": {"id": node_id},
    }
    if embedding is not None:
        metadata["embedding"] = embedding
    return {
        "text": f"table {node_id}",
        "path": f"auto_ontology:{node_id}",
        "page_number": -1,
        "metadata": metadata,
    }


@pytest.fixture
def resets(monkeypatch: pytest.MonkeyPatch) -> list[_RecordingVdb]:
    import auto_ontology.ingestion_service.ingest as mod

    monkeypatch.setattr(mod, "get_embed_params", lambda: {})
    monkeypatch.setattr(mod, "ingest_catalog", lambda c: ([1], [1]))
    monkeypatch.setattr(mod, "is_pii_detection_enabled", lambda: False)
    monkeypatch.setattr(mod, "CatalogEmbeddingRowsOp", lambda **kw: lambda pair: pair)

    created: list[_RecordingVdb] = []

    def _get_data_vdb(**kwargs: Any) -> _RecordingVdb:
        assert kwargs.get("reset") is True
        vdb = _RecordingVdb()
        created.append(vdb)
        return vdb

    monkeypatch.setattr(mod, "get_data_vdb", _get_data_vdb)
    return created


def _embed_yielding(monkeypatch: pytest.MonkeyPatch, *chunks: list[dict]) -> None:
    import auto_ontology.ingestion_service.ingest as mod

    def _chunks(rows: Any, params: Any, label: str = "") -> Iterator[pd.DataFrame]:
        for chunk in chunks:
            yield pd.DataFrame(chunk)

    monkeypatch.setattr(mod, "batch_embed_chunks", _chunks)


def test_a_chunk_that_failed_to_embed_does_not_reset_the_catalog(
    resets: list[_RecordingVdb], monkeypatch: pytest.MonkeyPatch
) -> None:
    from auto_ontology.ingestion_service.ingest import run_ingest

    _embed_yielding(
        monkeypatch,
        [_row("a", None), _row("b", [])],
    )

    run_ingest(_Connector())

    assert resets == []


def test_the_reset_waits_for_the_first_chunk_that_embedded(
    resets: list[_RecordingVdb], monkeypatch: pytest.MonkeyPatch
) -> None:
    from auto_ontology.ingestion_service.ingest import run_ingest

    _embed_yielding(
        monkeypatch,
        [_row("a", None)],
        [_row("b", [0.1, 0.2]), _row("c", [])],
    )

    run_ingest(_Connector())

    assert len(resets) == 1
    written = [
        record["metadata"]["id"] for batch in resets[0].written for record in batch
    ]
    assert written == ["b"]
