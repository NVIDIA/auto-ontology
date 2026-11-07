# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for :func:`auto_ontology.catalog.ingest.ingest_catalog`.

``ingest_catalog`` replaces the library's ``TabularSchemaExtractOp``. These
cover the parts that were the operator's contract rather than its plumbing: the
empty paths, and the ``(tables_df, columns_df)`` concat across schemas that
``CatalogEmbeddingRowsOp`` consumes downstream.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from auto_ontology.catalog.ingest import ingest_catalog


def _schema(
    tables: pd.DataFrame | None, columns: pd.DataFrame | None
) -> SimpleNamespace:
    return SimpleNamespace(tables_df=tables, columns_df=columns)


def _assert_empty_pair(pair: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    tables_df, columns_df = pair
    assert tables_df.empty and columns_df.empty


def test_returns_empty_pair_for_no_connector() -> None:
    tables_df, columns_df = ingest_catalog(None)
    assert tables_df.empty and columns_df.empty


def test_returns_empty_pair_when_extraction_yields_nothing() -> None:
    """A connector with no tables must not reach the writer at all."""
    connector = SimpleNamespace(dialect="sqlite", database_name="empty")
    with (
        patch(
            "auto_ontology.catalog.ingest.extract_tabular_db_data", return_value={}
        ) as extract,
        patch("auto_ontology.catalog.ingest.populate_tabular_data") as write,
    ):
        _assert_empty_pair(ingest_catalog(connector))

    extract.assert_called_once_with(connector)
    write.assert_not_called()


def test_returns_empty_pair_when_writer_reports_no_schemas() -> None:
    connector = SimpleNamespace(dialect="sqlite", database_name="empty")
    with (
        patch(
            "auto_ontology.catalog.ingest.extract_tabular_db_data",
            return_value={"tables": 1},
        ),
        patch("auto_ontology.catalog.ingest.populate_tabular_data", return_value={}),
    ):
        _assert_empty_pair(ingest_catalog(connector))


def test_concatenates_every_schema_and_keeps_them_distinguishable() -> None:
    """The per-row ``table_schema`` is what keeps two schemas apart downstream."""
    schemas = {
        "public": _schema(
            pd.DataFrame(
                [{"id": "t1", "table_name": "film", "table_schema": "public"}]
            ),
            pd.DataFrame(
                [{"id": "c1", "table_name": "film", "table_schema": "public"}]
            ),
        ),
        "analytics": _schema(
            pd.DataFrame(
                [{"id": "t2", "table_name": "film", "table_schema": "analytics"}]
            ),
            pd.DataFrame(
                [{"id": "c2", "table_name": "film", "table_schema": "analytics"}]
            ),
        ),
    }
    connector = SimpleNamespace(dialect="postgres", database_name="pagila")
    with (
        patch(
            "auto_ontology.catalog.ingest.extract_tabular_db_data",
            return_value={"tables": 1},
        ),
        patch(
            "auto_ontology.catalog.ingest.populate_tabular_data", return_value=schemas
        ) as write,
    ):
        tables_df, columns_df = ingest_catalog(connector)

    assert write.call_args.kwargs == {"num_workers": 4, "dialect": "postgres"}
    assert list(tables_df["id"]) == ["t1", "t2"]
    assert list(columns_df["id"]) == ["c1", "c2"]
    assert sorted(tables_df["table_schema"]) == ["analytics", "public"]


def test_skips_schemas_that_produced_no_frames() -> None:
    """``Schema.tables_df`` is None until the schema is written; drop those."""
    schemas = {
        "public": _schema(
            pd.DataFrame([{"id": "t1", "table_schema": "public"}]),
            None,
        ),
        "empty": _schema(None, None),
    }
    connector = SimpleNamespace(dialect="postgres", database_name="pagila")
    with (
        patch(
            "auto_ontology.catalog.ingest.extract_tabular_db_data",
            return_value={"tables": 1},
        ),
        patch(
            "auto_ontology.catalog.ingest.populate_tabular_data", return_value=schemas
        ),
    ):
        tables_df, columns_df = ingest_catalog(connector)

    assert list(tables_df["id"]) == ["t1"]
    assert columns_df.empty


def test_delete_table_cleans_up_orphaned_semantic_rows(monkeypatch) -> None:
    """Deleting a catalog table also removes its attributes and unrepresented terms."""
    from contextlib import nullcontext

    from auto_ontology.catalog.store import db as store_db

    writes: list[str] = []

    class _FakeStore:
        def query_read(self, statement):
            sql = str(statement)
            if "FROM catalog_column" in sql:
                return [{"id": "col-1"}]
            if "FROM table__term" in sql:
                return [{"term_id": "term-1"}]
            if "FROM column__has_attribute" in sql:
                return [{"attribute_id": "attr-1"}]
            return []

        def query_write(self, statement):
            writes.append(str(statement))
            return []

    monkeypatch.setattr(store_db, "write_transaction", nullcontext)
    monkeypatch.setattr(store_db, "store", lambda: _FakeStore())
    monkeypatch.setattr(
        store_db, "untag_attributes_of_columns_being_deleted", lambda _ids: None
    )

    store_db.delete_table("tbl-1")

    assert any("DELETE FROM column_attribute" in sql for sql in writes)
    assert any("DELETE FROM catalog_table" in sql for sql in writes)
    assert any("DELETE FROM term" in sql for sql in writes)
