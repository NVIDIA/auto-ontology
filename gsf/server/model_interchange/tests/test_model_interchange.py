# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for GSF model YAML export/import."""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
import yaml

from gsf.dal.model_interchange import (
    UnknownDatabaseIdsError,
    _resolve_entity,
    assemble_export_document,
    resolve_sql_column_ids,
    validate_database_ids,
)
from gsf.semantic.constants import (
    SQL_ATTR_SOURCE_BRIDGE,
    SQL_ATTR_SOURCE_MANUAL,
    SQL_ATTR_SOURCE_SQL,
    SQL_ATTR_SOURCE_TABLE,
)
from gsf.server.model_interchange import service
from gsf.server.model_interchange.embed import (
    ImportEmbedBuffer,
    build_column_data_row,
    build_table_data_row,
    flush_import_embeddings,
)
from gsf.server.model_interchange.schemas import ExportRequest, GsfModelDocument


_NEEDS_LIVE_STORE = (
    "Needs a live graph store. These patch three DAL functions, but "
    "service.export_model also calls _dialect_by_database_name() -> "
    "list_connections(), which is unpatched and queries the store directly. "
    "Pre-dates the Neo4j-to-Postgres refactor; Phase 10 rewrites these against "
    "the Pagila/Chinook fixture. See docs/refactor/drop-neo4j/PLAN.md."
)


@contextmanager
def _null_transaction():
    """Stand in for ``write_transaction`` so unit tests need no live Neo4j."""
    yield


def _catalog_rows(*, db_id: str = "db-1", db_name: str = "retail") -> list[dict]:
    return [
        {
            "db_id": db_id,
            "db_name": db_name,
            "schema_id": "sch-1",
            "schema_name": "main",
            "table_id": "tbl-1",
            "table_name": "orders",
            "table_description": "Orders table",
            "pk": ["id"],
            "table_type": "table",
            "column_id": "col-1",
            "column_name": "id",
            "column_description": "",
            "column_type": "INTEGER",
            "sample_values": '["1", "2"]',
            "is_unique": True,
            "is_nullable": False,
            "ordinal_position": 1,
        },
    ]


def _export_rows(*, db_id: str = "db-1", db_name: str = "retail") -> dict:
    return {
        "catalog": _catalog_rows(db_id=db_id, db_name=db_name),
        "foreign_keys": [
            {"source_column_id": "col-1", "target_column_id": "col-2"},
        ],
        "joins": [
            {
                "source_table_id": "tbl-1",
                "target_table_id": "tbl-2",
                "join_columns": [{"source": "customer_id", "target": "id"}],
            },
        ],
        "terms": [
            {
                "id": "term-1",
                "name": "Order",
                "description": "An order",
                "represents": ["tbl-1"],
                "columns_attributes": [
                    {
                        "id": "attr-1",
                        "name": "order id",
                        "description": "",
                        "column_id": "col-1",
                    },
                ],
            },
        ],
        "semantic_fks": [
            {"column_id": "col-3", "column_attribute_id": "attr-1"},
        ],
        "sql_attributes": [
            {
                "id": "sa-manual",
                "name": "manual attr",
                "description": "",
                "expression": "SELECT id FROM orders",
                "source": SQL_ATTR_SOURCE_MANUAL,
                "sql": "SELECT id FROM orders",
                "term_id": "term-1",
                "database_name": db_name,
            },
            {
                "id": "sa-table",
                "name": "table attr",
                "description": "",
                "expression": "SELECT 1",
                "source": SQL_ATTR_SOURCE_TABLE,
                "sql": "SELECT 1",
                "term_id": "term-1",
                "database_name": db_name,
            },
            {
                "id": "sa-sql",
                "name": "sql attr",
                "description": "",
                "expression": "SELECT 2",
                "source": SQL_ATTR_SOURCE_SQL,
                "sql": "SELECT 2",
                "term_id": "term-1",
                "database_name": db_name,
            },
            {
                "id": "sa-bridge",
                "name": "bridge attr",
                "description": "",
                "expression": "SELECT 3",
                "source": SQL_ATTR_SOURCE_BRIDGE,
                "sql": "SELECT 3",
                "term_id": "term-1",
                "database_name": db_name,
            },
        ],
        "custom_analyses": [
            {
                "id": "ca-1",
                "name": "Top orders",
                "description": "Example",
                "sql": "SELECT * FROM orders",
                "database_name": db_name,
            },
        ],
    }


def test_assemble_export_document_groups_sql_attributes_by_source() -> None:
    document = assemble_export_document(
        _export_rows(),
        dialect_by_db_name={"retail": "sqlite"},
        sql_column_resolver=lambda _sql, _db: ["col-1"],
    )

    assert document.data_layer.databases[0].dialect == "sqlite"
    assert document.data_layer.databases[0].schemas[0].tables[0].columns[0].is_unique
    assert len(document.semantic_layer.sql_attributes.manual) == 1
    assert len(document.semantic_layer.sql_attributes.table) == 1
    assert len(document.semantic_layer.sql_attributes.sql) == 1
    assert len(document.semantic_layer.sql_attributes.bridge_table) == 1
    assert document.semantic_layer.semantic_fks[0].column_attribute_id == "attr-1"


def test_export_yaml_round_trips_through_safe_load() -> None:
    document = assemble_export_document(
        _export_rows(),
        dialect_by_db_name={"retail": "sqlite"},
        sql_column_resolver=lambda _sql, _db: [],
    )
    yaml_text = yaml.safe_dump(
        document.model_dump(mode="python"),
        sort_keys=False,
        default_flow_style=False,
    )
    loaded = yaml.safe_load(yaml_text)
    round_tripped = GsfModelDocument.model_validate(loaded)
    assert round_tripped.data_layer.databases[0].id == "db-1"
    assert (
        round_tripped.semantic_layer.terms[0].columns_attributes[0].column_id == "col-1"
    )


@pytest.mark.skip(reason=_NEEDS_LIVE_STORE)
@patch(
    "gsf.server.model_interchange.service.dal.resolve_sql_column_ids", return_value=[]
)
@patch("gsf.server.model_interchange.service.dal.fetch_export_rows")
@patch("gsf.server.model_interchange.service.dal.validate_database_ids")
def test_export_model_filters_by_database_id(
    mock_validate: MagicMock,
    mock_fetch: MagicMock,
    _mock_resolver: MagicMock,
) -> None:
    mock_fetch.return_value = _export_rows(db_id="db-2", db_name="inventory")

    yaml_text = service.export_model(ExportRequest(databases=["db-2"]))
    payload = yaml.safe_load(yaml_text)

    mock_validate.assert_called_once_with(["db-2"])
    mock_fetch.assert_called_once_with(["db-2"])
    assert payload["data_layer"]["databases"][0]["id"] == "db-2"


@pytest.mark.skip(reason=_NEEDS_LIVE_STORE)
@patch(
    "gsf.server.model_interchange.service.dal.resolve_sql_column_ids", return_value=[]
)
@patch("gsf.server.model_interchange.service.dal.fetch_export_rows")
@patch("gsf.server.model_interchange.service.dal.validate_database_ids")
def test_export_model_all_databases_uses_empty_filter(
    mock_validate: MagicMock,
    mock_fetch: MagicMock,
    _mock_resolver: MagicMock,
) -> None:
    mock_fetch.return_value = _export_rows()

    service.export_model(ExportRequest(databases=[]))

    mock_validate.assert_called_once_with([])
    mock_fetch.assert_called_once_with([])


@patch("gsf.dal.model_interchange.graph")
def test_validate_database_ids_raises_for_unknown(mock_conn: MagicMock) -> None:
    mock_conn.return_value.query_read.return_value = [{"ids": ["db-1"]}]

    with pytest.raises(UnknownDatabaseIdsError) as exc:
        validate_database_ids(["db-1", "missing"])

    assert exc.value.database_ids == ["missing"]


@patch("gsf.server.model_interchange.service.flush_import_embeddings")
@patch("gsf.server.model_interchange.service.dal.apply_import_model")
def test_import_model_flushes_embeddings_when_embed_true(
    mock_apply: MagicMock,
    mock_flush: MagicMock,
) -> None:
    mock_apply.return_value = {"terms": 1}
    mock_flush.return_value = {"skipped": False, "data_rows": 2, "semantic_rows": 3}
    document = assemble_export_document(
        _export_rows(),
        dialect_by_db_name={"retail": "sqlite"},
        sql_column_resolver=lambda _sql, _db: [],
    )
    yaml_text = yaml.safe_dump(document.model_dump(mode="python"))

    summary = service.import_model(yaml_text, replace=True, embed=True)

    mock_flush.assert_called_once()
    assert summary["embeddings"]["semantic_rows"] == 3


@patch("gsf.server.model_interchange.service.flush_import_embeddings")
@patch("gsf.server.model_interchange.service.dal.apply_import_model")
def test_import_model_skips_flush_when_embed_false(
    mock_apply: MagicMock,
    mock_flush: MagicMock,
) -> None:
    mock_apply.return_value = {"terms": 1}
    document = assemble_export_document(
        _export_rows(),
        dialect_by_db_name={"retail": "sqlite"},
        sql_column_resolver=lambda _sql, _db: [],
    )
    yaml_text = yaml.safe_dump(document.model_dump(mode="python"))

    service.import_model(yaml_text, replace=True, embed=False)

    mock_flush.assert_not_called()
    mock_apply.assert_called_once()
    assert mock_apply.call_args.kwargs["embed_buffer"] is None


@patch("gsf.server.model_interchange.embed.resolve", return_value="")
def test_flush_import_embeddings_skips_without_api_key(
    _mock_resolve: MagicMock,
) -> None:
    buffer = ImportEmbedBuffer(data_rows=[{"text": "x"}])
    result = flush_import_embeddings(buffer)
    assert result["skipped"] is True
    assert result["data_rows"] == 0


def test_build_catalog_embed_rows_match_tabular_shape() -> None:
    column_row = build_column_data_row(
        live_id="col-live",
        column_name="id",
        column_description="pk",
        data_type="INTEGER",
        sample_values=["1", "2"],
        table_name="orders",
        schema_name="main",
        database_name="retail",
    )
    table_row = build_table_data_row(
        live_id="tbl-live",
        table_name="orders",
        table_description="Orders",
        schema_name="main",
        database_name="retail",
        columns=[{"column_name": "id", "data_type": "INTEGER", "description": "pk"}],
    )
    assert column_row["metadata"]["label"] == "Column"
    assert column_row["metadata"]["id"] == "col-live"
    assert "column_name: id" in column_row["text"]
    assert table_row["metadata"]["label"] == "Table"
    assert "columns:" in table_row["text"]


@patch("gsf.dal.model_interchange.get_schemas")
@patch("gsf.dal.model_interchange.get_dialects")
@patch("gsf.dal.model_interchange.validate_sql")
def test_resolve_sql_column_ids_returns_parser_column_ids(
    mock_validate: MagicMock,
    mock_dialects: MagicMock,
    mock_schemas: MagicMock,
) -> None:
    mock_dialects.return_value = ["sqlite"]
    mock_schemas.return_value = {}
    mock_validate.return_value.get_column_ids.return_value = ["col-1", "col-2"]

    assert resolve_sql_column_ids("SELECT id FROM orders", "retail") == [
        "col-1",
        "col-2",
    ]


@patch("gsf.dal.model_interchange.graph")
def test_resolve_entity_skips_when_imported_id_exists(mock_conn: MagicMock) -> None:
    mock_conn.return_value.query_read.return_value = [{"id": "live-1"}]

    live_id, created = _resolve_entity("Database", "yaml-1")

    assert live_id == "live-1"
    assert created is False
    mock_conn.return_value.query_write.assert_called()
    mock_conn.return_value.query_read.assert_called_once()


@patch("gsf.dal.model_interchange.graph")
def test_resolve_entity_creates_with_imported_id(mock_conn: MagicMock) -> None:
    mock_conn.return_value.query_read.return_value = []
    mock_conn.return_value.query_write.return_value = [{"id": "new-1"}]

    live_id, created = _resolve_entity(
        "Table",
        "yaml-table",
        create_props={"name": "orders"},
    )

    assert live_id == "new-1"
    assert created is True
    write_params = mock_conn.return_value.query_write.call_args[0][1]
    assert write_params["props"]["imported_id"] == "yaml-table"
    assert write_params["props"]["name"] == "orders"


@patch("gsf.dal.model_interchange._ensure_import_indexes")
@patch("gsf.dal.model_interchange.write_transaction", _null_transaction)
@patch("gsf.dal.model_interchange._import_custom_analyses")
@patch("gsf.dal.model_interchange._import_sql_attributes")
@patch("gsf.dal.model_interchange._import_semantic_fks")
@patch("gsf.dal.model_interchange._import_column_attributes")
@patch("gsf.dal.model_interchange._import_terms")
@patch("gsf.dal.model_interchange._import_joins")
@patch("gsf.dal.model_interchange._import_foreign_keys")
@patch("gsf.dal.model_interchange._delete_scoped_semantics_not_in_payload")
@patch("gsf.dal.model_interchange._import_catalog")
def test_apply_import_model_creates_when_catalog_missing(
    mock_catalog: MagicMock,
    mock_delete: MagicMock,
    *_mocks: MagicMock,
) -> None:
    from gsf.dal.model_interchange import apply_import_model

    mock_catalog.return_value = ["live-db"]
    document = assemble_export_document(
        _export_rows(),
        dialect_by_db_name={"retail": "sqlite"},
        sql_column_resolver=lambda _sql, _db: [],
    )

    summary = apply_import_model(document, replace=True, embed_buffer=None)

    mock_catalog.assert_called_once()
    assert summary["database_ids"] == ["live-db"]
    assert "created" in summary
    assert "skipped" in summary


@patch("gsf.dal.model_interchange._ensure_import_indexes")
@patch("gsf.dal.model_interchange._import_terms", side_effect=RuntimeError("bad sql"))
@patch("gsf.dal.model_interchange._import_joins")
@patch("gsf.dal.model_interchange._import_foreign_keys")
@patch("gsf.dal.model_interchange._delete_scoped_semantics_not_in_payload")
@patch("gsf.dal.model_interchange._import_catalog")
@patch("gsf.dal.model_interchange.write_transaction")
def test_apply_import_model_runs_in_one_transaction(
    mock_transaction: MagicMock,
    mock_catalog: MagicMock,
    *_mocks: MagicMock,
) -> None:
    """A failure part-way through must not escape the transaction scope."""
    from gsf.dal.model_interchange import apply_import_model

    mock_transaction.return_value = _null_transaction()
    mock_catalog.return_value = ["live-db"]
    document = assemble_export_document(
        _export_rows(),
        dialect_by_db_name={"retail": "sqlite"},
        sql_column_resolver=lambda _sql, _db: [],
    )

    with pytest.raises(RuntimeError, match="bad sql"):
        apply_import_model(document, replace=True, embed_buffer=None)

    mock_transaction.assert_called_once()
