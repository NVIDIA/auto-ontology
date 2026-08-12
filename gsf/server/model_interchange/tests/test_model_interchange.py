# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for GSF model YAML export/import.

Scope note (Phase 11): the tests that mocked ``apply_import_model``'s internals
and ``_resolve_entity`` went with the Neo4j implementation they patched. Their
coverage did not — ``gsf/dal/tests/test_model_interchange.py`` asserts the same
behaviours against a live store, which is a better test of "the import is one
transaction" than a mocked ``write_transaction`` ever was.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import yaml

from gsf.dal.model_interchange import assemble_export_document

from gsf.dal.model_interchange import resolve_sql_column_ids
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


# `export_model` reaches the store twice: once through the three DAL functions
# each export test patches, and once through `_dialect_by_database_name()`,
# which asks the connector registry and `list_connections()` what dialect each
# database speaks. That second call is what kept the two export tests skipped
# until Phase 10; both now patch it, since neither is about dialect resolution.


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


@patch("gsf.server.model_interchange.service.get_connectors", return_value=[])
@patch("gsf.server.model_interchange.service.list_connections", return_value=[])
@patch(
    "gsf.server.model_interchange.service.dal.resolve_sql_column_ids", return_value=[]
)
@patch("gsf.server.model_interchange.service.dal.fetch_export_rows")
@patch("gsf.server.model_interchange.service.dal.validate_database_ids")
def test_export_model_filters_by_database_id(
    mock_validate: MagicMock,
    mock_fetch: MagicMock,
    _mock_resolver: MagicMock,
    _mock_connections: MagicMock,
    _mock_connectors: MagicMock,
) -> None:
    mock_fetch.return_value = _export_rows(db_id="db-2", db_name="inventory")

    yaml_text = service.export_model(ExportRequest(databases=["db-2"]))
    payload = yaml.safe_load(yaml_text)

    mock_validate.assert_called_once_with(["db-2"])
    mock_fetch.assert_called_once_with(["db-2"])
    assert payload["data_layer"]["databases"][0]["id"] == "db-2"


@patch("gsf.server.model_interchange.service.get_connectors", return_value=[])
@patch("gsf.server.model_interchange.service.list_connections", return_value=[])
@patch(
    "gsf.server.model_interchange.service.dal.resolve_sql_column_ids", return_value=[]
)
@patch("gsf.server.model_interchange.service.dal.fetch_export_rows")
@patch("gsf.server.model_interchange.service.dal.validate_database_ids")
def test_export_model_all_databases_uses_empty_filter(
    mock_validate: MagicMock,
    mock_fetch: MagicMock,
    _mock_resolver: MagicMock,
    _mock_connections: MagicMock,
    _mock_connectors: MagicMock,
) -> None:
    mock_fetch.return_value = _export_rows()

    service.export_model(ExportRequest(databases=[]))

    mock_validate.assert_called_once_with([])
    mock_fetch.assert_called_once_with([])


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
