# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for GSF model YAML export/import."""

from __future__ import annotations

import json
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
import yaml
from ossie_gsf import GSFConversionError, convert_gsf_to_ossie

from gsf.dal.model_interchange import (
    _EXPORT_CATALOG_QUERY,
    _EXPORT_CUSTOM_ANALYSES_QUERY,
    _EXPORT_FKS_QUERY,
    _EXPORT_JOINS_QUERY,
    _EXPORT_SEMANTIC_FKS_QUERY,
    _EXPORT_SQL_ATTRIBUTES_QUERY,
    _EXPORT_TERMS_QUERY,
    UnknownDatabaseIdsError,
    _changed_entity_ids,
    _resolve_entity,
    _resolve_entities_batch,
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
from gsf.server.model_interchange.schemas import (
    ExportRequest,
    GsfModelDocument,
    ModelFormat,
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


def _in_scope_export_rows() -> dict:
    """Export rows whose semantic layer stays inside the exported catalog."""
    rows = _export_rows()
    rows["catalog"] = [
        *_catalog_rows(),
        {
            **_catalog_rows()[0],
            "column_id": "col-2",
            "column_name": "customer_id",
            "column_description": "",
            "sample_values": "[]",
            "is_unique": False,
            "is_nullable": True,
            "ordinal_position": 2,
        },
    ]
    rows["foreign_keys"] = [
        {"source_column_id": "col-2", "target_column_id": "col-1"},
    ]
    rows["joins"] = []
    rows["semantic_fks"] = [
        {"column_id": "col-2", "column_attribute_id": "attr-1"},
    ]
    return rows


def test_assemble_export_document_drops_references_outside_the_catalog() -> None:
    """A scoped export must not point at objects its own catalog omits.

    The fixture's foreign key, join and semantic FK all reach for tbl-2/col-2/
    col-3, which no exported database contains; keeping them would make the
    importer reject the file this very export produced.
    """
    document = assemble_export_document(
        _export_rows(),
        dialect_by_db_name={"retail": "sqlite"},
        sql_column_resolver=lambda _sql, _db: ["col-1", "col-outside"],
    )

    assert document.data_layer.foreign_keys == []
    assert document.data_layer.joins == []
    assert document.semantic_layer.semantic_fks == []
    assert document.semantic_layer.sql_attributes.manual[0].sql_column_is == ["col-1"]
    assert document.semantic_layer.custom_analyses[0].sql_column_is == ["col-1"]


def test_assemble_export_document_keeps_in_scope_references() -> None:
    document = assemble_export_document(
        _in_scope_export_rows(),
        dialect_by_db_name={"retail": "sqlite"},
        sql_column_resolver=lambda _sql, _db: [],
    )

    assert document.data_layer.foreign_keys[0].source_column_id == "col-2"
    assert document.semantic_layer.semantic_fks[0].column_attribute_id == "attr-1"
    assert document.semantic_layer.terms[0].represents == ["tbl-1"]


def test_assemble_export_document_drops_sql_attributes_of_absent_terms() -> None:
    rows = _export_rows()
    rows["terms"] = []
    document = assemble_export_document(
        rows,
        dialect_by_db_name={"retail": "sqlite"},
        sql_column_resolver=lambda _sql, _db: [],
    )

    assert document.semantic_layer.sql_attributes.manual == []
    assert document.semantic_layer.sql_attributes.table == []


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


@patch(
    "gsf.server.model_interchange.service.dal.make_cached_sql_column_resolver",
    return_value=lambda *args, **kwargs: [],
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
    assert (
        payload["data_layer"]["databases"][0]["schemas"][0]["database_name"]
        == "inventory"
    )


@patch(
    "gsf.server.model_interchange.service.dal.make_cached_sql_column_resolver",
    return_value=lambda *args, **kwargs: [],
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


@patch(
    "gsf.server.model_interchange.service.dal.make_cached_sql_column_resolver",
    return_value=lambda *args, **kwargs: [],
)
@patch("gsf.server.model_interchange.service.dal.fetch_export_rows")
@patch("gsf.server.model_interchange.service.dal.validate_database_ids")
def test_export_model_ossie_format_emits_ossie_document(
    _mock_validate: MagicMock,
    mock_fetch: MagicMock,
    _mock_resolver: MagicMock,
) -> None:
    mock_fetch.return_value = _export_rows()

    yaml_text = service.export_model(
        ExportRequest(databases=[], format=ModelFormat.OSSIE),
    )
    payload = yaml.safe_load(yaml_text)

    assert "data_layer" not in payload
    model = payload["semantic_model"][0]
    assert [dataset["name"] for dataset in model["datasets"]] == ["Order"]
    assert model["datasets"][0]["source"] == "retail.main.orders"


@patch(
    "gsf.server.model_interchange.service.dal.make_cached_sql_column_resolver",
    return_value=lambda *args, **kwargs: [],
)
@patch("gsf.server.model_interchange.service.dal.fetch_export_rows")
@patch("gsf.server.model_interchange.service.dal.validate_database_ids")
def test_export_model_ossie_reports_all_duplicate_custom_analysis_names(
    _mock_validate: MagicMock,
    mock_fetch: MagicMock,
    _mock_resolver: MagicMock,
) -> None:
    rows = _export_rows()
    rows["custom_analyses"] = [
        {
            "id": "ca-1",
            "name": "name25",
            "description": "",
            "sql": "SELECT 1",
            "database_name": "retail",
        },
        {
            "id": "ca-2",
            "name": "name25",
            "description": "",
            "sql": "SELECT 1",
            "database_name": "retail",
        },
        {
            "id": "ca-3",
            "name": "name3",
            "description": "",
            "sql": "SELECT 1",
            "database_name": "retail",
        },
        {
            "id": "ca-4",
            "name": "name3",
            "description": "",
            "sql": "SELECT 1",
            "database_name": "retail",
        },
    ]
    mock_fetch.return_value = rows

    with pytest.raises(
        GSFConversionError,
        match=r"duplicate names.*'name25' \(2\), 'name3' \(2\)",
    ):
        service.export_model(ExportRequest(databases=[], format=ModelFormat.OSSIE))


def _multi_table_term_rows() -> dict:
    """A term representing two tables, as a junction concept usually does."""
    rows = _export_rows()
    rows["catalog"] = [
        *_catalog_rows(),
        {
            **_catalog_rows()[0],
            "table_id": "tbl-2",
            "table_name": "categories",
            "table_description": "Categories table",
            "column_id": "col-2",
            "column_name": "category_id",
            "column_description": "",
            "sample_values": "[]",
            "is_unique": False,
            "is_nullable": True,
            "ordinal_position": 1,
        },
    ]
    rows["joins"] = []
    rows["semantic_fks"] = []
    rows["sql_attributes"] = []
    rows["terms"] = [
        {
            "id": "term-1",
            "name": "Category",
            "description": "A category",
            "represents": ["tbl-1", "tbl-2"],
            "columns_attributes": [
                {
                    "id": "attr-1",
                    "name": "order id",
                    "description": "",
                    "column_id": "col-1",
                },
                {
                    "id": "attr-2",
                    "name": "category id",
                    "description": "",
                    "column_id": "col-2",
                },
                {
                    "id": "attr-3",
                    "name": "category id again",
                    "description": "",
                    "column_id": "col-2",
                },
            ],
        },
    ]
    return rows


@patch(
    "gsf.server.model_interchange.service.dal.make_cached_sql_column_resolver",
    return_value=lambda *args, **kwargs: [],
)
@patch("gsf.server.model_interchange.service.dal.fetch_export_rows")
@patch("gsf.server.model_interchange.service.dal.validate_database_ids")
def test_export_model_ossie_keeps_one_table_per_term(
    _mock_validate: MagicMock,
    mock_fetch: MagicMock,
    _mock_resolver: MagicMock,
) -> None:
    """A multi-table term exports as the single dataset Ossie can hold.

    The table contributing most of the term's column attributes wins, and the
    attributes of the other table go with it — an Ossie field may only name a
    column of its own dataset.
    """
    mock_fetch.return_value = _multi_table_term_rows()

    payload = yaml.safe_load(
        service.export_model(ExportRequest(databases=[], format=ModelFormat.OSSIE)),
    )

    datasets = payload["semantic_model"][0]["datasets"]
    assert [dataset["name"] for dataset in datasets] == ["Category"]
    assert datasets[0]["source"] == "retail.main.categories"
    assert [field["name"] for field in datasets[0]["fields"]] == [
        "category id",
        "category id again",
    ]


@patch(
    "gsf.server.model_interchange.service.dal.make_cached_sql_column_resolver",
    return_value=lambda *args, **kwargs: [],
)
@patch("gsf.server.model_interchange.service.dal.fetch_export_rows")
@patch("gsf.server.model_interchange.service.dal.validate_database_ids")
def test_export_model_gsf_keeps_every_represented_table(
    _mock_validate: MagicMock,
    mock_fetch: MagicMock,
    _mock_resolver: MagicMock,
) -> None:
    mock_fetch.return_value = _multi_table_term_rows()

    payload = yaml.safe_load(service.export_model(ExportRequest(databases=[])))

    term = payload["semantic_layer"]["terms"][0]
    assert term["represents"] == ["tbl-1", "tbl-2"]
    assert len(term["columns_attributes"]) == 3


def test_detect_model_format_tells_the_vocabularies_apart() -> None:
    assert service.detect_model_format({"data_layer": {}}) is ModelFormat.GSF
    assert service.detect_model_format({"semantic_model": []}) is ModelFormat.OSSIE


@patch("gsf.server.model_interchange.service.dal.apply_import_model")
def test_import_model_converts_ossie_document_back_to_gsf(
    mock_apply: MagicMock,
) -> None:
    """An Ossie file must reach the DAL as the GSF document it describes."""
    mock_apply.return_value = {}
    document = assemble_export_document(
        _export_rows(),
        dialect_by_db_name={"retail": "sqlite"},
        sql_column_resolver=lambda _sql, _db: [],
    )
    gsf_yaml = yaml.safe_dump(document.model_dump(mode="python"), sort_keys=False)
    ossie_yaml = convert_gsf_to_ossie(gsf_yaml)

    summary = service.import_model(ossie_yaml, replace=True, embed=False)

    assert summary["format"] == ModelFormat.OSSIE.value
    imported = mock_apply.call_args[0][0]
    assert isinstance(imported, GsfModelDocument)
    assert [term.name for term in imported.semantic_layer.terms] == ["Order"]
    table = imported.data_layer.databases[0].schemas[0].tables[0]
    assert table.name == "orders"
    assert [column.name for column in table.columns] == ["id"]


@patch("gsf.server.model_interchange.service.dal.apply_import_model")
def test_import_model_reports_gsf_format(mock_apply: MagicMock) -> None:
    mock_apply.return_value = {}
    document = assemble_export_document(
        _export_rows(),
        dialect_by_db_name={"retail": "sqlite"},
        sql_column_resolver=lambda _sql, _db: [],
    )
    yaml_text = yaml.safe_dump(document.model_dump(mode="python"))

    summary = service.import_model(yaml_text, replace=True, embed=False)

    assert summary["format"] == ModelFormat.GSF.value


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


def test_export_queries_use_portable_ids() -> None:
    assert "coalesce(db.imported_id, db.id) AS db_id" in _EXPORT_CATALOG_QUERY
    assert "coalesce(sch.imported_id, sch.id) AS schema_id" in _EXPORT_CATALOG_QUERY
    assert "coalesce(tbl.imported_id, tbl.id) AS table_id" in _EXPORT_CATALOG_QUERY
    assert "coalesce(col.imported_id, col.id) AS column_id" in _EXPORT_CATALOG_QUERY
    assert "coalesce(src.imported_id, src.id)" in _EXPORT_FKS_QUERY
    assert "coalesce(t1.imported_id, t1.id)" in _EXPORT_JOINS_QUERY
    assert "coalesce(term.imported_id, term.id) AS id" in _EXPORT_TERMS_QUERY
    assert "coalesce(attr.imported_id, attr.id)" in _EXPORT_SEMANTIC_FKS_QUERY
    assert "coalesce(attr.imported_id, attr.id) AS id" in _EXPORT_SQL_ATTRIBUTES_QUERY
    assert "coalesce(ca.imported_id, ca.id) AS id" in _EXPORT_CUSTOM_ANALYSES_QUERY


@patch("gsf.dal.model_interchange.graph")
def test_changed_entity_ids_returns_only_modified_payloads(
    mock_conn: MagicMock,
) -> None:
    mock_conn.return_value.query_read.return_value = [
        {
            "id": "unchanged",
            "props": {"name": "Orders", "description": "", "source": "GSF"},
        },
        {
            "id": "changed",
            "props": {"name": "Orders", "description": "old", "source": "GSF"},
        },
    ]

    changed = _changed_entity_ids(
        "Term",
        {
            "unchanged": {"name": "Orders", "description": "", "source": "GSF"},
            "changed": {"name": "Orders", "description": "new", "source": "GSF"},
        },
    )

    assert changed == {"changed"}


@patch("gsf.dal.model_interchange.graph")
def test_resolve_entities_batch_migrates_database_by_name(
    mock_conn: MagicMock,
) -> None:
    mock_conn.return_value.query_read.side_effect = [
        [],
        [{"imported_id": "portable-db-id", "live_ids": ["legacy-live-id"]}],
    ]

    result = _resolve_entities_batch(
        "Database",
        [("portable-db-id", {"name": "dor_prod"})],
        match_by_name=True,
    )

    assert result == {"portable-db-id": ("legacy-live-id", False)}
    write_params = mock_conn.return_value.query_write.call_args[0][1]
    assert write_params["rows"] == [
        {
            "id": "legacy-live-id",
            "imported_id": "portable-db-id",
            "replace_imported_id": True,
        }
    ]


@patch("gsf.dal.model_interchange._database_names_for_columns")
@patch("gsf.dal.model_interchange._persist_sql_object")
@patch("gsf.dal.model_interchange.detach_ca_sql_edges")
@patch("gsf.dal.model_interchange._ids_missing_has_sql")
@patch("gsf.dal.model_interchange._custom_analysis_ids_with_changed_payload")
@patch("gsf.dal.model_interchange._resolve_entities_batch")
def test_import_custom_analyses_replace_restores_changed_name(
    mock_resolve_batch: MagicMock,
    mock_changed_payload: MagicMock,
    mock_missing_sql: MagicMock,
    mock_detach: MagicMock,
    mock_persist_sql: MagicMock,
    mock_db_names: MagicMock,
) -> None:
    """A replace import restores the exported name instead of silently skipping it."""
    from gsf.dal.model_interchange import _import_custom_analyses

    mock_resolve_batch.return_value = {"yaml-analysis": ("live-analysis", False)}
    mock_missing_sql.return_value = set()
    mock_changed_payload.return_value = {"live-analysis"}
    mock_db_names.return_value = {}
    document = GsfModelDocument.model_validate(
        {
            "semantic_layer": {
                "custom_analyses": [
                    {
                        "id": "yaml-analysis",
                        "name": "exported name",
                        "description": "description",
                        "sql": "SELECT 1",
                    },
                ],
            },
        },
    )

    buffer = ImportEmbedBuffer()
    _import_custom_analyses(
        document,
        {},
        {"custom_analyses": 0},
        {"custom_analyses": 0},
        buffer,
        {},
        replace=True,
    )

    mock_changed_payload.assert_called_once_with(
        document.semantic_layer.custom_analyses,
        {"yaml-analysis": "live-analysis"},
    )
    mock_detach.assert_called_once_with("live-analysis")
    assert mock_persist_sql.call_args.kwargs["name"] == "exported name"
    assert len(buffer.semantic_rows) == 1


@patch("gsf.dal.model_interchange._database_names_for_columns")
@patch("gsf.dal.model_interchange._persist_sql_object")
@patch("gsf.dal.model_interchange.detach_ca_sql_edges")
@patch("gsf.dal.model_interchange._ids_missing_has_sql")
@patch("gsf.dal.model_interchange._custom_analysis_ids_with_changed_payload")
@patch("gsf.dal.model_interchange._resolve_entities_batch")
def test_import_custom_analyses_uses_stable_id(
    mock_resolve_batch: MagicMock,
    mock_changed_payload: MagicMock,
    mock_missing_sql: MagicMock,
    _mock_detach: MagicMock,
    _mock_persist_sql: MagicMock,
    mock_db_names: MagicMock,
) -> None:
    """Only a matching stable id can select an existing analysis."""
    from gsf.dal.model_interchange import _import_custom_analyses

    mock_resolve_batch.return_value = {"yaml-analysis": ("stable-match", False)}
    mock_changed_payload.return_value = set()
    mock_missing_sql.return_value = set()
    mock_db_names.return_value = {}
    document = GsfModelDocument.model_validate(
        {
            "semantic_layer": {
                "custom_analyses": [
                    {
                        "id": "yaml-analysis",
                        "name": "exported name",
                        "description": "description",
                        "sql": "SELECT 1",
                    },
                ],
            },
        },
    )

    _import_custom_analyses(
        document,
        {},
        {"custom_analyses": 0},
        {"custom_analyses": 0},
        None,
        {},
        replace=True,
    )

    batch_items = mock_resolve_batch.call_args[0][1]
    assert [key for key, _props in batch_items] == ["yaml-analysis"]


@patch("gsf.dal.model_interchange.add_query")
@patch("gsf.dal.model_interchange.Neo4jNode")
@patch("gsf.dal.model_interchange.validate_sql")
def test_persist_sql_object_overwrites_existing_node_properties(
    mock_validate_sql: MagicMock,
    mock_neo4j_node: MagicMock,
    _mock_add_query: MagicMock,
) -> None:
    """A replace import must overwrite, not merely find, an existing analysis."""
    from gsf.dal.model_interchange import _persist_sql_object

    query_obj = MagicMock()
    query_obj.edges = []
    mock_validate_sql.return_value = query_obj

    _persist_sql_object(
        node_label="CustomAnalysis",
        node_id="live-analysis",
        name="exported name",
        description="exported description",
        sql="SELECT 1",
        database_name=None,
        schema_cache={None: (["sqlite"], {})},
    )

    assert mock_neo4j_node.call_args.kwargs["override_existing_props"] == {
        "name": "exported name",
        "description": "exported description",
    }


@patch("gsf.dal.model_interchange.graph")
def test_import_joins_stores_join_columns_as_a_json_string(
    mock_conn: MagicMock,
) -> None:
    """Neo4j properties can't be a list of maps; join_columns must be JSON-encoded.

    Writing the raw ``list[dict]`` straight to a relationship property (as
    the code used to) makes Neo4j reject the query with a CypherTypeError
    the moment a model with a non-empty join gets imported.
    """
    from gsf.dal.model_interchange import _import_joins

    document = GsfModelDocument.model_validate(
        {
            "data_layer": {
                "joins": [
                    {
                        "source_table_id": "tbl-1",
                        "target_table_id": "tbl-2",
                        "join_columns": [{"source": "customer_id", "target": "id"}],
                    },
                ],
            },
        },
    )

    _import_joins(document, {"tbl-1": "live-tbl-1", "tbl-2": "live-tbl-2"})

    write_params = mock_conn.return_value.query_write.call_args[0][1]
    row = write_params["rows"][0]
    assert isinstance(row["join_columns"], str)
    assert json.loads(row["join_columns"]) == [
        {"source": "customer_id", "target": "id"},
    ]


def test_assemble_export_document_parses_join_columns_json_string() -> None:
    """The DAL must read back what it now writes: a JSON-encoded property."""
    rows = _export_rows()
    rows["catalog"] = [
        *_catalog_rows(),
        {**_catalog_rows()[0], "table_id": "tbl-2", "table_name": "customers"},
    ]
    rows["joins"][0]["join_columns"] = '[{"source": "customer_id", "target": "id"}]'

    document = assemble_export_document(
        rows,
        dialect_by_db_name={"retail": "sqlite"},
        sql_column_resolver=lambda _sql, _db: [],
    )

    assert document.data_layer.joins[0].join_columns == [
        {"source": "customer_id", "target": "id"},
    ]


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
