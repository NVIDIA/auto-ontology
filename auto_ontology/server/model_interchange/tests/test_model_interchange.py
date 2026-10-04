# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-interchange service tests.

Unit tests for Auto Ontology model YAML export/import, run against mocks.

The storage-level behaviour these do not reach — portable ``imported_id``
matching, and the export/import round-trip — is covered against a live database
by ``auto_ontology/dal/tests/test_model_interchange.py`` and by the recorded golden
reads."""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
import yaml
from ossie_nvidia_gsf import GSFConversionError, convert_gsf_to_ossie

from auto_ontology.dal.model_interchange import (
    assemble_export_document,
    resolve_sql_column_ids,
)
from auto_ontology.semantic.constants import (
    SQL_ATTR_SOURCE_BRIDGE,
    SQL_ATTR_SOURCE_MANUAL,
    SQL_ATTR_SOURCE_SQL,
    SQL_ATTR_SOURCE_TABLE,
)
from auto_ontology.server.model_interchange import service
from auto_ontology.server.model_interchange.embed import (
    ImportEmbedBuffer,
    build_column_data_row,
    build_table_data_row,
    flush_import_embeddings,
)
from auto_ontology.server.model_interchange.schemas import (
    ExportRequest,
    AutoOntologyModelDocument,
    ModelFormat,
)


@contextmanager
def _null_transaction():
    """Stand in for ``write_transaction`` so unit tests need no live database."""
    yield


@pytest.fixture(autouse=True)
def _no_live_connection_lookup():
    """Sever the two connection lookups ``export_model`` reaches through.

    ``service.export_model`` calls ``_dialect_by_database_name()``, which reads
    ``get_connectors()`` and ``list_connections()`` — and the latter queries
    Postgres. Patching only the ``dal`` calls each test names leaves that path
    live, so these mock-backed tests quietly opened a connection and failed with
    ``OperationalError`` on any machine without the local stack up. CI never
    caught it: the workflow provides a Postgres service, so the call succeeded
    there and returned nothing.

    Autouse rather than per-test decorators so a test added later cannot
    reopen the hole. Both boundaries are stubbed instead of
    ``_dialect_by_database_name`` itself, which keeps its merge logic under
    test; a test that wants real dialects can patch these with its own values.
    """
    with (
        patch(
            "auto_ontology.server.model_interchange.service.get_connectors",
            return_value=[],
        ),
        patch(
            "auto_ontology.server.model_interchange.service.list_connections",
            return_value=[],
        ),
    ):
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


def test_assemble_export_document_keeps_sample_value_types() -> None:
    """An integer column must leave as numbers, or a re-import cannot restore them."""
    rows = _export_rows()
    rows["catalog"] = [{**_catalog_rows()[0], "sample_values": [1, 2]}]

    document = assemble_export_document(
        rows,
        dialect_by_db_name={"retail": "sqlite"},
        sql_column_resolver=lambda _sql, _db: [],
    )

    column = document.data_layer.databases[0].schemas[0].tables[0].columns[0]
    assert column.sample_values == [1, 2]
    assert all(type(value) is int for value in column.sample_values)


@pytest.mark.parametrize(
    ("supplied", "expected"),
    [
        ([1, 2], None),
        (["1", "2"], None),
        ([{"k": 1}], None),
        (["a", 1], None),
        ([], None),
    ],
)
@patch("auto_ontology.dal.model_interchange._resolve_entities_batch")
def test_import_catalog_drops_unclassified_sample_values(
    mock_resolve: MagicMock,
    supplied: list,
    expected: str | None,
) -> None:
    """Imported values cannot bypass local PII classification."""
    from collections import defaultdict

    from auto_ontology.dal import schema as s
    from auto_ontology.dal.model_interchange import _import_catalog

    mock_resolve.side_effect = lambda _table, items, **_kw: {
        imported_id: (f"live-{imported_id}", True) for imported_id, _ in items
    }
    rows = _export_rows()
    rows["catalog"] = [{**_catalog_rows()[0], "sample_values": supplied}]
    document = assemble_export_document(
        rows,
        dialect_by_db_name={"retail": "sqlite"},
        sql_column_resolver=lambda _sql, _db: [],
    )

    _import_catalog(document, {}, defaultdict(int), defaultdict(int), None, {})

    column_calls = [
        call for call in mock_resolve.call_args_list if call[0][0] is s.catalog_column
    ]
    props = column_calls[0][0][1][0][1]
    assert props["sample_values"] == expected


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
    round_tripped = AutoOntologyModelDocument.model_validate(loaded)
    assert round_tripped.data_layer.databases[0].id == "db-1"
    assert (
        round_tripped.semantic_layer.terms[0].columns_attributes[0].column_id == "col-1"
    )


@patch(
    "auto_ontology.server.model_interchange.service.dal.make_cached_sql_column_resolver",
    return_value=lambda *args, **kwargs: [],
)
@patch("auto_ontology.server.model_interchange.service.dal.fetch_export_rows")
@patch("auto_ontology.server.model_interchange.service.dal.validate_database_ids")
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
    "auto_ontology.server.model_interchange.service.dal.make_cached_sql_column_resolver",
    return_value=lambda *args, **kwargs: [],
)
@patch("auto_ontology.server.model_interchange.service.dal.fetch_export_rows")
@patch("auto_ontology.server.model_interchange.service.dal.validate_database_ids")
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
    "auto_ontology.server.model_interchange.service.dal.make_cached_sql_column_resolver",
    return_value=lambda *args, **kwargs: [],
)
@patch("auto_ontology.server.model_interchange.service.dal.fetch_export_rows")
@patch("auto_ontology.server.model_interchange.service.dal.validate_database_ids")
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
    "auto_ontology.server.model_interchange.service.dal.make_cached_sql_column_resolver",
    return_value=lambda *args, **kwargs: [],
)
@patch("auto_ontology.server.model_interchange.service.dal.fetch_export_rows")
@patch("auto_ontology.server.model_interchange.service.dal.validate_database_ids")
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
    "auto_ontology.server.model_interchange.service.dal.make_cached_sql_column_resolver",
    return_value=lambda *args, **kwargs: [],
)
@patch("auto_ontology.server.model_interchange.service.dal.fetch_export_rows")
@patch("auto_ontology.server.model_interchange.service.dal.validate_database_ids")
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
    "auto_ontology.server.model_interchange.service.dal.make_cached_sql_column_resolver",
    return_value=lambda *args, **kwargs: [],
)
@patch("auto_ontology.server.model_interchange.service.dal.fetch_export_rows")
@patch("auto_ontology.server.model_interchange.service.dal.validate_database_ids")
def test_export_model_auto_ontology_keeps_every_represented_table(
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
    assert service.detect_model_format({"data_layer": {}}) is ModelFormat.AUTO_ONTOLOGY
    assert service.detect_model_format({"semantic_model": []}) is ModelFormat.OSSIE


@patch("auto_ontology.server.model_interchange.service.dal.apply_import_model")
def test_import_model_converts_ossie_document_back_to_auto_ontology(
    mock_apply: MagicMock,
) -> None:
    """An Ossie file must reach the DAL as the Auto Ontology document it describes."""
    mock_apply.return_value = {}
    document = assemble_export_document(
        _export_rows(),
        dialect_by_db_name={"retail": "sqlite"},
        sql_column_resolver=lambda _sql, _db: [],
    )
    auto_ontology_yaml = yaml.safe_dump(
        document.model_dump(mode="python"), sort_keys=False
    )
    ossie_yaml = convert_gsf_to_ossie(auto_ontology_yaml)

    summary = service.import_model(ossie_yaml, replace=True, embed=False)

    assert summary["format"] == ModelFormat.OSSIE.value
    imported = mock_apply.call_args[0][0]
    assert isinstance(imported, AutoOntologyModelDocument)
    assert [term.name for term in imported.semantic_layer.terms] == ["Order"]
    table = imported.data_layer.databases[0].schemas[0].tables[0]
    assert table.name == "orders"
    assert [column.name for column in table.columns] == ["id"]


@patch("auto_ontology.server.model_interchange.service.dal.apply_import_model")
def test_import_model_reports_auto_ontology_format(mock_apply: MagicMock) -> None:
    mock_apply.return_value = {}
    document = assemble_export_document(
        _export_rows(),
        dialect_by_db_name={"retail": "sqlite"},
        sql_column_resolver=lambda _sql, _db: [],
    )
    yaml_text = yaml.safe_dump(document.model_dump(mode="python"))

    summary = service.import_model(yaml_text, replace=True, embed=False)

    assert summary["format"] == ModelFormat.AUTO_ONTOLOGY.value


@patch("auto_ontology.server.model_interchange.service.flush_import_embeddings")
@patch("auto_ontology.server.model_interchange.service.dal.apply_import_model")
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


@patch("auto_ontology.server.model_interchange.service.flush_import_embeddings")
@patch("auto_ontology.server.model_interchange.service.dal.apply_import_model")
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


@patch("auto_ontology.server.model_interchange.embed.resolve", return_value="")
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


@patch("auto_ontology.dal.model_interchange.get_schemas")
@patch("auto_ontology.dal.model_interchange.get_dialects")
@patch("auto_ontology.dal.model_interchange.validate_sql")
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
