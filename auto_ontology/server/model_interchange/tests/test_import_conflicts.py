# SPDX-FileCopyrightText: Copyright (c) 2026, AkilsuryaS.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Import identity checks against SQL-backed catalog rows and the HTTP route."""

from collections.abc import Iterator
from typing import Any
from unittest.mock import patch

import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import Column, MetaData, Table, Text, create_engine, select
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.pool import StaticPool

from auto_ontology.dal import model_interchange as dal
from auto_ontology.dal.session import StoreConn
from auto_ontology.server.model_interchange import service
from auto_ontology.server.model_interchange.router import router
from auto_ontology.server.model_interchange.schemas import (
    AutoOntologyModelDocument,
    ModelImportConflictResponse,
)


Catalog = tuple[Engine, dict[str, Table]]


@pytest.fixture
def catalog() -> Iterator[Catalog]:
    engine = create_engine(
        "sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False}
    )
    metadata = MetaData()
    tables = {}
    # Only the identity columns: the production metadata contains Postgres-only
    # types. Queries still use the real DAL tables, compiled by SQLAlchemy.
    for name, parent in (
        ("catalog_database", None),
        ("catalog_schema", "database_id"),
        ("catalog_table", "schema_id"),
        ("catalog_column", "table_id"),
    ):
        fields = [
            Column("id", Text, primary_key=True),
            Column("imported_id", Text),
            Column("name", Text),
        ]
        if parent:
            fields.append(Column(parent, Text))
        tables[name] = Table(name, metadata, *fields)
    for name, columns in {
        "term": ("id", "imported_id", "name", "source"),
        "column_attribute": (
            "id",
            "imported_id",
            "name",
            "source_column",
            "term_name",
            "table_id",
            "source",
        ),
        "sql_attribute": ("id", "imported_id", "name"),
        "custom_analysis": ("id", "imported_id", "name"),
        "table__term": ("table_id", "term_id"),
        "sql_attribute__sql": ("attribute_id", "sql_query_id"),
        "custom_analysis__sql": ("analysis_id", "sql_query_id"),
        "sql_query__table": ("sql_query_id", "table_id"),
    }.items():
        tables[name] = Table(
            name, metadata, *(Column(column, Text) for column in columns)
        )
    metadata.create_all(engine)
    with engine.begin() as connection:
        for name, row in (
            ("catalog_database", {"id": "db", "name": "shop"}),
            ("catalog_schema", {"id": "sch", "name": "main", "database_id": "db"}),
            ("catalog_table", {"id": "tbl", "name": "orders", "schema_id": "sch"}),
            ("catalog_column", {"id": "col", "name": "id", "table_id": "tbl"}),
        ):
            connection.execute(tables[name].insert().values(**row))
    with patch("auto_ontology.dal.session.get_engine", return_value=engine):
        yield engine, tables
    engine.dispose()


def document(prefix: str = "") -> AutoOntologyModelDocument:
    return AutoOntologyModelDocument.model_validate(
        {
            "data_layer": {
                "databases": [
                    {
                        "id": prefix + "db",
                        "schemas": [
                            {
                                "id": prefix + "sch",
                                "name": "main",
                                "database_name": "shop",
                                "tables": [
                                    {
                                        "id": prefix + "tbl",
                                        "name": "orders",
                                        "columns": [
                                            {"id": prefix + "col", "name": "id"}
                                        ],
                                    }
                                ],
                            }
                        ],
                    }
                ]
            }
        }
    )


def client() -> TestClient:
    app = FastAPI()
    app.include_router(router, prefix="/api")
    return TestClient(app)


@pytest.mark.parametrize("replace", [True, False])
def test_reports_all_path_conflicts_without_writes(
    catalog: Catalog, replace: bool
) -> None:
    engine, tables = catalog
    with patch.object(StoreConn, "query_write", autospec=True) as write:
        response = client().post(
            f"/api/model/import?embed=false&replace={str(replace).lower()}",
            content=yaml.safe_dump(document("external-").model_dump()),
        )
    assert response.status_code == 409
    ModelImportConflictResponse.model_validate(response.json())
    detail = response.json()["detail"]
    assert detail["code"] == "model_import_conflict"
    assert [c["existing_ids"] for c in detail["conflicts"]] == [
        ["db"],
        ["sch"],
        ["tbl"],
        ["col"],
    ]
    assert [c["proposed_id"] for c in detail["conflicts"]] == [
        "external-db",
        "external-sch",
        "external-tbl",
        "external-col",
    ]
    assert detail["conflicts"][-1]["path"] == ["shop", "main", "orders", "id"]
    assert "/api/model/export" in detail["resolution"]
    write.assert_not_called()
    with engine.connect() as connection:
        assert connection.execute(
            select(tables["catalog_database"])
        ).mappings().all() == [{"id": "db", "imported_id": None, "name": "shop"}]


@pytest.mark.parametrize("identity", ["live", "imported", "empty_database"])
def test_accepts_existing_identities(catalog: Catalog, identity: str) -> None:
    engine, tables = catalog
    doc = document()
    if identity == "imported":
        doc = document("external-")
        with engine.begin() as connection:
            for table, live_id in zip(tables.values(), ("db", "sch", "tbl", "col")):
                connection.execute(
                    table.update()
                    .where(table.c.id == live_id)
                    .values(imported_id="external-" + live_id)
                )
    elif identity == "empty_database":
        doc.data_layer.databases[0].schemas = []
    dal._preflight_import(doc, replace=False)


def test_new_database_does_not_collide_with_same_named_descendants(
    catalog: Catalog,
) -> None:
    doc = document("new-")
    doc.data_layer.databases[0].schemas[0].database_name = "another-shop"
    dal._preflight_import(doc, replace=False)


def test_ambiguous_id_is_rejected(catalog: Catalog) -> None:
    engine, tables = catalog
    with engine.begin() as connection:
        connection.execute(
            tables["catalog_database"]
            .insert()
            .values(id="other", name="another-shop", imported_id="db")
        )
    with pytest.raises(dal.ModelImportConflictError) as error:
        dal._preflight_import(document(), replace=False)
    assert error.value.conflicts[0]["kind"] == "ambiguous_id"
    assert error.value.conflicts[0]["existing_ids"] == ["db", "other"]


def test_id_cannot_bind_to_another_database(catalog: Catalog) -> None:
    engine, tables = catalog
    with engine.begin() as connection:
        connection.execute(
            tables["catalog_database"].insert().values(id="other", name="another-shop")
        )
    doc = document()
    doc.data_layer.databases[0].id = "other"
    doc.data_layer.databases[0].schemas[0].database_name = "another-shop"
    with pytest.raises(dal.ModelImportConflictError) as error:
        dal._preflight_import(doc, replace=False)
    assert error.value.conflicts[0]["kind"] == "identity_path_mismatch"
    assert error.value.conflicts[0]["proposed_id"] == "sch"


@pytest.mark.parametrize("duplicate", ["id", "path"])
def test_duplicate_payload_catalog_objects_are_rejected(
    catalog: Catalog, duplicate: str
) -> None:
    doc = document()
    schema = doc.data_layer.databases[0].schemas[0]
    table = schema.tables[0].model_copy(deep=True)
    table.columns = []
    if duplicate == "path":
        table.id = "another-table"
    schema.tables.append(table)
    with pytest.raises(dal.ModelImportConflictError) as error:
        dal._preflight_import(doc, replace=False)
    assert any(
        c["kind"] == f"duplicate_payload_{duplicate}" for c in error.value.conflicts
    )


def test_integrity_error_rolls_back_and_is_redacted(catalog: Catalog) -> None:
    engine, tables = catalog

    def fail_after_write(*args: Any, **kwargs: Any) -> None:
        dal.store().query_write(
            tables["catalog_database"].insert().values(id="partial", name="partial")
        )
        raise IntegrityError("private SQL", {"secret": "private"}, Exception("private"))

    with (
        patch.object(dal, "_import_catalog", side_effect=fail_after_write),
        patch.object(service, "flush_import_embeddings") as flush,
    ):
        response = client().post(
            "/api/model/import", content=yaml.safe_dump(document().model_dump())
        )
    assert response.status_code == 409
    ModelImportConflictResponse.model_validate(response.json())
    assert response.json()["detail"]["conflicts"] == [
        {"kind": "integrity_conflict", "existing_ids": []}
    ]
    assert "private" not in response.text
    flush.assert_not_called()
    with engine.connect() as connection:
        assert connection.execute(
            select(tables["catalog_database"].c.id)
        ).scalars().all() == ["db"]


@pytest.mark.parametrize("inconsistent", [False, True])
def test_shared_attribute_declarations_keep_one_identity(
    catalog: Catalog, inconsistent: bool
) -> None:
    from auto_ontology.server.model_interchange.schemas import (
        ModelColumn,
        ModelColumnAttribute,
        ModelTerm,
    )

    doc = document()
    table = doc.data_layer.databases[0].schemas[0].tables[0]
    table.columns.append(ModelColumn(id="another-column", name="customer_id"))
    first = ModelColumnAttribute(
        id="shared-attribute", name="Identifier", column_id="col"
    )
    second = first.model_copy(update={"column_id": "another-column"})
    if inconsistent:
        second.name = "Conflicting definition"
    doc.semantic_layer.terms.append(
        ModelTerm(id="term", name="Order", columns_attributes=[first, second])
    )
    if inconsistent:
        with pytest.raises(dal.ModelImportConflictError) as error:
            dal._preflight_import(doc, replace=False)
        assert error.value.conflicts[0]["kind"] == "duplicate_payload_id"
    else:
        dal._preflight_import(doc, replace=False)


def test_matched_term_name_edits_are_left_to_replacement_policy(
    catalog: Catalog,
) -> None:
    from auto_ontology.server.model_interchange.schemas import ModelTerm

    engine, tables = catalog
    with engine.begin() as connection:
        connection.execute(
            tables["term"].insert(),
            [
                {"id": "one", "name": "Order", "source": "semantic"},
                {"id": "two", "name": "Taken", "source": "semantic"},
            ],
        )
    doc = document()
    doc.semantic_layer.terms.append(ModelTerm(id="one", name="Taken"))
    # #581 retains the original name and reports a field-edit conflict. This
    # identity preflight must not replace that decision with a fatal HTTP 409.
    dal._preflight_import(doc, replace=True)


def test_catalog_labels_do_not_override_exact_id_bindings(catalog: Catalog) -> None:
    from auto_ontology.server.model_interchange.schemas import ModelColumn

    doc = document()
    table = doc.data_layer.databases[0].schemas[0].tables[0]
    table.columns.append(ModelColumn(id="new-column", name="customer_id"))
    # Exact IDs still identify the original columns. #267 does not implement
    # catalog-property updates from #250, or reject #581's attribute move.
    table.columns[0].name = "customer_id"
    dal._preflight_import(doc, replace=True)


def test_document_only_conflicts_explain_how_to_repair_new_objects(
    catalog: Catalog,
) -> None:
    from auto_ontology.server.model_interchange.schemas import ModelTerm

    doc = AutoOntologyModelDocument()
    doc.semantic_layer.terms = [
        ModelTerm(id="duplicate", name="One"),
        ModelTerm(id="duplicate", name="Two"),
    ]
    with pytest.raises(dal.ModelImportConflictError) as error:
        dal._preflight_import(doc, replace=False)
    detail = error.value.detail()
    assert detail["conflicts"][0]["existing_ids"] == []
    assert "distinct IDs" in detail["resolution"]
    assert "update their references" in detail["resolution"]
    assert "remove the duplicate declaration" in detail["resolution"]
    assert "/api/model/export" in detail["resolution"]


def test_new_terms_still_cannot_write_the_same_unique_key(catalog: Catalog) -> None:
    from auto_ontology.server.model_interchange.schemas import ModelTerm

    doc = AutoOntologyModelDocument()
    doc.semantic_layer.terms = [
        ModelTerm(id="one", name="Same"),
        ModelTerm(id="two", name="Same"),
    ]
    with pytest.raises(dal.ModelImportConflictError) as error:
        dal._preflight_import(doc, replace=False)
    assert error.value.conflicts[0]["kind"] == "duplicate_payload_path"


@pytest.mark.parametrize(
    "entity,identity,parent",
    [
        ("catalog_schema", "sch", "database_id"),
        ("catalog_table", "tbl", "schema_id"),
        ("catalog_column", "col", "table_id"),
    ],
)
def test_batch_resolver_validates_parents_before_stamping(
    catalog: Catalog, entity: str, identity: str, parent: str
) -> None:
    table = getattr(dal.s, entity)
    engine, tables = catalog
    expected_parent = {"database_id": "db", "schema_id": "sch", "table_id": "tbl"}[
        parent
    ]
    with engine.begin() as connection:
        connection.execute(
            tables[entity]
            .insert()
            .values(id="a-valid", name="valid", **{parent: expected_parent})
        )
    with patch.object(StoreConn, "query_write", autospec=True) as writes:
        with pytest.raises(dal.ModelImportConflictError) as error:
            dal._resolve_entities_batch(
                table,
                [
                    ("a-valid", {"name": "valid", parent: expected_parent}),
                    (identity, {"name": "candidate", parent: "wrong-parent"}),
                ],
            )
    writes.assert_not_called()
    assert error.value.conflicts == [
        {
            "kind": "identity_path_mismatch",
            "entity_type": entity,
            "proposed_id": identity,
            "existing_ids": [identity],
            "path": ["candidate"],
        }
    ]


def test_import_worker_does_not_block_unrelated_requests() -> None:
    import asyncio
    import threading
    import httpx

    started, release, finished = threading.Event(), threading.Event(), threading.Event()
    worker_threads: list[int] = []
    app = FastAPI()
    app.include_router(router, prefix="/api")

    @app.get("/ping")
    async def ping() -> dict[str, bool]:
        return {"ok": True}

    def slow_import(*args: Any, **kwargs: Any) -> dict:
        worker_threads.append(threading.get_ident())
        started.set()
        try:
            assert release.wait(10), "import worker was not released"
            return {}
        finally:
            finished.set()

    async def exercise() -> None:
        event_loop_thread = threading.get_ident()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as connection:
            request = asyncio.create_task(
                connection.post("/api/model/import", content="data_layer: {}")
            )
            try:
                assert await asyncio.to_thread(started.wait, 2)
                assert not finished.is_set()
                response = await asyncio.wait_for(connection.get("/ping"), timeout=2)
                assert response.status_code == 200
                assert len(worker_threads) == 1
                assert worker_threads[0] != event_loop_thread
            finally:
                release.set()
            assert (await request).status_code == 200

    with patch.object(service, "import_model", side_effect=slow_import):
        asyncio.run(exercise())


@pytest.mark.parametrize("entity", ["sql_attribute", "custom_analysis"])
def test_sql_owner_renames_check_the_key_that_is_actually_updated(
    catalog: Catalog, entity: str
) -> None:
    from auto_ontology.server.model_interchange.schemas import (
        ModelSqlAttribute,
        ModelCustomAnalysis,
    )

    engine, tables = catalog
    with engine.begin() as connection:
        connection.execute(
            tables[entity].insert(),
            [{"id": "one", "name": "Original"}, {"id": "two", "name": "Taken"}],
        )
    doc = document()
    if entity == "sql_attribute":
        doc.semantic_layer.sql_attributes.manual.append(
            ModelSqlAttribute(id="one", name="Taken")
        )
    else:
        doc.semantic_layer.custom_analyses.append(
            ModelCustomAnalysis(id="one", name="Taken")
        )
    with pytest.raises(dal.ModelImportConflictError) as error:
        dal._preflight_import(doc, replace=False)
    assert error.value.conflicts[0]["kind"] == "natural_name_conflict"
    assert set(error.value.conflicts[0]["existing_ids"]) == {"one", "two"}
