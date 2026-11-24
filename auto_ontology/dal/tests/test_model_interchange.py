# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model YAML export and import.

Two things carry the weight here.

**The round trip must be idempotent.** Export a catalog, import it back, import
it again — the second import must create nothing. That is the plan's Done
criterion, and it exercises the `imported_id` matching that the
whole import turns on: an entity is found by the YAML id against `imported_id`
*or* the live id, so a document lands once however many times it is applied.

**Bug 3.** The catalog stores `'YES'`/`'NO'` and the old reader was `bool(...)`,
so every column in every export claimed to be nullable. It is asserted directly
against both stored values, because the failure was invisible — a bool is what
the schema expects, and `True` is a plausible one.
"""

from __future__ import annotations

import os
import uuid
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi.testclient import TestClient
    from httpx import Response

import pytest

pytest.importorskip("sqlalchemy")

from sqlalchemy import select  # noqa: E402

from auto_ontology.dal import model_interchange as mi  # noqa: E402
from auto_ontology.dal import schema as s  # noqa: E402
from auto_ontology.dal.session import store  # noqa: E402
from auto_ontology.semantic.constants import SEMANTIC_SOURCE  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def require_schema():
    if not os.environ.get("POSTGRES_USER"):
        pytest.skip("POSTGRES_* not set")
    try:
        store().query_read("SELECT 1 FROM catalog_database LIMIT 1")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"auto_ontology schema unavailable (alembic upgrade head): {exc}")


def _add(table, **values) -> str:
    return store().query_write(table.insert().values(**values).returning(table.c.id))[
        0
    ]["id"]


def _link(table, **values) -> None:
    store().query_write(table.insert().values(**values))


class World:
    """A small but complete catalog: two tables, a key, a term, an attribute."""

    def __init__(self, prefix: str) -> None:
        self.prefix = prefix
        self.database = _add(s.catalog_database, name=prefix)
        self.schema = _add(s.catalog_schema, database_id=self.database, name="public")

        self.orders = _add(
            s.catalog_table,
            schema_id=self.schema,
            name="orders",
            description="an order",
            table_type="base table",
            pk=["id"],
        )
        self.customers = _add(s.catalog_table, schema_id=self.schema, name="customers")
        self.order_id = _add(
            s.catalog_column,
            table_id=self.orders,
            name="id",
            data_type="integer",
            ordinal_position=1,
            is_nullable=False,
        )
        self.customer_ref = _add(
            s.catalog_column,
            table_id=self.orders,
            name="customer_id",
            data_type="integer",
            ordinal_position=2,
            is_nullable=True,
        )
        self.customer_id = _add(
            s.catalog_column,
            table_id=self.customers,
            name="id",
            data_type="integer",
            ordinal_position=1,
            is_nullable=False,
        )
        _link(
            s.column__foreign_key,
            source_column_id=self.customer_ref,
            target_column_id=self.customer_id,
        )
        _link(
            s.table__join,
            source_table_id=self.orders,
            target_table_id=self.customers,
            join_columns=[{"source": "customer_id", "target": "id"}],
        )

        self.term = _add(
            s.term,
            name=f"{prefix}-Order",
            description="a purchase",
            source=SEMANTIC_SOURCE,
        )
        _link(s.table__term, table_id=self.orders, term_id=self.term)
        self.attribute = _add(
            s.column_attribute,
            name=f"{prefix}-order-id",
            description="the identifier",
            source_column="id",
            term_name=f"{prefix}-Order",
            table_id=self.orders,
        )
        _link(
            s.column__has_attribute,
            column_id=self.order_id,
            attribute_id=self.attribute,
        )
        _link(s.column_attribute__term, attribute_id=self.attribute, term_id=self.term)
        _link(
            s.column__semantic_fk,
            column_id=self.customer_ref,
            attribute_id=self.attribute,
        )


def _cleanup(prefix: str) -> None:
    store().query_write(
        s.catalog_database.delete().where(s.catalog_database.c.name.like(f"{prefix}%"))
    )
    for table in (s.term, s.column_attribute, s.sql_attribute, s.custom_analysis):
        store().query_write(table.delete().where(table.c.name.like(f"{prefix}%")))


@pytest.fixture
def world():
    prefix = f"m-{uuid.uuid4().hex[:8]}"
    w = World(prefix)
    yield w
    _cleanup(prefix)


def _reid(document, seed: str):
    """Rewrite every id, as an export from another deployment would look.

    A document exported here carries this store's live ids, and the import
    matches those against `catalog_table.id` — so re-importing it *adopts* the
    rows rather than copying them, which is correct and is what
    `test_importing_onto_the_source_catalog_adopts_it` asserts. To exercise the
    create path the ids have to be foreign, which is exactly what a document
    from a different deployment has.
    """
    mapping: dict[str, str] = {}

    def new(old: str) -> str:
        return mapping.setdefault(old, f"{seed}-{uuid.uuid4().hex[:8]}")

    for database in document.data_layer.databases:
        database.id = new(database.id)
        for schema in database.schemas:
            schema.id = new(schema.id)
            for table in schema.tables:
                table.id = new(table.id)
                for column in table.columns:
                    column.id = new(column.id)
    for fk in document.data_layer.foreign_keys:
        fk.source_column_id = new(fk.source_column_id)
        fk.target_column_id = new(fk.target_column_id)
    for join in document.data_layer.joins:
        join.source_table_id = new(join.source_table_id)
        join.target_table_id = new(join.target_table_id)
    for term in document.semantic_layer.terms:
        term.id = new(term.id)
        term.represents = [new(table_id) for table_id in term.represents]
        # Names are re-seeded too. `term` is UNIQUE(name, source) and
        # `column_attribute` has a five-part merge key, so a document whose
        # entities merely *look* like existing ones collides — which is the
        # constraint doing its job, and is pinned separately below rather than
        # fought here. A genuinely foreign document has foreign names.
        term.name = f"{seed}-{term.name}"
        for attr in term.columns_attributes:
            attr.id = new(attr.id)
            attr.column_id = new(attr.column_id)
            attr.name = f"{seed}-{attr.name}"
    for fk in document.semantic_layer.semantic_fks:
        fk.column_id = new(fk.column_id)
        fk.column_attribute_id = new(fk.column_attribute_id)
    return document


def _document(world: World):
    rows = mi.fetch_export_rows([world.database])
    return mi.assemble_export_document(
        rows,
        dialect_by_db_name={world.prefix: "postgres"},
        sql_column_resolver=lambda sql, database_name: [],
    )


# --------------------------------------------------------------------------
# Bug 3 — nullability
# --------------------------------------------------------------------------


def test_nullability_survives_the_export(world) -> None:
    """Bug 3, asserted against both stored values.

    `bool('NO')` is `True`, so before this every column exported as nullable —
    and nothing failed, because a bool is exactly what the schema expects.
    """
    document = _document(world)
    columns = {
        column.name: column
        for database in document.data_layer.databases
        for schema in database.schemas
        for table in schema.tables
        if table.name == "orders"
        for column in table.columns
    }
    assert columns["id"].is_nullable is False
    assert columns["customer_id"].is_nullable is True


def test_an_undetermined_column_exports_as_nullable() -> None:
    """NULL means the connector could not tell, and absent has always meant
    nullable — the permissive reading for a constraint we cannot prove.

    The export schema types ``is_nullable`` as a plain ``bool``, so this is the
    one place the store's three states collapse into two.
    """
    assert mi._nullable_or_default(None) is True
    assert mi._nullable_or_default(True) is True
    assert mi._nullable_or_default(False) is False


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------


def test_export_carries_the_catalog(world) -> None:
    document = _document(world)
    database = document.data_layer.databases[0]

    assert database.id == world.database
    assert database.dialect == "postgres"
    tables = {table.name: table for table in database.schemas[0].tables}
    assert set(tables) == {"orders", "customers"}
    assert tables["orders"].description == "an order"
    assert tables["orders"].pk == ["id"]
    assert [c.name for c in tables["orders"].columns] == ["id", "customer_id"]


def test_export_carries_keys_joins_and_semantics(world) -> None:
    document = _document(world)

    assert [
        (fk.source_column_id, fk.target_column_id)
        for fk in document.data_layer.foreign_keys
    ] == [(world.customer_ref, world.customer_id)]
    assert [
        (j.source_table_id, j.target_table_id, j.join_columns)
        for j in document.data_layer.joins
    ] == [(world.orders, world.customers, [{"source": "customer_id", "target": "id"}])]

    term = document.semantic_layer.terms[0]
    assert term.id == world.term
    assert term.represents == [world.orders]
    assert [a.id for a in term.columns_attributes] == [world.attribute]
    assert [
        (fk.column_id, fk.column_attribute_id, fk.use_in_join_paths)
        for fk in document.semantic_layer.semantic_fks
    ] == [(world.customer_ref, world.attribute, True)]


def test_an_empty_id_list_exports_everything_not_nothing(world) -> None:
    """

    Reading it the other way would silently produce an empty document, which
    looks like a successful export of a catalog with nothing in it.
    """
    ids = {
        database.id
        for database in mi.assemble_export_document(
            mi.fetch_export_rows([]),
            dialect_by_db_name={},
            sql_column_resolver=lambda sql, database_name: [],
        ).data_layer.databases
    }
    assert world.database in ids
    # A *second* database, created here rather than assumed: the fixture builds
    # exactly one, so asserting `> 1` against whatever else happens to be in the
    # store passes only on a developer's dirty database and fails on a clean one.
    other = _add(s.catalog_database, name=f"{world.prefix}-second")
    # Down to a column: the export walks the whole hierarchy and joins through
    # catalog_column, so a database with no columns does not appear at all.
    other_schema = _add(s.catalog_schema, database_id=other, name="shop")
    other_table = _add(s.catalog_table, schema_id=other_schema, name="elsewhere")
    _add(
        s.catalog_column,
        table_id=other_table,
        name="id",
        data_type="integer",
        ordinal_position=1,
    )
    try:
        ids_again = {
            database.id
            for database in mi.assemble_export_document(
                mi.fetch_export_rows([]),
                dialect_by_db_name={},
                sql_column_resolver=lambda sql, database_name: [],
            ).data_layer.databases
        }
        assert {world.database, other} <= ids_again
    finally:
        store().query_write(
            s.catalog_database.delete().where(s.catalog_database.c.id == other)
        )


def test_validate_database_ids_rejects_an_unknown_one(world) -> None:
    mi.validate_database_ids([world.database])
    with pytest.raises(mi.UnknownDatabaseIdsError) as raised:
        mi.validate_database_ids([world.database, "no-such-database"])
    assert raised.value.database_ids == ["no-such-database"]


def test_validate_accepts_an_empty_list() -> None:
    """Empty means everything, so there is nothing to validate."""
    mi.validate_database_ids([])


def test_a_join_leaving_the_scope_is_not_exported(world) -> None:
    """Otherwise the document carries a join the importer cannot resolve."""
    other_prefix = f"{world.prefix}-other"
    other_db = _add(s.catalog_database, name=other_prefix)
    other_schema = _add(s.catalog_schema, database_id=other_db, name="public")
    outside = _add(s.catalog_table, schema_id=other_schema, name="elsewhere")
    _link(
        s.table__join,
        source_table_id=world.orders,
        target_table_id=outside,
        join_columns=[{"source": "x", "target": "y"}],
    )

    document = _document(world)
    targets = {join.target_table_id for join in document.data_layer.joins}
    assert outside not in targets
    assert world.customers in targets


# --------------------------------------------------------------------------
# The round trip
# --------------------------------------------------------------------------


def test_importing_an_export_into_an_empty_store_recreates_it(world) -> None:
    target = f"{world.prefix}-copy"
    document = _reid(_document(world), target)
    document.data_layer.databases[0].schemas[0].database_name = target
    document.semantic_layer.semantic_fks[0].use_in_join_paths = False

    try:
        summary = mi.apply_import_model(document, replace=False)

        assert summary["created"]["databases"] == 1
        assert summary["created"]["tables"] == 2
        assert summary["created"]["columns"] == 3
        assert summary["created"]["terms"] == 1
        assert summary["created"]["column_attributes"] == 1

        copied = store().query_read(
            select(s.catalog_database.c.id).where(s.catalog_database.c.name == target)
        )
        assert copied
        copied_semantic_fk = store().query_read(
            select(s.column__semantic_fk.c.use_in_join_paths)
            .select_from(
                s.column__semantic_fk.join(
                    s.catalog_column,
                    s.catalog_column.c.id == s.column__semantic_fk.c.column_id,
                )
            )
            .where(
                s.catalog_column.c.imported_id
                == document.semantic_layer.semantic_fks[0].column_id
            )
        )
        assert copied_semantic_fk == [{"use_in_join_paths": False}]
    finally:
        _cleanup(f"{world.prefix}-copy")


def test_the_round_trip_is_idempotent(world) -> None:
    """Idempotence.

    A second import of the same document must create nothing. This is what
    `imported_id` buys, and it is the difference between a re-import being a
    no-op and a re-import doubling the catalog.
    """
    target = f"{world.prefix}-copy"
    document = _reid(_document(world), target)
    document.data_layer.databases[0].schemas[0].database_name = target

    try:
        mi.apply_import_model(document, replace=False)
        second = mi.apply_import_model(document, replace=False)

        assert sum(second["created"].values()) == 0, second["created"]
        assert second["skipped"]["databases"] == 1
        assert second["skipped"]["tables"] == 2
        assert second["skipped"]["columns"] == 3
        assert second["skipped"]["terms"] == 1
    finally:
        _cleanup(f"{world.prefix}-copy")


def test_importing_onto_the_source_catalog_adopts_it(world) -> None:
    """The YAML ids *are* the live ids here, so nothing should be created.

    This is the `n.id = imported_id` half of the match, and the reason a first
    import onto an existing catalog does not duplicate every row.
    """
    summary = mi.apply_import_model(_document(world), replace=False)
    assert sum(summary["created"].values()) == 0, summary["created"]


def test_reimport_updates_semantic_fk_join_path_selection(world) -> None:
    document = _document(world)
    document.semantic_layer.semantic_fks[0].use_in_join_paths = False

    mi.apply_import_model(document, replace=False)

    selected = store().query_read(
        select(s.column__semantic_fk.c.use_in_join_paths).where(
            s.column__semantic_fk.c.column_id == world.customer_ref,
            s.column__semantic_fk.c.attribute_id == world.attribute,
        )
    )
    assert selected == [{"use_in_join_paths": False}]


def test_an_adopted_row_is_stamped_so_the_next_import_is_faster(world) -> None:
    mi.apply_import_model(_document(world), replace=False)
    row = store().query_read(
        select(s.catalog_table.c.imported_id).where(
            s.catalog_table.c.id == world.orders
        )
    )[0]
    assert row["imported_id"] == world.orders


def test_nullability_survives_the_whole_round_trip(world) -> None:
    """Bug 3 again, end to end: export, import, re-export.

    The export path alone can look correct while the import path writes a value
    that reads back inverted; only a round trip catches that.
    """
    target = f"{world.prefix}-copy"
    document = _reid(_document(world), target)
    document.data_layer.databases[0].schemas[0].database_name = target

    try:
        mi.apply_import_model(document, replace=False)
        copied_id = store().query_read(
            select(s.catalog_database.c.id).where(s.catalog_database.c.name == target)
        )[0]["id"]

        reexported = mi.assemble_export_document(
            mi.fetch_export_rows([copied_id]),
            dialect_by_db_name={},
            sql_column_resolver=lambda sql, database_name: [],
        )
        columns = {
            column.name: column
            for schema in reexported.data_layer.databases[0].schemas
            for table in schema.tables
            if table.name == "orders"
            for column in table.columns
        }
        assert columns["id"].is_nullable is False
        assert columns["customer_id"].is_nullable is True
    finally:
        _cleanup(f"{world.prefix}-copy")


def test_an_unresolvable_reference_is_rejected(world) -> None:
    """A payload naming an id it does not carry must fail, not half-apply."""
    document = _document(world)
    document.data_layer.foreign_keys[0].target_column_id = "no-such-column"

    with pytest.raises(mi.ModelImportValidationError, match="foreign-key"):
        mi.apply_import_model(document, replace=False)


def test_a_failed_import_leaves_nothing_behind(world) -> None:
    """The whole import is one transaction — the simplification Phase 10 banked.

    A failure anywhere rolls back everything, so an import cannot leave a
    catalog with half its semantics on top.
    """
    target = f"{world.prefix}-copy"
    document = _reid(_document(world), target)
    document.data_layer.databases[0].schemas[0].database_name = target
    # A structural reference, not a `represents` entry: an unresolvable foreign
    # key endpoint is a corrupt document and still raises, whereas a term
    # representing a table outside the export is expected and now skipped.
    document.data_layer.foreign_keys[0].target_column_id = "no-such-column"

    try:
        with pytest.raises(mi.ModelImportValidationError):
            mi.apply_import_model(document, replace=False)

        assert (
            store().query_read(
                select(s.catalog_database.c.id).where(
                    s.catalog_database.c.name == target
                )
            )
            == []
        )
    finally:
        _cleanup(f"{world.prefix}-copy")


# --------------------------------------------------------------------------
# replace
# --------------------------------------------------------------------------


def test_replace_drops_a_term_the_payload_omits(world) -> None:
    document = _document(world)
    stale = _add(
        s.term,
        name=f"{world.prefix}-Stale",
        source=SEMANTIC_SOURCE,
    )
    _link(s.table__term, table_id=world.orders, term_id=stale)

    mi.apply_import_model(document, replace=True)

    assert store().query_read(select(s.term.c.id).where(s.term.c.id == stale)) == []
    assert store().query_read(select(s.term.c.id).where(s.term.c.id == world.term))


def test_a_cross_database_term_stays_importable(world) -> None:
    """A scoped export must be importable on its own.

    A Term can be represented by tables in more than one database. `_scoped_ids`
    drops the out-of-scope ones at export, so the document only references
    tables it carries; the importer is *also* tolerant of ids it cannot resolve
    (`_remap_optional`), so a document produced elsewhere still imports.

    Belt and braces, and deliberately so: the export half keeps documents
    self-contained, the import half stops one bad reference aborting a whole
    import. Either alone leaves a gap — before both, such an export raised
    ModelImportValidationError and could not be imported anywhere.
    """
    other_prefix = f"{world.prefix}-other"
    other_db = _add(s.catalog_database, name=other_prefix)
    other_schema = _add(s.catalog_schema, database_id=other_db, name="public")
    outside = _add(s.catalog_table, schema_id=other_schema, name="elsewhere")

    shared = _add(s.term, name=f"{world.prefix}-Shared", source=SEMANTIC_SOURCE)
    _link(s.table__term, table_id=world.orders, term_id=shared)
    _link(s.table__term, table_id=outside, term_id=shared)

    document = _document(world)
    represents = {term.id: term.represents for term in document.semantic_layer.terms}
    assert represents[shared] == [world.orders], (
        "the export carries only the table this document also carries"
    )
    assert outside not in represents[shared], (
        "the out-of-scope table is dropped, so the document imports anywhere"
    )

    mi.apply_import_model(document, replace=True)

    linked = {
        row["table_id"]
        for row in store().query_read(
            select(s.table__term.c.table_id).where(s.table__term.c.term_id == shared)
        )
    }
    assert world.orders in linked, "the in-scope table is represented"


def test_replace_keeps_a_term_that_left_this_databases_scope(world) -> None:
    """A term representing a table outside the import is not `replace`'s to drop.

    The same all-or-nothing rule the reads and the reset apply: it belongs to
    that other database too.
    """
    other_prefix = f"{world.prefix}-other"
    other_db = _add(s.catalog_database, name=other_prefix)
    other_schema = _add(s.catalog_schema, database_id=other_db, name="public")
    outside = _add(s.catalog_table, schema_id=other_schema, name="elsewhere")

    document = _document(world)

    # Created *after* the export, so the payload does not mention it at all.
    shared = _add(s.term, name=f"{world.prefix}-Shared", source=SEMANTIC_SOURCE)
    _link(s.table__term, table_id=world.orders, term_id=shared)
    _link(s.table__term, table_id=outside, term_id=shared)

    mi.apply_import_model(document, replace=True)

    assert store().query_read(select(s.term.c.id).where(s.term.c.id == shared))


def test_replace_leaves_the_catalog_alone(world) -> None:
    """`replace` is about semantics; a table the payload omits is not deleted."""
    extra = _add(s.catalog_table, schema_id=world.schema, name="untouched")
    document = _document(world)
    document.data_layer.databases[0].schemas[0].tables = [
        table
        for table in document.data_layer.databases[0].schemas[0].tables
        if table.name != "untouched"
    ]

    mi.apply_import_model(document, replace=True)

    assert store().query_read(
        select(s.catalog_table.c.id).where(s.catalog_table.c.id == extra)
    )


def test_an_import_cannot_duplicate_a_term_name(world) -> None:
    """A constraint the graph did not have, and a behaviour change.

    `term` is `UNIQUE(name, source)`, so importing a document whose term shares
    a name with an existing one raises rather than creating a duplicate.

    Preflight reports the conflict before database mutation, rather than
    exposing the constraint violation to the HTTP caller.
    """
    document = _reid(_document(world), f"{world.prefix}-copy")
    document.data_layer.databases[0].schemas[0].database_name = f"{world.prefix}-copy"
    # Same name as the term already in the store, but a foreign id.
    document.semantic_layer.terms[0].name = f"{world.prefix}-Order"

    try:
        with pytest.raises(mi.ModelImportConflictError) as error:
            mi.apply_import_model(document, replace=False)
        assert error.value.conflicts[0]["entity_type"] == "term"
    finally:
        _cleanup(f"{world.prefix}-copy")


@pytest.mark.parametrize("replace", [True, False])
def test_foreign_ids_on_existing_paths_report_all_conflicts(
    world: World, replace: bool
) -> None:
    """A different deployment's IDs must not cause a partial import or a raw DB error."""
    before = mi.fetch_export_rows([world.database])
    document = _reid(_document(world), f"{world.prefix}-foreign")
    with pytest.raises(mi.ModelImportConflictError) as error:
        mi.apply_import_model(document, replace=replace)
    assert (
        len(error.value.conflicts) == 7
    )  # Database, schema, two tables, three columns.
    assert {c["kind"] for c in error.value.conflicts} == {"natural_name_conflict"}
    assert mi.fetch_export_rows([world.database]) == before


def _http_client() -> TestClient:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from auto_ontology.server.model_interchange.router import router

    app = FastAPI()
    app.include_router(router, prefix="/api")
    return TestClient(app)


def _post_document(
    document: mi.AutoOntologyModelDocument, *, replace: bool = False
) -> Response:
    import yaml

    return _http_client().post(
        f"/api/model/import?embed=false&replace={str(replace).lower()}",
        content=yaml.safe_dump(document.model_dump()),
        headers={"Content-Type": "application/yaml"},
    )


def test_http_reports_catalog_and_all_semantic_conflicts_before_writes(
    world: World,
) -> None:
    from unittest.mock import patch
    from auto_ontology.dal.session import StoreConn
    from auto_ontology.server.model_interchange.schemas import (
        ModelSqlAttribute,
        ModelCustomAnalysis,
    )

    document = _document(world)
    document.semantic_layer.terms[0].id = "foreign-term"
    document.semantic_layer.terms[0].columns_attributes[0].id = "foreign-attribute"
    sql_id = _add(s.sql_attribute, name=f"{world.prefix}-sql")
    ca_id = _add(s.custom_analysis, name=f"{world.prefix}-analysis")
    document.semantic_layer.sql_attributes.manual.append(
        ModelSqlAttribute(
            id="foreign-sql", name=f"{world.prefix}-sql", term_id="foreign-term"
        )
    )
    document.semantic_layer.custom_analyses.append(
        ModelCustomAnalysis(id="foreign-analysis", name=f"{world.prefix}-analysis")
    )
    # Catalog and semantic errors must be reported together.
    document.data_layer.databases[0].schemas[0].tables[0].id = "foreign-table"
    before = mi.fetch_export_rows([world.database])
    with patch.object(StoreConn, "query_write", autospec=True) as writes:
        response = _post_document(document)
    assert response.status_code == 409
    writes.assert_not_called()
    conflicts = response.json()["detail"]["conflicts"]
    assert {c["entity_type"] for c in conflicts} == {
        "catalog_table",
        "term",
        "column_attribute",
        "sql_attribute",
        "custom_analysis",
    }
    assert {sql_id, ca_id, world.term, world.attribute} <= {
        identity for conflict in conflicts for identity in conflict["existing_ids"]
    }
    assert mi.fetch_export_rows([world.database]) == before


def test_http_export_and_bind_recovers_from_foreign_physical_ids(world: World) -> None:
    import yaml
    from auto_ontology.server.model_interchange.schemas import ModelColumn

    def rewrite(value: Any, mapping: dict[str, str], key: str = "") -> Any:
        if isinstance(value, dict):
            return {k: rewrite(v, mapping, k) for k, v in value.items()}
        if isinstance(value, list):
            return [rewrite(v, mapping, key) for v in value]
        if isinstance(value, str) and (
            key == "id" or key.endswith("_id") or key in ("represents", "sql_column_is")
        ):
            return mapping.get(value, value)
        return value

    def identities(
        document: mi.AutoOntologyModelDocument,
    ) -> dict[tuple[str, ...], str]:
        paths: dict[tuple[str, ...], str] = {}
        for db in document.data_layer.databases:
            name = db.schemas[0].database_name
            paths[("database", name)] = db.id
            for schema in db.schemas:
                paths[("schema", name, schema.name)] = schema.id
                for table in schema.tables:
                    paths[("table", name, schema.name, table.name)] = table.id
                    for column in table.columns:
                        paths[
                            ("column", name, schema.name, table.name, column.name)
                        ] = column.id
        for term in document.semantic_layer.terms:
            paths[("term", term.name)] = term.id
            for attr in term.columns_attributes:
                paths[("attribute", term.name, attr.name)] = attr.id
        return paths

    source = _document(world)
    foreign_ids = {
        identity: f"foreign-{identity}" for identity in identities(source).values()
    }
    rejected = mi.AutoOntologyModelDocument.model_validate(
        rewrite(source.model_dump(), foreign_ids)
    )
    orders = next(
        t
        for t in rejected.data_layer.databases[0].schemas[0].tables
        if t.name == "orders"
    )
    new_column_id = f"{world.prefix}-new-column"
    orders.columns.append(
        ModelColumn(
            id=new_column_id, name="new_evidence", description="Keep this new field"
        )
    )
    response = _post_document(rejected)
    assert response.status_code == 409
    exported = _http_client().post(
        "/api/model/export", json={"databases": [world.database]}
    )
    assert exported.status_code == 200
    target = identities(
        mi.AutoOntologyModelDocument.model_validate(yaml.safe_load(exported.text))
    )
    mapping = {
        identity: target[path]
        for path, identity in identities(rejected).items()
        if path in target
    }
    rebound = mi.AutoOntologyModelDocument.model_validate(
        rewrite(rejected.model_dump(), mapping)
    )
    result = _post_document(rebound, replace=True)
    assert result.status_code == 200, result.text
    created = result.json()["summary"]["created"]
    assert created["columns"] == 1
    assert all(value == 0 for key, value in created.items() if key != "columns")
    added = store().query_read(
        select(s.catalog_column).where(s.catalog_column.c.imported_id == new_column_id)
    )
    assert len(added) == 1
    assert added[0]["table_id"] == world.orders
    assert added[0]["description"] == "Keep this new field"
    assert [
        (fk.source_column_id, fk.target_column_id)
        for fk in rebound.data_layer.foreign_keys
    ] == [(world.customer_ref, world.customer_id)]
    assert rebound.semantic_layer.terms[0].represents == [world.orders]
    assert (
        rebound.semantic_layer.terms[0].columns_attributes[0].column_id
        == world.order_id
    )
    assert rebound.semantic_layer.semantic_fks[0].column_attribute_id == world.attribute
    repeated = _post_document(rebound, replace=True)
    assert repeated.status_code == 200
    assert all(value == 0 for value in repeated.json()["summary"]["created"].values())


@pytest.mark.parametrize(
    "entity", ["database", "schema", "table", "column", "term", "attribute"]
)
def test_http_rejects_ambiguous_live_and_imported_ids(
    world: World, entity: str
) -> None:
    table, identity, props = {
        "database": (
            s.catalog_database,
            world.database,
            {"name": f"{world.prefix}-other"},
        ),
        "schema": (
            s.catalog_schema,
            world.schema,
            {"name": "other", "database_id": world.database},
        ),
        "table": (
            s.catalog_table,
            world.orders,
            {"name": "other", "schema_id": world.schema},
        ),
        "column": (
            s.catalog_column,
            world.order_id,
            {"name": "other", "table_id": world.orders},
        ),
        "term": (
            s.term,
            world.term,
            {"name": f"{world.prefix}-other", "source": SEMANTIC_SOURCE},
        ),
        "attribute": (
            s.column_attribute,
            world.attribute,
            {
                "name": f"{world.prefix}-other",
                "source_column": "id",
                "term_name": "other",
                "table_id": world.orders,
            },
        ),
    }[entity]
    other = _add(table, imported_id=identity, **props)
    response = _post_document(_document(world))
    assert response.status_code == 409
    assert any(
        c["kind"] == "ambiguous_id"
        and c["proposed_id"] == identity
        and set(c["existing_ids"]) == {identity, other}
        for c in response.json()["detail"]["conflicts"]
    )


def test_http_imported_ids_and_exact_live_ids_both_survive_replace(
    world: World,
) -> None:
    document = _document(world)
    term = document.semantic_layer.terms[0]
    attr = term.columns_attributes[0]
    # Stamped aliases differ from live IDs. Exact live IDs must remain legal
    # under replacement, without deleting the objects they identify.
    store().query_write(
        s.term.update()
        .where(s.term.c.id == world.term)
        .values(imported_id="external-term")
    )
    store().query_write(
        s.column_attribute.update()
        .where(s.column_attribute.c.id == world.attribute)
        .values(imported_id="external-attr")
    )
    assert _post_document(document, replace=True).status_code == 200
    term.id = "external-term"
    attr.id = "external-attr"
    document.semantic_layer.semantic_fks[0].column_attribute_id = attr.id
    result = _post_document(document, replace=True)
    assert result.status_code == 200
    assert result.json()["summary"]["created"]["terms"] == 0
    assert result.json()["summary"]["created"]["column_attributes"] == 0
    assert store().query_read(select(s.term.c.id).where(s.term.c.id == world.term))
    assert store().query_read(
        select(s.column_attribute.c.id).where(
            s.column_attribute.c.id == world.attribute
        )
    )


def test_http_replacement_allows_in_scope_semantic_id_change(world: World) -> None:
    document = _document(world)
    term = document.semantic_layer.terms[0]
    term.id = "replacement-term"
    term.columns_attributes[0].id = "replacement-attribute"
    document.semantic_layer.semantic_fks[
        0
    ].column_attribute_id = "replacement-attribute"
    result = _post_document(document, replace=True)
    assert result.status_code == 200
    assert result.json()["summary"]["created"]["terms"] == 1
    assert not store().query_read(select(s.term.c.id).where(s.term.c.id == world.term))


def test_http_replacement_cannot_delete_a_cross_database_term(world: World) -> None:
    other_db = _add(s.catalog_database, name=f"{world.prefix}-other")
    other_schema = _add(s.catalog_schema, name="public", database_id=other_db)
    other_table = _add(s.catalog_table, name="orders", schema_id=other_schema)
    _link(s.table__term, table_id=other_table, term_id=world.term)
    document = _document(world)
    document.semantic_layer.terms[0].id = "replacement-term"
    response = _post_document(document, replace=True)
    assert response.status_code == 409
    assert any(
        c["entity_type"] == "term" and world.term in c["existing_ids"]
        for c in response.json()["detail"]["conflicts"]
    )
    assert store().query_read(
        select(s.table__term).where(s.table__term.c.table_id == other_table)
    )


def test_http_real_integrity_failure_rolls_back_and_redacts(world: World) -> None:
    from unittest.mock import patch

    document = _document(world)
    before = mi.fetch_export_rows([world.database])
    original = mi._import_terms

    def violate_constraint(*args: Any, **kwargs: Any) -> None:
        # Simulate a writer racing preflight: earlier catalog stamps/writes
        # must roll back when a real Postgres uniqueness constraint fails.
        store().query_write(s.term.insert().values(name=f"{world.prefix}-Order"))
        original(*args, **kwargs)

    with patch.object(mi, "_import_terms", side_effect=violate_constraint):
        response = _post_document(document, replace=True)
    assert response.status_code == 409
    assert response.json()["detail"]["conflicts"] == [
        {"kind": "integrity_conflict", "existing_ids": []}
    ]
    assert "INSERT INTO" not in response.text
    assert "uq_term" not in response.text
    assert mi.fetch_export_rows([world.database]) == before


def test_sql_attributes_and_analyses_create_and_reimport_by_both_id_forms(
    world: World,
) -> None:
    from auto_ontology.server.model_interchange.schemas import (
        ModelSqlAttribute,
        ModelCustomAnalysis,
    )

    document = _document(world)
    document.semantic_layer.sql_attributes.manual.append(
        ModelSqlAttribute(
            id=f"{world.prefix}-external-sql",
            name=f"{world.prefix}-sql",
            term_id=world.term,
            sql="SELECT id FROM public.orders",
            sql_column_is=[world.order_id],
        )
    )
    document.semantic_layer.custom_analyses.append(
        ModelCustomAnalysis(
            id=f"{world.prefix}-external-analysis",
            name=f"{world.prefix}-analysis",
            sql="SELECT id FROM public.orders",
            sql_column_is=[world.order_id],
        )
    )
    result = _post_document(document, replace=True)
    assert result.status_code == 200, result.text
    assert result.json()["summary"]["created"]["sql_attributes"] == 1
    assert result.json()["summary"]["created"]["custom_analyses"] == 1
    second = _post_document(document, replace=True)
    assert second.status_code == 200, second.text
    assert all(value == 0 for value in second.json()["summary"]["created"].values())
    sql_live = store().query_read(
        select(s.sql_attribute.c.id).where(
            s.sql_attribute.c.name == f"{world.prefix}-sql"
        )
    )[0]["id"]
    analysis_live = store().query_read(
        select(s.custom_analysis.c.id).where(
            s.custom_analysis.c.name == f"{world.prefix}-analysis"
        )
    )[0]["id"]
    document.semantic_layer.sql_attributes.manual[0].id = sql_live
    document.semantic_layer.custom_analyses[0].id = analysis_live
    third = _post_document(document, replace=True)
    assert third.status_code == 200, third.text
    assert all(value == 0 for value in third.json()["summary"]["created"].values())


@pytest.mark.parametrize("entity", ["sql_attribute", "custom_analysis"])
def test_sql_owner_identity_ambiguity_is_reported(world: World, entity: str) -> None:
    from auto_ontology.server.model_interchange.schemas import (
        ModelSqlAttribute,
        ModelCustomAnalysis,
    )

    document = _document(world)
    table = getattr(s, entity)
    live_id = _add(table, name=f"{world.prefix}-one")
    other_id = _add(table, name=f"{world.prefix}-two", imported_id=live_id)
    if entity == "sql_attribute":
        document.semantic_layer.sql_attributes.manual.append(
            ModelSqlAttribute(
                id=live_id, name=f"{world.prefix}-one", term_id=world.term
            )
        )
    else:
        document.semantic_layer.custom_analyses.append(
            ModelCustomAnalysis(id=live_id, name=f"{world.prefix}-one")
        )
    response = _post_document(document)
    assert response.status_code == 409
    assert any(
        c["kind"] == "ambiguous_id" and set(c["existing_ids"]) == {live_id, other_id}
        for c in response.json()["detail"]["conflicts"]
    )


def test_identity_ambiguity_appearing_after_preflight_rolls_back(world: World) -> None:
    from unittest.mock import patch

    document = _document(world)
    before = mi.fetch_export_rows([world.database])
    original = mi._import_catalog

    def race_after_preflight(*args: Any, **kwargs: Any) -> list[str]:
        result = original(*args, **kwargs)
        _add(s.term, name=f"{world.prefix}-racing", imported_id=world.term)
        return result

    with patch.object(mi, "_import_catalog", side_effect=race_after_preflight):
        response = _post_document(document)
    assert response.status_code == 409
    assert response.json()["detail"]["conflicts"][0]["kind"] == "ambiguous_id"
    assert mi.fetch_export_rows([world.database]) == before
    assert not store().query_read(
        select(s.term.c.id).where(s.term.c.name == f"{world.prefix}-racing")
    )


def test_two_payload_aliases_cannot_bind_to_one_semantic_entity(world: World) -> None:
    document = _document(world)
    store().query_write(
        s.term.update()
        .where(s.term.c.id == world.term)
        .values(imported_id="other-alias")
    )
    duplicate = document.semantic_layer.terms[0].model_copy(deep=True)
    duplicate.id = "other-alias"
    duplicate.name = f"{world.prefix}-new-name"
    duplicate.columns_attributes = []
    document.semantic_layer.terms.append(duplicate)
    response = _post_document(document)
    assert response.status_code == 409
    assert any(
        c["kind"] == "duplicate_payload_path" and c["proposed_id"] == "other-alias"
        for c in response.json()["detail"]["conflicts"]
    )


@pytest.mark.parametrize("replace", [False, True])
def test_unapplied_matched_term_names_do_not_create_path_conflicts(
    world: World, replace: bool
) -> None:
    other = _add(s.term, name=f"{world.prefix}-Customer", source=SEMANTIC_SOURCE)
    _link(s.table__term, table_id=world.customers, term_id=other)
    document = _document(world)
    for term in document.semantic_layer.terms:
        term.name = f"{world.prefix}-Customer"
    response = _post_document(document, replace=replace)
    assert response.status_code == 200, response.text
    assert response.json()["summary"]["created"]["terms"] == 0
    rows = store().query_read(
        select(s.term.c.id, s.term.c.name).where(s.term.c.id.in_([world.term, other]))
    )
    assert {r["id"]: r["name"] for r in rows} == {
        world.term: f"{world.prefix}-Order",
        other: f"{world.prefix}-Customer",
    }


@pytest.mark.parametrize("match", ["id", "imported_id"])
def test_wrong_parent_committed_after_preflight_is_not_reused(
    world: World, match: str
) -> None:
    from unittest.mock import patch
    from auto_ontology.dal.session import get_engine
    from auto_ontology.server.model_interchange.schemas import ModelTable, ModelColumn

    outside_schema = _add(s.catalog_schema, database_id=world.database, name="outside")
    document = _document(world)
    proposed_id = f"{world.prefix}-late-table"
    proposed_column = f"{world.prefix}-late-column"
    document.data_layer.databases[0].schemas[0].tables.append(
        ModelTable(
            id=proposed_id,
            name="late_table",
            columns=[ModelColumn(id=proposed_column, name="id")],
        )
    )
    original = mi._import_catalog
    other_id = proposed_id if match == "id" else f"{world.prefix}-other-live-id"
    before = mi.fetch_export_rows([world.database])

    def concurrent_insert(*args: Any, **kwargs: Any) -> list[str]:
        # A different transaction commits after preflight but before resolution.
        with get_engine().begin() as connection:
            connection.execute(
                s.catalog_table.insert().values(
                    id=other_id,
                    imported_id=proposed_id if match == "imported_id" else None,
                    name="late_table",
                    schema_id=outside_schema,
                )
            )
        return original(*args, **kwargs)

    with patch.object(mi, "_import_catalog", side_effect=concurrent_insert):
        response = _post_document(document)
    assert response.status_code == 409
    assert any(
        c["kind"] == "identity_path_mismatch" and c["proposed_id"] == proposed_id
        for c in response.json()["detail"]["conflicts"]
    )
    # The concurrent writer's row remains, but this import must not attach or
    # create columns on it. Earlier stamps in our transaction are rolled back.
    row = store().query_read(
        select(s.catalog_table).where(s.catalog_table.c.id == other_id)
    )[0]
    assert row["schema_id"] == outside_schema
    assert not store().query_read(
        select(s.catalog_column.c.id).where(s.catalog_column.c.table_id == other_id)
    )
    assert not store().query_read(
        select(s.catalog_column.c.id).where(
            s.catalog_column.c.imported_id == proposed_column
        )
    )
    assert mi.fetch_export_rows([world.database]) == before
