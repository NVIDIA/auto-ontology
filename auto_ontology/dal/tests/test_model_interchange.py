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

import pytest

pytest.importorskip("sqlalchemy")

from sqlalchemy import select  # noqa: E402

from auto_ontology.dal import model_interchange as mi  # noqa: E402
from auto_ontology.dal import schema as s  # noqa: E402
from auto_ontology.dal.session import store  # noqa: E402
from auto_ontology.server.model_interchange.schemas import AutoOntologyModelDocument  # noqa: E402
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

    Left to surface: the constraint describes a genuine ambiguity between two
    terms, and swallowing it would leave a duplicate no read can choose
    between.
    """
    document = _reid(_document(world), f"{world.prefix}-copy")
    document.data_layer.databases[0].schemas[0].database_name = f"{world.prefix}-copy"
    # Same name as the term already in the store, but a foreign id.
    document.semantic_layer.terms[0].name = f"{world.prefix}-Order"

    try:
        with pytest.raises(Exception, match="uq_term_name_source|already exists"):
            mi.apply_import_model(document, replace=False)
    finally:
        _cleanup(f"{world.prefix}-copy")


# --------------------------------------------------------------------------
# #525 — replace=true applies edits to matched terms and column attributes
# --------------------------------------------------------------------------


def _term_row(term_id: str) -> dict:
    return store().query_read(
        select(s.term.c.name, s.term.c.description).where(s.term.c.id == term_id)
    )[0]


def _attribute_row(attr_id: str) -> dict:
    return store().query_read(
        select(
            s.column_attribute.c.name,
            s.column_attribute.c.description,
            s.column_attribute.c.term_name,
        ).where(s.column_attribute.c.id == attr_id)
    )[0]


def _edited(world: World, **term_edits):
    """Export, then edit the term and its attribute as a person would."""
    document = _document(world)
    term = document.semantic_layer.terms[0]
    for field, value in term_edits.items():
        setattr(term, field, value)
    term.columns_attributes[0].description = "the order's identifier, edited"
    return document


def test_replace_applies_edits_to_matched_terms_and_attributes(world) -> None:
    """The issue's reproduction: export, edit descriptions, import with replace."""
    summary = mi.apply_import_model(
        _edited(world, description="a customer's purchase, edited"), replace=True
    )

    assert _term_row(world.term)["description"] == "a customer's purchase, edited"
    assert (
        _attribute_row(world.attribute)["description"]
        == "the order's identifier, edited"
    )
    assert summary["updated"]["terms"] == 1
    assert summary["updated"]["column_attributes"] == 1
    assert summary["skipped"]["terms"] == 0
    assert summary["conflicts"] == []


def test_replace_keeps_certified_fields_and_reports_them(world) -> None:
    store().query_write(
        s.term.update()
        .where(s.term.c.id == world.term)
        .values(description_certified=True)
    )
    store().query_write(
        s.column_attribute.update()
        .where(s.column_attribute.c.id == world.attribute)
        .values(certified=True)
    )

    summary = mi.apply_import_model(
        _edited(world, description="overwritten?"), replace=True
    )

    assert _term_row(world.term)["description"] == "a purchase"
    assert _attribute_row(world.attribute)["description"] == "the identifier"
    reported = {(c["entity"], c["field"], c["reason"]) for c in summary["conflicts"]}
    assert reported == {
        ("term", "description", "certified"),
        ("column_attribute", "description", "certified"),
    }
    assert summary["updated"]["terms"] == 0


def test_replace_renames_a_term_and_its_attributes_follow(world) -> None:
    new_name = f"{world.prefix}-Purchase"

    mi.apply_import_model(_edited(world, name=new_name), replace=True)

    assert _term_row(world.term)["name"] == new_name
    assert _attribute_row(world.attribute)["term_name"] == new_name


def test_a_rename_onto_another_terms_name_is_reported_not_fatal(world) -> None:
    taken = f"{world.prefix}-Taken"
    _add(s.term, name=taken, source=SEMANTIC_SOURCE)

    summary = mi.apply_import_model(_edited(world, name=taken), replace=True)

    assert _term_row(world.term)["name"] == f"{world.prefix}-Order"
    assert ("term", "name", "name in use") in {
        (c["entity"], c["field"], c["reason"]) for c in summary["conflicts"]
    }


def test_an_additive_import_still_leaves_matched_rows_alone(world) -> None:
    summary = mi.apply_import_model(
        _edited(world, description="not applied without replace"), replace=False
    )

    assert _term_row(world.term)["description"] == "a purchase"
    assert summary["skipped"]["terms"] == 1
    assert summary["updated"]["terms"] == 0


def test_an_identical_replace_import_updates_nothing(world) -> None:
    summary = mi.apply_import_model(_document(world), replace=True)

    assert sum(summary["updated"].values()) == 0
    assert summary["skipped"]["terms"] == 1


def test_replace_queues_changed_rows_for_reembedding(world) -> None:
    from auto_ontology.server.model_interchange.embed import ImportEmbedBuffer

    buffer = ImportEmbedBuffer()

    mi.apply_import_model(
        _edited(world, description="re-embed me"), replace=True, embed_buffer=buffer
    )

    assert {world.term, world.attribute} <= buffer.replaced_semantic_ids
    queued = {row["metadata"]["id"] for row in buffer.semantic_rows}
    assert world.term in queued


def _attribute_columns(attr_id: str) -> set[str]:
    return {
        row["column_id"]
        for row in store().query_read(
            select(s.column__has_attribute.c.column_id).where(
                s.column__has_attribute.c.attribute_id == attr_id
            )
        )
    }


def test_a_rename_alone_reembeds_the_terms_attributes(world: World) -> None:
    """The rename rewrites each attribute's term_name, which its vector embeds,
    so the attribute is re-embedded even when nothing else about it changed."""
    from auto_ontology.server.model_interchange.embed import ImportEmbedBuffer

    document = _document(world)
    new_name = f"{world.prefix}-Purchase"
    document.semantic_layer.terms[0].name = new_name
    buffer = ImportEmbedBuffer()

    summary = mi.apply_import_model(document, replace=True, embed_buffer=buffer)

    assert _attribute_row(world.attribute)["term_name"] == new_name
    assert world.attribute in buffer.replaced_semantic_ids
    queued = {row["metadata"]["id"]: row for row in buffer.semantic_rows}
    assert world.attribute in queued
    assert summary["updated"]["column_attributes"] == 1


def test_a_rejected_rename_keeps_attributes_on_the_stored_name(world: World) -> None:
    """A certified name keeps its value, so the attributes must keep copying it:
    the Term's attribute list is matched through term_name."""
    from auto_ontology.dal.terms import fetch_column_attributes_by_term_id

    store().query_write(
        s.term.update().where(s.term.c.id == world.term).values(name_certified=True)
    )
    document = _document(world)
    document.semantic_layer.terms[0].name = f"{world.prefix}-Rejected"

    summary = mi.apply_import_model(document, replace=True)

    stored = f"{world.prefix}-Order"
    assert _term_row(world.term)["name"] == stored
    assert _attribute_row(world.attribute)["term_name"] == stored
    assert [a["id"] for a in fetch_column_attributes_by_term_id(world.term)] == [
        world.attribute
    ]
    assert ("term", "name", "certified") in {
        (c["entity"], c["field"], c["reason"]) for c in summary["conflicts"]
    }


def test_moving_an_attribute_replaces_its_column_link(world: World) -> None:
    document = _document(world)
    document.semantic_layer.terms[0].columns_attributes[
        0
    ].column_id = world.customer_ref

    mi.apply_import_model(document, replace=True)
    again = mi.apply_import_model(document, replace=True)

    assert _attribute_columns(world.attribute) == {world.customer_ref}
    assert _attribute_row(world.attribute)["term_name"] == f"{world.prefix}-Order"
    assert again["updated"]["column_attributes"] == 0


def _shared_attribute_document(world: World) -> AutoOntologyModelDocument:
    """The attribute linked to orders.id and orders.customer_id, declared once
    per column, the stored source (orders.id) first and the other last."""
    _link(
        s.column__has_attribute,
        column_id=world.customer_ref,
        attribute_id=world.attribute,
    )
    document = _document(world)
    declarations = document.semantic_layer.terms[0].columns_attributes
    order = {world.order_id: 0, world.customer_ref: 1}
    declarations.sort(key=lambda declaration: order[declaration.column_id])
    assert [d.column_id for d in declarations] == [world.order_id, world.customer_ref]
    return document


def test_a_shared_attribute_keeps_every_declared_link(world: World) -> None:
    """An unchanged export of an attribute on two columns re-imports as is:
    neither link is dropped, and nothing counts as updated or re-embedded."""
    from auto_ontology.server.model_interchange.embed import ImportEmbedBuffer

    document = _shared_attribute_document(world)
    buffer = ImportEmbedBuffer()

    summary = mi.apply_import_model(document, replace=True, embed_buffer=buffer)

    assert _attribute_columns(world.attribute) == {world.order_id, world.customer_ref}
    stored_source = store().query_read(
        select(s.column_attribute.c.source_column).where(
            s.column_attribute.c.id == world.attribute
        )
    )[0]["source_column"]
    assert stored_source == "id"
    assert summary["updated"]["column_attributes"] == 0
    assert world.attribute not in buffer.replaced_semantic_ids


def test_every_declared_column_is_linked(world: World) -> None:
    document = _shared_attribute_document(world)
    store().query_write(
        s.column__has_attribute.delete().where(
            s.column__has_attribute.c.column_id == world.customer_ref
        )
    )

    summary = mi.apply_import_model(document, replace=True)

    assert _attribute_columns(world.attribute) == {world.order_id, world.customer_ref}
    assert summary["updated"]["column_attributes"] == 1


def test_a_link_the_document_drops_is_removed(world: World) -> None:
    from auto_ontology.server.model_interchange.embed import ImportEmbedBuffer

    document = _shared_attribute_document(world)
    declarations = document.semantic_layer.terms[0].columns_attributes
    del declarations[1]
    buffer = ImportEmbedBuffer()

    summary = mi.apply_import_model(document, replace=True, embed_buffer=buffer)

    assert _attribute_columns(world.attribute) == {world.order_id}
    assert summary["updated"]["column_attributes"] == 1
    # The vectors follow the links, so a removed link re-embeds.
    assert world.attribute in buffer.replaced_semantic_ids


def test_a_scoped_import_keeps_links_to_other_databases(world: World) -> None:
    """The document describes only its own database's columns, so a shared
    attribute's link into another database is not the import's to remove."""
    other_db = _add(s.catalog_database, name=f"{world.prefix}-other")
    other_schema = _add(s.catalog_schema, database_id=other_db, name="public")
    other_table = _add(s.catalog_table, schema_id=other_schema, name="returns")
    other_column = _add(
        s.catalog_column,
        table_id=other_table,
        name="order_id",
        data_type="integer",
        ordinal_position=1,
        is_nullable=True,
    )
    _link(s.column__has_attribute, column_id=other_column, attribute_id=world.attribute)
    document = _document(world)

    summary = mi.apply_import_model(document, replace=True)

    assert _attribute_columns(world.attribute) == {world.order_id, other_column}
    assert summary["updated"]["column_attributes"] == 0


def _other_database_column(world: World) -> tuple[str, str]:
    """A table and column in a second database, which the world's Term also
    represents. Named under the world's prefix so the fixture cleans it up."""
    other_db = _add(s.catalog_database, name=f"{world.prefix}-other")
    other_schema = _add(s.catalog_schema, database_id=other_db, name="sales")
    other_table = _add(s.catalog_table, schema_id=other_schema, name="returns")
    other_column = _add(
        s.catalog_column,
        table_id=other_table,
        name="order_id",
        data_type="integer",
        ordinal_position=1,
        is_nullable=True,
    )
    return other_table, other_column


def _queued(buffer, live_id: str) -> list[dict]:
    return [r for r in buffer.semantic_rows if r["metadata"].get("id") == live_id]


def test_a_shared_terms_vector_is_rebuilt_in_every_database(world: World) -> None:
    """The flush deletes a replaced Term's vectors by id, which removes the
    copy in every database it represents, so each copy must be rebuilt."""
    from auto_ontology.server.model_interchange.embed import ImportEmbedBuffer

    other_table, _ = _other_database_column(world)
    _link(s.table__term, table_id=other_table, term_id=world.term)
    document = _document(world)
    document.semantic_layer.terms[0].description = "an order a customer placed"
    buffer = ImportEmbedBuffer()

    mi.apply_import_model(document, replace=True, embed_buffer=buffer)

    assert world.term in buffer.replaced_semantic_ids
    assert {r["metadata"]["database_name"] for r in _queued(buffer, world.term)} == {
        world.prefix,
        f"{world.prefix}-other",
    }


def test_a_rename_reembeds_attributes_outside_the_document(world: World) -> None:
    """A database-scoped document leaves out the shared Term's attribute in
    the other database, but the rename rewrote its term_name too."""
    from auto_ontology.server.model_interchange.embed import ImportEmbedBuffer

    other_table, other_column = _other_database_column(world)
    outside = _add(
        s.column_attribute,
        name=f"{world.prefix}-returned-order",
        description="the returned order",
        source_column="order_id",
        term_name=f"{world.prefix}-Order",
        table_id=other_table,
    )
    _link(s.column__has_attribute, column_id=other_column, attribute_id=outside)
    _link(s.column_attribute__term, attribute_id=outside, term_id=world.term)
    document = _document(world)
    new_name = f"{world.prefix}-Purchase"
    document.semantic_layer.terms[0].name = new_name
    buffer = ImportEmbedBuffer()

    mi.apply_import_model(document, replace=True, embed_buffer=buffer)

    assert outside in buffer.replaced_semantic_ids
    rows = _queued(buffer, outside)
    assert rows and all(new_name in r["text"] for r in rows)
    assert {r["metadata"]["database_name"] for r in rows} == {f"{world.prefix}-other"}


def test_a_rename_reembeds_the_terms_sql_attributes(world: World) -> None:
    from auto_ontology.server.model_interchange.embed import ImportEmbedBuffer
    from auto_ontology.server.model_interchange.schemas import ModelSqlAttribute

    document = _document(world)
    document.semantic_layer.sql_attributes.manual.append(
        ModelSqlAttribute(
            id=f"{world.prefix}-order-count",
            name=f"{world.prefix}-order-count",
            description="orders placed",
            sql="SELECT COUNT(*) FROM public.orders",
            term_id=document.semantic_layer.terms[0].id,
        )
    )
    mi.apply_import_model(document, replace=True)
    sql_attr = store().query_read(
        select(s.sql_attribute.c.id).where(
            s.sql_attribute.c.name == f"{world.prefix}-order-count"
        )
    )[0]["id"]
    new_name = f"{world.prefix}-Purchase"
    document.semantic_layer.terms[0].name = new_name
    buffer = ImportEmbedBuffer()

    mi.apply_import_model(document, replace=True, embed_buffer=buffer)

    assert sql_attr in buffer.replaced_semantic_ids
    rows = _queued(buffer, sql_attr)
    assert len(rows) == 1 and f"term: {new_name}" in rows[0]["text"]


def test_a_move_to_a_column_named_like_the_old_one_replaces_the_link(
    world: World,
) -> None:
    """The document gives the new column the old one's name, so source_column
    doesn't change; the live column ids still decide which link goes."""
    document = _document(world)
    document.semantic_layer.terms[0].columns_attributes[
        0
    ].column_id = world.customer_ref
    for database in document.data_layer.databases:
        for schema in database.schemas:
            for table in schema.tables:
                for column in table.columns:
                    if column.id == world.customer_ref:
                        column.name = "id"

    mi.apply_import_model(document, replace=True)

    assert _attribute_columns(world.attribute) == {world.customer_ref}


def test_a_created_sql_attribute_embeds_the_name_the_term_kept(world: World) -> None:
    from auto_ontology.server.model_interchange.embed import ImportEmbedBuffer
    from auto_ontology.server.model_interchange.schemas import ModelSqlAttribute

    store().query_write(
        s.term.update().where(s.term.c.id == world.term).values(name_certified=True)
    )
    document = _document(world)
    document.semantic_layer.terms[0].name = f"{world.prefix}-Rejected"
    document.semantic_layer.sql_attributes.manual.append(
        ModelSqlAttribute(
            id=f"{world.prefix}-order-count",
            name=f"{world.prefix}-order-count",
            sql="SELECT COUNT(*) FROM public.orders",
            term_id=document.semantic_layer.terms[0].id,
        )
    )
    buffer = ImportEmbedBuffer()

    mi.apply_import_model(document, replace=True, embed_buffer=buffer)

    texts = [
        r["text"] for r in buffer.semantic_rows if r["text"].startswith("sql_attribute")
    ]
    assert texts and all(f"term: {world.prefix}-Order," in t for t in texts)


def test_a_term_moved_to_another_database_is_rebuilt_there(world: World) -> None:
    """The vectors follow the representation the import commits, not the one
    the Term had when its description changed."""
    from auto_ontology.server.model_interchange.embed import ImportEmbedBuffer

    other_table, _ = _other_database_column(world)
    other_db = store().query_read(
        select(s.catalog_database.c.id).where(
            s.catalog_database.c.name == f"{world.prefix}-other"
        )
    )[0]["id"]
    rows = mi.fetch_export_rows([world.database, other_db])
    document = mi.assemble_export_document(
        rows,
        dialect_by_db_name={
            world.prefix: "postgres",
            f"{world.prefix}-other": "postgres",
        },
        sql_column_resolver=lambda sql, database_name: [],
    )
    term = document.semantic_layer.terms[0]
    term.represents = [other_table]
    term.description = "a returned order"
    buffer = ImportEmbedBuffer()

    mi.apply_import_model(document, replace=True, embed_buffer=buffer)

    represented = {
        row["table_id"]
        for row in store().query_read(
            select(s.table__term.c.table_id).where(
                s.table__term.c.term_id == world.term
            )
        )
    }
    assert represented == {other_table}
    assert world.term in buffer.replaced_semantic_ids
    assert {r["metadata"]["database_name"] for r in _queued(buffer, world.term)} == {
        f"{world.prefix}-other"
    }


def test_a_rename_rebuilds_an_attribute_in_each_of_its_databases(world: World) -> None:
    """One attribute with columns in two databases has a vector in each; a
    scoped rename must rebuild both, since the flush deletes by id."""
    from auto_ontology.server.model_interchange.embed import ImportEmbedBuffer

    _, other_column = _other_database_column(world)
    _link(s.column__has_attribute, column_id=other_column, attribute_id=world.attribute)
    document = _document(world)
    new_name = f"{world.prefix}-Purchase"
    document.semantic_layer.terms[0].name = new_name
    buffer = ImportEmbedBuffer()

    mi.apply_import_model(document, replace=True, embed_buffer=buffer)

    rows = _queued(buffer, world.attribute)
    assert world.attribute in buffer.replaced_semantic_ids
    assert {r["metadata"]["database_name"] for r in rows} == {
        world.prefix,
        f"{world.prefix}-other",
    }
    assert all(new_name in r["text"] for r in rows)


def test_a_term_leaving_a_database_is_reembedded(world: World) -> None:
    """Only the representation changes, but the Term's vector in the database
    it left must go, so the move alone re-embeds it and counts as an update."""
    from auto_ontology.server.model_interchange.embed import ImportEmbedBuffer
    from auto_ontology.server.model_interchange.schemas import ModelSqlAttribute

    other_table, _ = _other_database_column(world)
    _link(s.table__term, table_id=other_table, term_id=world.term)
    other_db = store().query_read(
        select(s.catalog_database.c.id).where(
            s.catalog_database.c.name == f"{world.prefix}-other"
        )
    )[0]["id"]
    rows = mi.fetch_export_rows([world.database, other_db])
    document = mi.assemble_export_document(
        rows,
        dialect_by_db_name={
            world.prefix: "postgres",
            f"{world.prefix}-other": "postgres",
        },
        sql_column_resolver=lambda sql, database_name: [],
    )
    document.semantic_layer.terms[0].represents = [world.orders]
    document.semantic_layer.sql_attributes.manual.append(
        ModelSqlAttribute(
            id=f"{world.prefix}-order-count",
            name=f"{world.prefix}-order-count",
            sql="SELECT COUNT(*) FROM public.orders",
            term_id=document.semantic_layer.terms[0].id,
        )
    )
    mi.apply_import_model(document, replace=False)
    sql_attr = store().query_read(
        select(s.sql_attribute.c.id).where(
            s.sql_attribute.c.name == f"{world.prefix}-order-count"
        )
    )[0]["id"]
    _link(s.table__term, table_id=other_table, term_id=world.term)
    buffer = ImportEmbedBuffer()

    summary = mi.apply_import_model(document, replace=True, embed_buffer=buffer)

    assert summary["updated"]["terms"] == 1
    assert world.term in buffer.replaced_semantic_ids
    # The SqlAttribute is placed by its Term's databases, so it moves too.
    assert sql_attr in buffer.replaced_semantic_ids
    assert {r["metadata"]["database_name"] for r in _queued(buffer, sql_attr)} == {
        world.prefix
    }
    assert {r["metadata"]["database_name"] for r in _queued(buffer, world.term)} == {
        world.prefix
    }

    again = ImportEmbedBuffer()
    repeat = mi.apply_import_model(document, replace=True, embed_buffer=again)
    assert world.term not in again.replaced_semantic_ids
    assert repeat["updated"]["terms"] == 0


def test_an_entity_with_no_buildable_vector_keeps_its_old_one(world: World) -> None:
    """An attribute with no name builds no vector; replacing would delete its
    old one and write nothing."""
    from auto_ontology.server.model_interchange.embed import ImportEmbedBuffer

    store().query_write(
        s.column_attribute.update()
        .where(s.column_attribute.c.id == world.attribute)
        .values(name="")
    )
    document = _document(world)
    document.semantic_layer.terms[0].name = f"{world.prefix}-Purchase"
    buffer = ImportEmbedBuffer()

    mi.apply_import_model(document, replace=True, embed_buffer=buffer)

    assert world.term in buffer.replaced_semantic_ids
    assert world.attribute not in buffer.replaced_semantic_ids
