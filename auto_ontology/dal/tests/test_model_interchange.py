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
from datetime import date
from decimal import Decimal

import pytest
import yaml

pytest.importorskip("sqlalchemy")

from sqlalchemy import inspect, select  # noqa: E402

from auto_ontology.dal import model_interchange as mi  # noqa: E402
from auto_ontology.dal import schema as s  # noqa: E402
from auto_ontology.dal.session import get_engine, store  # noqa: E402
from auto_ontology.semantic.constants import SEMANTIC_SOURCE  # noqa: E402
from auto_ontology.server.model_interchange import schemas as models  # noqa: E402
from auto_ontology.server.model_interchange import service  # noqa: E402


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


@pytest.fixture
def governed_result_source():
    """Source-owned facts and membership live outside the exchanged model."""
    scope = f"governed_{uuid.uuid4().hex[:10]}"
    with get_engine().connect() as source:
        # PostgreSQL rolls back this schema and its contents when the connection closes.
        source.exec_driver_sql(f"CREATE SCHEMA {scope}")
        source.exec_driver_sql(f"""
            CREATE TABLE {scope}.entity (id integer PRIMARY KEY);
            CREATE TABLE {scope}.membership (
                producer text, run_id text, entity_id integer REFERENCES {scope}.entity,
                PRIMARY KEY (producer, run_id, entity_id)
            );
            CREATE TABLE {scope}.result (
                id text PRIMARY KEY, producer text NOT NULL, run_id text NOT NULL,
                entity_id integer NOT NULL REFERENCES {scope}.entity,
                starts date NOT NULL, ends date NOT NULL, amount numeric NOT NULL,
                unit text NOT NULL, status text NOT NULL, scenario text NOT NULL,
                commitment text NOT NULL, evidence text NOT NULL,
                predecessor text REFERENCES {scope}.result
            );
            INSERT INTO {scope}.entity VALUES (1), (2), (3);
            INSERT INTO {scope}.membership VALUES
                ('a', 'r1', 1), ('a', 'r1', 2), ('a', 'r2', 1), ('b', 'r1', 1);
            INSERT INTO {scope}.result VALUES
                ('old', 'a', 'r1', 1, '2026-01-01', '2026-01-03', 99,
                    'kg', 'superseded', 'base', 'published', 'modeled', NULL),
                ('qty', 'a', 'r1', 1, '2026-01-01', '2026-01-03', 3,
                    'kg', 'current', 'base', 'published', 'modeled', 'old'),
                ('rate', 'a', 'r1', 1, '2026-01-03', '2026-01-08', 250,
                    'g/day', 'current', 'base', 'published', 'modeled', NULL),
                ('entity', 'a', 'r1', 2, '2026-01-01', '2026-01-03', 7,
                    'kg', 'current', 'base', 'published', 'modeled', NULL),
                ('run', 'a', 'r2', 1, '2026-01-01', '2026-01-03', 8,
                    'kg', 'current', 'base', 'published', 'modeled', NULL),
                ('producer', 'b', 'r1', 1, '2026-01-01', '2026-01-03', 9,
                    'kg', 'current', 'base', 'published', 'modeled', NULL),
                ('invalid', 'a', 'r1', 1, '2026-01-08', '2026-01-09', 99,
                    'kg', 'invalid', 'base', 'published', 'modeled', NULL),
                ('duplicate-a', 'a', 'r1', 1, '2026-01-01', '2026-01-08', 99,
                    'kg', 'current', 'base', 'published', 'modeled', NULL),
                ('duplicate-b', 'a', 'r1', 1, '2026-01-01', '2026-01-08', 100,
                    'kg', 'current', 'base', 'published', 'modeled', NULL),
                ('no-member', 'a', 'r1', 3, '2026-01-01', '2026-01-03', 99,
                    'kg', 'current', 'base', 'published', 'modeled', NULL),
                ('wrong-unit', 'a', 'r1', 1, '2026-01-10', '2026-01-11', 99,
                    'm', 'current', 'base', 'published', 'modeled', NULL),
                ('empty-window', 'a', 'r1', 1, '2026-01-11', '2026-01-11', 99,
                    'kg', 'current', 'base', 'published', 'modeled', NULL),
                ('scenario', 'a', 'r1', 1, '2026-01-01', '2026-01-03', 99,
                    'kg', 'current', 'alternate', 'published', 'modeled', NULL),
                ('unpublished', 'a', 'r1', 1, '2026-01-11', '2026-01-12', 99,
                    'kg', 'current', 'base', 'draft', 'modeled', NULL),
                ('wrong-evidence', 'a', 'r1', 1, '2026-01-12', '2026-01-13', 99,
                    'kg', 'current', 'base', 'published', 'estimate', NULL);
        """)
        try:
            yield scope, source
        finally:
            _cleanup(scope)


def _governed_result_model(scope, source):
    catalog = inspect(source)
    tables = []
    foreign_keys = []
    for name in ("entity", "membership", "result"):
        table_id = f"{scope}.{name}"
        tables.append(
            models.ModelTable(
                id=table_id,
                name=name,
                pk=catalog.get_pk_constraint(name, schema=scope)["constrained_columns"],
                type="base table",
                columns=[
                    models.ModelColumn(
                        id=f"{table_id}.{column['name']}",
                        name=column["name"],
                        type=str(column["type"]).lower(),
                        is_nullable=column["nullable"],
                    )
                    for column in catalog.get_columns(name, schema=scope)
                ],
            )
        )
        foreign_keys.extend(
            models.ModelForeignKey(
                source_column_id=f"{table_id}.{column}",
                target_column_id=f"{scope}.{fk['referred_table']}.{target}",
            )
            for fk in catalog.get_foreign_keys(name, schema=scope)
            for column, target in zip(
                fk["constrained_columns"], fk["referred_columns"], strict=True
            )
        )

    result = f"{scope}.result"
    # The source contract selects a scenario, commitment and evidence class.
    # Two compatible rows at the same grain are ambiguous, so discard both.
    sql = f"""
        WITH eligible AS (
            SELECT r.id, r.producer, r.run_id, r.entity_id, r.starts, r.ends,
                   r.predecessor, r.evidence,
                   CASE r.unit
                       WHEN 'kg' THEN r.amount
                       WHEN 'g/day' THEN r.amount * (r.ends - r.starts) / 1000
                   END AS quantity_kg,
                   COUNT(*) OVER (
                       PARTITION BY r.producer, r.run_id, r.entity_id, r.starts, r.ends
                   ) AS candidates
            FROM {result} AS r
            JOIN {scope}.membership AS m
              ON r.producer = m.producer AND r.run_id = m.run_id AND r.entity_id = m.entity_id
            WHERE r.status = 'current' AND r.scenario = 'base'
              AND r.commitment = 'published' AND r.evidence = 'modeled'
              AND r.starts < r.ends AND r.unit IN ('kg', 'g/day')
        )
        SELECT id, producer, run_id, entity_id, starts, ends, predecessor, evidence,
               quantity_kg,
               SUM(quantity_kg) OVER (PARTITION BY producer, run_id, entity_id) AS horizon_kg
        FROM eligible
        WHERE candidates = 1
        ORDER BY producer, run_id, entity_id, starts, ends
    """.strip()
    return models.AutoOntologyModelDocument(
        data_layer=models.ModelDataLayer(
            databases=[
                models.ModelDatabase(
                    id=f"{scope}-db",
                    dialect="postgres",
                    schemas=[
                        models.ModelSchema(
                            id=f"{scope}-schema",
                            name=scope,
                            database_name=scope,
                            tables=tables,
                        )
                    ],
                )
            ],
            foreign_keys=foreign_keys,
            joins=[
                models.ModelJoin(
                    source_table_id=result,
                    target_table_id=f"{scope}.membership",
                    join_columns=[
                        {"source": key, "target": key}
                        for key in ("producer", "run_id", "entity_id")
                    ],
                )
            ],
        ),
        semantic_layer=models.ModelSemanticLayer(
            terms=[
                models.ModelTerm(
                    id=f"{scope}-term", name=f"{scope} Result", represents=[result]
                )
            ],
            custom_analyses=[
                models.ModelCustomAnalysis(
                    id=f"{scope}-analysis",
                    name=f"{scope} Published quantity",
                    description="Modeled quantities in kilograms over source windows [starts, ends).",
                    sql=sql,
                    sql_column_is=[f"{result}.id"],
                )
            ],
        ),
    )


def test_governed_result_round_trip_preserves_keyed_quantities(
    governed_result_source,
) -> None:
    scope, source = governed_result_source
    model = _governed_result_model(scope, source)
    first = mi.apply_import_model(model, replace=False)
    exported = service.export_model(
        models.ExportRequest(databases=first["database_ids"])
    )
    # Remove just the catalog/semantic scope. The source facts stay available.
    _cleanup(scope)
    second = service.import_model(exported, replace=False, embed=False)
    assert second["created"]["custom_analyses"] == 1
    rebuilt = models.AutoOntologyModelDocument.model_validate(
        yaml.safe_load(
            service.export_model(models.ExportRequest(databases=second["database_ids"]))
        )
    )
    (database,) = rebuilt.data_layer.databases
    (schema,) = database.schemas
    columns = {c.id: (t.name, c.name, c.type) for t in schema.tables for c in t.columns}
    assert {
        (columns[fk.source_column_id], columns[fk.target_column_id])
        for fk in rebuilt.data_layer.foreign_keys
    } == {
        (("membership", "entity_id", "integer"), ("entity", "id", "integer")),
        (("result", "entity_id", "integer"), ("entity", "id", "integer")),
        (("result", "predecessor", "text"), ("result", "id", "text")),
    }
    tables = {table.id: table.name for table in schema.tables}
    assert {table.name: table.pk for table in schema.tables} == {
        "entity": ["id"],
        "membership": ["producer", "run_id", "entity_id"],
        "result": ["id"],
    }
    (join,) = rebuilt.data_layer.joins
    assert (tables[join.source_table_id], tables[join.target_table_id]) == (
        "result",
        "membership",
    )
    assert join.join_columns == model.data_layer.joins[0].join_columns
    (term,) = rebuilt.semantic_layer.terms
    assert [tables[table_id] for table_id in term.represents] == ["result"]
    (analysis,) = rebuilt.semantic_layer.custom_analyses
    bound_columns = {columns[column_id][:2] for column_id in analysis.sql_column_is}
    assert {
        ("membership", "producer"),
        ("membership", "run_id"),
        ("membership", "entity_id"),
        ("result", "amount"),
        ("result", "starts"),
        ("result", "ends"),
        ("result", "predecessor"),
    } <= bound_columns

    jan_1, jan_3, jan_8 = (date(2026, 1, day) for day in (1, 3, 8))
    rows = source.exec_driver_sql(analysis.sql).all()
    assert [
        (r.id, r.producer, r.run_id, r.entity_id, r.starts, r.ends) for r in rows
    ] == [
        ("qty", "a", "r1", 1, jan_1, jan_3),
        ("rate", "a", "r1", 1, jan_3, jan_8),
        ("entity", "a", "r1", 2, jan_1, jan_3),
        ("run", "a", "r2", 1, jan_1, jan_3),
        ("producer", "b", "r1", 1, jan_1, jan_3),
    ]
    assert [r.predecessor for r in rows] == ["old", None, None, None, None]
    assert {r.evidence for r in rows} == {"modeled"}
    assert [(r.quantity_kg, r.horizon_kg) for r in rows] == [
        (Decimal("3"), Decimal("4.25")),
        (Decimal("1.25"), Decimal("4.25")),
        (Decimal("7"), Decimal("7")),
        (Decimal("8"), Decimal("8")),
        (Decimal("9"), Decimal("9")),
    ]
