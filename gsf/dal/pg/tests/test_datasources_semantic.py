# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The seven `datasources` functions that reach the semantic tier.

These landed a phase after the other 19 (DECISION-009) precisely because they
could not be checked without terms and attributes to join to. So they get a
hand-built fixture rather than the Pagila ingest: a catalog with exactly the
semantic shapes that make each join distinguishable from a wrong one.

Pagila is not enough on its own for the case that matters most.
``fetch_bridge_table_candidates`` returns **nothing** on Pagila — its junction
tables (``film_actor``, ``film_category``) each carry a ``last_update`` column,
so no table has an FK on every column. The golden capture agrees: Neo4j returned
``[]`` too. Two empty lists agreeing tells you nothing about the query, which is
why the bridge fixture below is built by hand.
"""

from __future__ import annotations

import os
import uuid

import pytest

pytest.importorskip("sqlalchemy")

from sqlalchemy import select  # noqa: E402

from gsf.dal.pg import datasources as d  # noqa: E402
from gsf.dal.pg import schema as s  # noqa: E402
from gsf.dal.pg.session import store  # noqa: E402
from gsf.semantic.constants import (  # noqa: E402
    SQL_ATTR_SOURCE_BRIDGE,
    SQL_ATTR_SOURCE_MANUAL,
)


@pytest.fixture(scope="module", autouse=True)
def require_schema():
    if not os.environ.get("POSTGRES_USER"):
        pytest.skip("POSTGRES_* not set")
    try:
        store().query_read(f"SELECT 1 FROM {s.SCHEMA}.column_attribute LIMIT 1")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"gsf schema unavailable (alembic upgrade head): {exc}")


def _add(table, **values) -> str:
    return store().query_write(table.insert().values(**values).returning(table.c.id))[
        0
    ]["id"]


def _link(table, **values) -> None:
    store().query_write(table.insert().values(**values))


class World:
    """A tiny catalog, built so each join has something to get wrong.

    ``orders`` and ``customers`` are ordinary tables. ``order_tag`` is a pure
    junction: two columns, both foreign keys, nothing else — the shape
    ``fetch_bridge_table_candidates`` is looking for.
    """

    def __init__(self, prefix: str) -> None:
        self.prefix = prefix
        self.db = _add(s.catalog_database, name=prefix)
        self.schema = _add(s.catalog_schema, database_id=self.db, name="shop")
        self.tables: dict[str, str] = {}
        self.columns: dict[str, str] = {}

    def table(self, name: str, **kwargs) -> str:
        tid = _add(s.catalog_table, schema_id=self.schema, name=name, **kwargs)
        self.tables[name] = tid
        return tid

    def column(self, table: str, name: str, position: int = 1, **kwargs) -> str:
        cid = _add(
            s.catalog_column,
            table_id=self.tables[table],
            name=name,
            ordinal_position=position,
            **kwargs,
        )
        self.columns[f"{table}.{name}"] = cid
        return cid

    def term(self, name: str, description: str | None = None) -> str:
        return _add(s.term, name=f"{self.prefix}-{name}", description=description)

    def attribute(self, name: str, description: str | None = None) -> str:
        return _add(
            s.column_attribute,
            name=f"{self.prefix}-{name}",
            description=description,
            source_column=name,
            term_name=name,
            table_id="",
        )


@pytest.fixture
def world():
    w = World(f"t-{uuid.uuid4().hex[:8]}")

    w.table("customers", table_type="base table", description="People who buy")
    w.column("customers", "customer_id", 1)
    w.column("customers", "name", 2)

    w.table("orders", table_type="base table")
    w.column("orders", "order_id", 1)
    w.column("orders", "customer_id", 2)
    w.column("orders", "total", 3)

    # A pure junction: every column is a foreign key, and nothing else.
    w.table("order_tag", table_type="base table")
    w.column("order_tag", "order_id", 1)
    w.column("order_tag", "tag_id", 2)

    w.table("tags", table_type="base table")
    w.column("tags", "tag_id", 1)

    _link(
        s.column_foreign_key,
        source_column_id=w.columns["orders.customer_id"],
        target_column_id=w.columns["customers.customer_id"],
    )
    _link(
        s.column_foreign_key,
        source_column_id=w.columns["order_tag.order_id"],
        target_column_id=w.columns["orders.order_id"],
    )
    _link(
        s.column_foreign_key,
        source_column_id=w.columns["order_tag.tag_id"],
        target_column_id=w.columns["tags.tag_id"],
    )

    yield w

    store().query_write(
        s.catalog_database.delete().where(s.catalog_database.c.name == w.prefix)
    )
    for table in (s.term, s.column_attribute, s.sql_attribute):
        store().query_write(table.delete().where(table.c.name.like(f"{w.prefix}%")))


def _by_name(rows: list[dict]) -> dict[str, dict]:
    return {r["name"]: r for r in rows}


# --------------------------------------------------------------------------
# fetch_tables_for_schema
# --------------------------------------------------------------------------


def test_tables_for_schema_counts_columns_and_terms(world) -> None:
    """The Term count is a union of two routes, deduplicated.

    A table reaches a Term directly (`REPRESENTS`) and through its columns'
    attributes. Counting only one route, or counting the union with duplicates,
    both produce a plausible-looking number — so the fixture makes one Term
    reachable *both* ways and asserts it is counted once.
    """
    shared = world.term("Customer", "a buyer")
    other = world.term("Contact")
    _link(s.table_term, table_id=world.tables["customers"], term_id=shared)

    attribute = world.attribute("customer-name")
    _link(
        s.column_has_attribute,
        column_id=world.columns["customers.name"],
        attribute_id=attribute,
    )
    # Same Term as the direct link -> must not count twice.
    _link(s.column_attribute_term, attribute_id=attribute, term_id=shared)
    # A second, distinct Term via the SEMANTIC_FK route.
    fk_attribute = world.attribute("customer-ref")
    _link(
        s.column_semantic_fk,
        column_id=world.columns["customers.customer_id"],
        attribute_id=fk_attribute,
    )
    _link(s.column_attribute_term, attribute_id=fk_attribute, term_id=other)

    rows = _by_name(d.fetch_tables_for_schema(world.schema))
    assert rows["customers"]["terms_count"] == 2
    assert rows["customers"]["columns_count"] == 2
    assert rows["orders"]["terms_count"] == 0
    assert rows["orders"]["columns_count"] == 3


def test_tables_for_schema_counts_sql(world) -> None:
    query = _add(s.sql_query, sql_full_query=f"SELECT 1 -- {world.prefix}")
    _link(s.sql_query_table, sql_query_id=query, table_id=world.tables["orders"])

    rows = _by_name(d.fetch_tables_for_schema(world.schema))
    assert rows["orders"]["sql_count"] == 1
    assert rows["customers"]["sql_count"] == 0

    store().query_write(s.sql_query.delete().where(s.sql_query.c.id == query))


def test_tables_for_schema_uses_the_description_fallback(world) -> None:
    """`orders` has no description of its own; its Term supplies one."""
    term = world.term("Order", "a purchase")
    _link(s.table_term, table_id=world.tables["orders"], term_id=term)

    rows = _by_name(d.fetch_tables_for_schema(world.schema))
    assert rows["orders"]["description"] == "a purchase"
    # `customers` has its own, which outranks any Term.
    assert rows["customers"]["description"] == "People who buy"


def test_tables_for_schema_is_ordered_by_name(world) -> None:
    """Compared against the *database's* sort, not Python's.

    They disagree: `order_tag` sorts before `orders` by codepoint and after it
    under a locale collation that ignores punctuation at the first level. The
    contract is "ordered by name", and the database is what defines that order.
    """
    names = [r["name"] for r in d.fetch_tables_for_schema(world.schema)]
    expected = [
        r["name"]
        for r in store().query_read(
            select(s.catalog_table.c.name)
            .where(s.catalog_table.c.schema_id == world.schema)
            .order_by(s.catalog_table.c.name)
        )
    ]
    assert names == expected
    assert len(names) > 1


def test_tables_for_schema_ignores_database_name(world) -> None:
    """Accepted for call compatibility; schema ids are globally unique.

    Pinned because the argument is silently unused — a future reader would
    otherwise be right to assume it filters, and wrong.
    """
    assert d.fetch_tables_for_schema(
        world.schema, database_name="not-this-database"
    ) == d.fetch_tables_for_schema(world.schema)


# --------------------------------------------------------------------------
# fetch_all_tables_without_term
# --------------------------------------------------------------------------


def test_tables_without_term_excludes_termed_tables(world) -> None:
    before = {r["name"] for r in d.fetch_all_tables_without_term(world.prefix)}
    assert before == {"customers", "orders", "order_tag", "tags"}

    term = world.term("Order")
    _link(s.table_term, table_id=world.tables["orders"], term_id=term)

    after = {r["name"] for r in d.fetch_all_tables_without_term(world.prefix)}
    assert after == before - {"orders"}


def test_tables_without_term_scopes_to_one_database(world) -> None:
    """Unscoped means every database, not the default one.

    Several databases share a store, and each compile pass tags its embeddings
    with a database name — so an unscoped pass would file one database's tables
    under another's.
    """
    scoped = d.fetch_all_tables_without_term(world.prefix)
    everything = d.fetch_all_tables_without_term()
    assert len(everything) > len(scoped)
    assert {r["name"] for r in scoped} <= {r["name"] for r in everything}


def test_tables_without_term_reads_the_tables_own_description(world) -> None:
    """No Term means no Term description to fall back to, by construction."""
    rows = _by_name(d.fetch_all_tables_without_term(world.prefix))
    assert rows["customers"]["description"] == "People who buy"
    assert rows["orders"]["description"] is None


# --------------------------------------------------------------------------
# fetch_columns_for_table
# --------------------------------------------------------------------------


def test_columns_for_table_returns_every_column_by_default(world) -> None:
    table = d.fetch_columns_for_table(world.tables["orders"])
    assert table["table_name"] == "orders"
    assert table["schema_name"] == "shop"
    assert table["database_name"] == world.prefix
    assert [c["column_name"] for c in table["columns"]] == [
        "order_id",
        "customer_id",
        "total",
    ]


def test_columns_for_table_pages_in_ordinal_order(world) -> None:
    first = d.fetch_columns_for_table(world.tables["orders"], limit=2)
    second = d.fetch_columns_for_table(world.tables["orders"], skip=2, limit=2)
    assert [c["column_name"] for c in first["columns"]] == ["order_id", "customer_id"]
    assert [c["column_name"] for c in second["columns"]] == ["total"]


def test_a_page_past_the_end_is_not_a_missing_table(world) -> None:
    """The distinction the two-query split exists to preserve.

    A single join with OFFSET would return no rows here, and the caller would
    read "page 3 of a 2-page table" as "no such table".
    """
    table = d.fetch_columns_for_table(world.tables["orders"], skip=99)
    assert table is not None
    assert table["table_name"] == "orders"
    assert table["columns"] == []


def test_columns_for_table_returns_none_for_a_missing_table() -> None:
    assert d.fetch_columns_for_table("no-such-table") is None


def test_columns_for_table_uses_the_description_fallback(world) -> None:
    attribute = world.attribute("order-total", "money, in cents")
    _link(
        s.column_has_attribute,
        column_id=world.columns["orders.total"],
        attribute_id=attribute,
    )
    columns = {
        c["column_name"]: c
        for c in d.fetch_columns_for_table(world.tables["orders"])["columns"]
    }
    assert columns["total"]["description"] == "money, in cents"
    assert columns["order_id"]["description"] is None


def test_columns_for_table_parses_sample_values(world) -> None:
    """Stored as a JSON string; callers expect a list."""
    store().query_write(
        s.catalog_column.update()
        .where(s.catalog_column.c.id == world.columns["orders.total"])
        .values(sample_values='["1", "2"]')
    )
    columns = {
        c["column_name"]: c
        for c in d.fetch_columns_for_table(world.tables["orders"])["columns"]
    }
    assert columns["total"]["sample_values"] == ["1", "2"]


# --------------------------------------------------------------------------
# fetch_tables_by_ids
# --------------------------------------------------------------------------


def test_tables_by_ids_nests_column_summaries(world) -> None:
    rows = {
        r["name"]: r
        for r in d.fetch_tables_by_ids(
            [world.tables["orders"], world.tables["customers"]]
        )
    }
    assert set(rows) == {"orders", "customers"}
    assert rows["orders"]["label"] == "Table"
    assert [c["name"] for c in rows["orders"]["columns"]] == [
        "order_id",
        "customer_id",
        "total",
    ]


def test_tables_by_ids_is_empty_for_an_empty_list() -> None:
    assert d.fetch_tables_by_ids([]) == []


def test_tables_by_ids_drops_a_table_with_no_columns(world) -> None:
    """Preserved from the Cypher, whose second MATCH was an inner join.

    Not fixed here: a column-less table in the catalog is a symptom worth seeing
    where it originates, not something to paper over in a read.
    """
    empty = world.table("empty_table")
    assert d.fetch_tables_by_ids([empty]) == []


def test_tables_by_ids_normalises_absent_strings_to_empty(world) -> None:
    """Callers concatenate these into prompts; None would render as "None"."""
    row = d.fetch_tables_by_ids([world.tables["orders"]])[0]
    assert row["description"] == ""


# --------------------------------------------------------------------------
# fetch_table_context
# --------------------------------------------------------------------------


def test_table_context_returns_columns_and_outgoing_fks(world) -> None:
    context = d.fetch_table_context(world.tables["orders"])
    assert [c["name"] for c in context["columns"]] == [
        "order_id",
        "customer_id",
        "total",
    ]
    assert context["fks"] == [
        {
            "source_column": "customer_id",
            "target_column": "customer_id",
            "target_table": "customers",
            "target_table_id": world.tables["customers"],
        }
    ]


def test_table_context_fks_are_outgoing_only(world) -> None:
    """`customers` is *pointed at* by orders, and has no FKs of its own.

    Reversing the direction would look correct on any self-referential table
    and wrong everywhere else, so it is worth one explicit assertion.
    """
    assert d.fetch_table_context(world.tables["customers"])["fks"] == []


def test_table_context_of_a_missing_table_is_empty_not_an_error() -> None:
    assert d.fetch_table_context("no-such-table") == {"columns": [], "fks": []}


# --------------------------------------------------------------------------
# fetch_tables_and_columns_by_node_ids
# --------------------------------------------------------------------------


def test_a_table_id_pulls_in_all_of_its_columns(world) -> None:
    tables, columns, database = d.fetch_tables_and_columns_by_node_ids(
        [world.tables["orders"]]
    )
    assert list(tables["table_name"]) == ["orders"]
    assert sorted(columns["column_name"]) == ["customer_id", "order_id", "total"]
    assert database == world.prefix


def test_a_column_id_pulls_in_only_itself(world) -> None:
    tables, columns, database = d.fetch_tables_and_columns_by_node_ids(
        [world.columns["orders.total"]]
    )
    assert tables.empty
    assert list(columns["column_name"]) == ["total"]
    # The database name still resolves, from the column frame.
    assert database == world.prefix


def test_no_ids_yields_empty_frames_and_no_database() -> None:
    tables, columns, database = d.fetch_tables_and_columns_by_node_ids([])
    assert tables.empty and columns.empty
    assert database == ""


def test_node_ids_uses_the_fallback_for_columns_but_not_tables(world) -> None:
    """Asymmetric in the Cypher, and preserved.

    These frames feed embeddings. Widening what gets embedded — by applying the
    table fallback here too — would shift retrieval results with no test
    anywhere failing.
    """
    table_term = world.term("Order", "a purchase")
    _link(s.table_term, table_id=world.tables["orders"], term_id=table_term)
    attribute = world.attribute("order-total", "money, in cents")
    _link(
        s.column_has_attribute,
        column_id=world.columns["orders.total"],
        attribute_id=attribute,
    )

    tables, columns, _ = d.fetch_tables_and_columns_by_node_ids(
        [world.tables["orders"]]
    )
    assert tables.iloc[0]["description"] is None
    by_name = columns.set_index("column_name")["description"].to_dict()
    assert by_name["total"] == "money, in cents"


# --------------------------------------------------------------------------
# fetch_bridge_table_candidates
# --------------------------------------------------------------------------


def test_a_pure_junction_table_is_a_candidate(world) -> None:
    rows = d.fetch_bridge_table_candidates(world.prefix)
    assert [r["table_name"] for r in rows] == ["order_tag"]

    pairs = sorted(rows[0]["fk_pairs"], key=lambda p: p["source_column"])
    assert pairs == [
        {
            "source_column": "order_id",
            "target_table": "orders",
            "target_schema": "shop",
            "target_column": "order_id",
            "target_table_id": world.tables["orders"],
        },
        {
            "source_column": "tag_id",
            "target_table": "tags",
            "target_schema": "shop",
            "target_column": "tag_id",
            "target_table_id": world.tables["tags"],
        },
    ]


def test_one_non_fk_column_disqualifies_the_table(world) -> None:
    """Exactly what rules out Pagila's `film_actor`: a `last_update` column."""
    world.column("order_tag", "last_update", 3)
    assert d.fetch_bridge_table_candidates(world.prefix) == []


def test_an_existing_attribute_disqualifies_the_table(world) -> None:
    """A column already carrying meaning is not raw join plumbing."""
    attribute = world.attribute("tag-ref")
    _link(
        s.column_has_attribute,
        column_id=world.columns["order_tag.tag_id"],
        attribute_id=attribute,
    )
    assert d.fetch_bridge_table_candidates(world.prefix) == []


def test_an_existing_bridge_disqualifies_the_table(world) -> None:
    """Without this the suggester re-proposes a bridge it already created."""
    query = _add(s.sql_query, sql_full_query=f"SELECT 2 -- {world.prefix}")
    _link(s.sql_query_table, sql_query_id=query, table_id=world.tables["order_tag"])
    attribute = _add(
        s.sql_attribute,
        name=f"{world.prefix}-bridge",
        source=SQL_ATTR_SOURCE_BRIDGE,
    )
    _link(s.sql_attribute_sql, attribute_id=attribute, sql_query_id=query)

    assert d.fetch_bridge_table_candidates(world.prefix) == []

    store().query_write(s.sql_query.delete().where(s.sql_query.c.id == query))


def test_a_bridge_from_some_other_source_does_not_disqualify(world) -> None:
    """The source filter is load-bearing; any SqlAttribute would over-exclude."""
    query = _add(s.sql_query, sql_full_query=f"SELECT 3 -- {world.prefix}")
    _link(s.sql_query_table, sql_query_id=query, table_id=world.tables["order_tag"])
    attribute = _add(
        s.sql_attribute, name=f"{world.prefix}-manual", source=SQL_ATTR_SOURCE_MANUAL
    )
    _link(s.sql_attribute_sql, attribute_id=attribute, sql_query_id=query)

    assert [r["table_name"] for r in d.fetch_bridge_table_candidates(world.prefix)] == [
        "order_tag"
    ]

    store().query_write(s.sql_query.delete().where(s.sql_query.c.id == query))


def test_a_semantic_fk_counts_as_a_foreign_key(world) -> None:
    """SEMANTIC_FK stands in for a constraint the source database never declared.

    Its direction is the subtle part: the column points at a ColumnAttribute
    that some *other* column owns, and that other column is the join target.
    Read backwards, every bridge would point at itself — which this asserts is
    not what happens.
    """
    world.table("order_note", table_type="base table")
    world.column("order_note", "order_ref", 1)
    world.column("order_note", "tag_ref", 2)

    _link(
        s.column_foreign_key,
        source_column_id=world.columns["order_note.order_ref"],
        target_column_id=world.columns["orders.order_id"],
    )
    # No real constraint for tag_ref -- only a semantic one, owned by tags.tag_id.
    attribute = world.attribute("tag-identity")
    _link(
        s.column_has_attribute,
        column_id=world.columns["tags.tag_id"],
        attribute_id=attribute,
    )
    _link(
        s.column_semantic_fk,
        column_id=world.columns["order_note.tag_ref"],
        attribute_id=attribute,
    )

    rows = {r["table_name"]: r for r in d.fetch_bridge_table_candidates(world.prefix)}
    assert "order_note" in rows
    semantic = next(
        p for p in rows["order_note"]["fk_pairs"] if p["source_column"] == "tag_ref"
    )
    assert semantic["target_table"] == "tags"
    assert semantic["target_column"] == "tag_id"


def test_a_self_referential_bridge_is_allowed(world) -> None:
    """`also_buy(product_id, also_buy_product_id)` — both ends the same table."""
    world.table("also_order", table_type="base table")
    world.column("also_order", "order_id", 1)
    world.column("also_order", "also_order_id", 2)
    for column in ("order_id", "also_order_id"):
        _link(
            s.column_foreign_key,
            source_column_id=world.columns[f"also_order.{column}"],
            target_column_id=world.columns["orders.order_id"],
        )

    rows = {r["table_name"]: r for r in d.fetch_bridge_table_candidates(world.prefix)}
    assert "also_order" in rows
    assert {p["target_table"] for p in rows["also_order"]["fk_pairs"]} == {"orders"}
    assert len(rows["also_order"]["fk_pairs"]) == 2


def test_a_single_column_table_is_not_a_bridge(world) -> None:
    """`tags` is all-FK-free but more to the point has one column."""
    world.table("solo", table_type="base table")
    world.column("solo", "order_id", 1)
    _link(
        s.column_foreign_key,
        source_column_id=world.columns["solo.order_id"],
        target_column_id=world.columns["orders.order_id"],
    )
    names = [r["table_name"] for r in d.fetch_bridge_table_candidates(world.prefix)]
    assert "solo" not in names


def test_an_fk_pointing_outside_the_catalog_disqualifies(world) -> None:
    """The second of the Cypher's two size checks, and the reason both exist.

    A column with a `column_foreign_key` row whose target column is gone passes
    "every column has an outgoing edge" and fails "every edge lands somewhere".
    Here the target column is deleted after the fact, which is what a partial
    ingest leaves behind.
    """
    orphan = world.column("tags", "external_id", 2)
    world.table("tag_link", table_type="base table")
    world.column("tag_link", "tag_id", 1)
    world.column("tag_link", "other_id", 2)
    _link(
        s.column_foreign_key,
        source_column_id=world.columns["tag_link.tag_id"],
        target_column_id=world.columns["tags.tag_id"],
    )
    _link(
        s.column_foreign_key,
        source_column_id=world.columns["tag_link.other_id"],
        target_column_id=orphan,
    )
    assert "tag_link" in [
        r["table_name"] for r in d.fetch_bridge_table_candidates(world.prefix)
    ]

    # The FK row cascades away with its target, leaving other_id unresolvable.
    store().query_write(
        s.catalog_column.delete().where(s.catalog_column.c.id == orphan)
    )
    assert "tag_link" not in [
        r["table_name"] for r in d.fetch_bridge_table_candidates(world.prefix)
    ]
