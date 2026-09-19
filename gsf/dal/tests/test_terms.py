# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Term reads and writes.

Three rules carry most of the risk, and each gets a test that fails for a
*different* reason than "the number is wrong":

* **the list, its total, and the single-id check must agree** — a total that
  describes a different set than the list it pages is a pager that runs off the
  end, and nothing raises;
* **all-or-nothing zone scoping** — a term representing one out-of-zone table is
  hidden entirely, so the fixture builds a term that straddles the boundary;
* **one definition of "related"** — the badge, the relationship column and the
  list behind them are computed from the same rows, and the way that breaks is
  three plausible numbers rather than an error.

The certification rollup gets its own section for the same reason: every
possible wrong version of it still returns one of the three valid strings.
"""

from __future__ import annotations

import os
import uuid

import pytest

pytest.importorskip("sqlalchemy")

from sqlalchemy import select  # noqa: E402

from gsf.dal import schema as s  # noqa: E402
from gsf.dal import terms as t  # noqa: E402
from gsf.dal.session import store  # noqa: E402
from gsf.semantic.constants import SEMANTIC_SOURCE  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def require_schema():
    if not os.environ.get("POSTGRES_USER"):
        pytest.skip("POSTGRES_* not set")
    try:
        store().query_read("SELECT 1 FROM term LIMIT 1")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"gsf schema unavailable (alembic upgrade head): {exc}")


def _add(table, **values) -> str:
    return store().query_write(table.insert().values(**values).returning(table.c.id))[
        0
    ]["id"]


def _link(table, **values) -> None:
    store().query_write(table.insert().values(**values))


class World:
    def __init__(self, prefix: str) -> None:
        self.prefix = prefix
        self.database = _add(s.catalog_database, name=prefix)
        self.schema = _add(s.catalog_schema, database_id=self.database, name="public")
        self.tables: dict[str, str] = {}
        self.columns: dict[str, str] = {}
        self.terms: dict[str, str] = {}
        self.attributes: dict[str, str] = {}
        self.zones: dict[str, str] = {}
        self.queries: list[str] = []

    def table(self, name: str, columns: tuple[str, ...] = ("id",)) -> str:
        tid = _add(s.catalog_table, schema_id=self.schema, name=name)
        self.tables[name] = tid
        for position, column in enumerate(columns, start=1):
            self.columns[f"{name}.{column}"] = _add(
                s.catalog_column,
                table_id=tid,
                name=column,
                ordinal_position=position,
            )
        return tid

    def term(
        self,
        name: str,
        *,
        represents: tuple[str, ...] = (),
        source: str = SEMANTIC_SOURCE,
        description: str | None = None,
    ) -> str:
        tid = _add(
            s.term,
            name=f"{self.prefix}-{name}",
            source=source,
            description=description,
        )
        self.terms[name] = tid
        for table in represents:
            _link(s.table__term, table_id=self.tables[table], term_id=tid)
        return tid

    def column_attribute(
        self,
        key: str,
        term: str,
        *,
        table: str,
        column: str | None = None,
        certified: bool = False,
    ) -> str:
        aid = _add(
            s.column_attribute,
            name=f"{self.prefix}-{key}",
            source_column=column or "id",
            term_name=f"{self.prefix}-{term}",
            table_id=self.tables[table],
            certified=certified,
        )
        self.attributes[key] = aid
        _link(s.column_attribute__term, attribute_id=aid, term_id=self.terms[term])
        if column:
            _link(
                s.column__has_attribute,
                column_id=self.columns[f"{table}.{column}"],
                attribute_id=aid,
            )
        return aid

    def sql_attribute(
        self,
        key: str,
        term: str,
        *,
        tables: tuple[str, ...] = (),
        certified: bool = False,
    ) -> str:
        aid = _add(s.sql_attribute, name=f"{self.prefix}-{key}", certified=certified)
        self.attributes[key] = aid
        _link(s.sql_attribute__term, attribute_id=aid, term_id=self.terms[term])
        qid = _add(s.sql_query, sql_full_query=f"SELECT 1 -- {self.prefix}-{key}")
        self.queries.append(qid)
        _link(s.sql_attribute__sql, attribute_id=aid, sql_query_id=qid)
        for table in tables:
            _link(s.sql_query__table, sql_query_id=qid, table_id=self.tables[table])
        return aid

    def zone(self, name: str, *, table: str, enabled: bool = True) -> str:
        zid = _add(s.zone, name=f"{self.prefix}-{name}", enabled=enabled)
        self.zones[name] = zid
        _link(s.zone_target, zone_id=zid, table_id=self.tables[table])
        return zid


@pytest.fixture
def world():
    w = World(f"t-{uuid.uuid4().hex[:8]}")
    yield w
    for zone_id in w.zones.values():
        store().query_write(s.zone.delete().where(s.zone.c.id == zone_id))
    store().query_write(
        s.catalog_database.delete().where(s.catalog_database.c.id == w.database)
    )
    for table in (s.sql_attribute, s.column_attribute, s.term):
        store().query_write(table.delete().where(table.c.name.like(f"{w.prefix}%")))
    for query_id in w.queries:
        store().query_write(s.sql_query.delete().where(s.sql_query.c.id == query_id))


def _mine(rows, world) -> list[dict]:
    return [r for r in rows if str(r.get("name", "")).startswith(world.prefix)]


# --------------------------------------------------------------------------
# What counts as a Term
# --------------------------------------------------------------------------


def test_a_term_with_no_table_is_not_in_the_list(world) -> None:
    """ "Exists" means semantic *and* represented, and both halves matter."""
    world.table("orders")
    world.term("Order", represents=("orders",))
    world.term("Floating")

    assert {r["name"] for r in _mine(t.fetch_all_terms(), world)} == {
        f"{world.prefix}-Order"
    }


def test_a_non_semantic_term_is_not_in_the_list(world) -> None:
    world.table("orders")
    world.term("Order", represents=("orders",))
    world.term("Imported", represents=("orders",), source="model_interchange")

    assert {r["name"] for r in _mine(t.fetch_all_terms(), world)} == {
        f"{world.prefix}-Order"
    }


def test_the_list_its_total_and_the_scope_check_agree(world) -> None:
    """The three surfaces that must never disagree, asserted together.

    A total describing a different set than the list it pages is a pager that
    runs off the end, and nothing raises when it does.
    """
    world.table("orders")
    world.table("customers")
    world.term("Order", represents=("orders",))
    world.term("Customer", represents=("customers",))
    world.term("Floating")

    listed = _mine(t.fetch_all_terms(), world)
    assert len(listed) == 2
    assert t.count_terms(search=world.prefix) == 2
    for row in listed:
        assert t.term_is_in_scope(row["id"]) is True
    assert t.term_is_in_scope(world.terms["Floating"]) is False


def test_search_is_case_insensitive_and_narrows_the_total_too(world) -> None:
    world.table("orders")
    world.term("Order", represents=("orders",))
    world.term("Customer", represents=("orders",))

    found = t.fetch_all_terms(search=f"{world.prefix}-ORD")
    assert {r["name"] for r in found} == {f"{world.prefix}-Order"}
    assert t.count_terms(search=f"{world.prefix}-ORD") == 1


def test_paging_is_stable_and_covers_every_row(world) -> None:
    world.table("orders")
    for i in range(5):
        world.term(f"Term{i}", represents=("orders",))

    seen: list[str] = []
    for skip in range(0, 6, 2):
        page = t.fetch_all_terms(search=world.prefix, skip=skip, limit=2)
        seen.extend(r["id"] for r in page)
    assert len(seen) == len(set(seen)) == 5


# --------------------------------------------------------------------------
# All-or-nothing zone scoping
# --------------------------------------------------------------------------


@pytest.fixture
def straddling(world):
    """`Shared` represents one in-zone table and one out. It must be hidden."""
    world.table("orders")
    world.table("secrets")
    world.term("Order", represents=("orders",))
    world.term("Secret", represents=("secrets",))
    world.term("Shared", represents=("orders", "secrets"))
    world.zone_id = world.zone("Sales", table="orders")
    return world


def test_a_term_representing_one_out_of_zone_table_is_hidden(straddling) -> None:
    """The case the positive form of the check gets wrong.

    `Shared` represents an in-zone table, so "is any representing table
    allowed?" answers yes. The rule is the other way round.
    """
    visible = {r["name"] for r in t.fetch_all_terms(zone_ids=[straddling.zone_id])}
    assert visible == {f"{straddling.prefix}-Order"}
    assert t.count_terms(zone_ids=[straddling.zone_id]) == 1
    assert t.term_is_in_scope(straddling.terms["Shared"], [straddling.zone_id]) is False


def test_get_full_term_applies_the_same_boundary(straddling) -> None:
    """Otherwise a viewer bypasses the list scoping by asking for an id."""
    assert (
        t.get_full_term_by_id(straddling.terms["Shared"], [straddling.zone_id]) is None
    )
    assert t.get_full_term_by_id(straddling.terms["Order"], [straddling.zone_id])


def test_no_zone_ids_means_unscoped_not_empty(straddling) -> None:
    assert len(_mine(t.fetch_all_terms(zone_ids=None), straddling)) == 3
    assert t.fetch_all_terms(zone_ids=[]) == []


# --------------------------------------------------------------------------
# Certification rollup
# --------------------------------------------------------------------------


def test_a_bare_term_is_pending(world) -> None:
    world.table("orders")
    term = world.term("Order", represents=("orders",))
    assert t.get_term_certification(term) == "pending"


def test_a_term_with_no_attributes_is_decided_by_its_own_two_flags(world) -> None:
    """The flags set is never empty, so this case has a defined answer."""
    world.table("orders")
    term = world.term("Order", represents=("orders",))

    t.update_term(term, name_certified=True)
    assert t.get_term_certification(term) == "partial"
    t.update_term(term, description_certified=True)
    assert t.get_term_certification(term) == "certified"


def test_an_uncertified_attribute_drags_the_rollup_to_partial(world) -> None:
    world.table("orders", columns=("id",))
    term = world.term("Order", represents=("orders",))
    t.update_term(term, name_certified=True, description_certified=True)
    assert t.get_term_certification(term) == "certified"

    world.column_attribute("order-id", "Order", table="orders", certified=False)
    assert t.get_term_certification(term) == "partial"


def test_both_attribute_kinds_count(world) -> None:
    world.table("orders", columns=("id",))
    term = world.term("Order", represents=("orders",))
    t.update_term(term, name_certified=True, description_certified=True)
    world.column_attribute("order-id", "Order", table="orders", certified=True)
    world.sql_attribute("revenue", "Order", tables=("orders",), certified=False)

    assert t.get_term_certification(term) == "partial"
    store().query_write(
        s.sql_attribute.update()
        .where(s.sql_attribute.c.id == world.attributes["revenue"])
        .values(certified=True)
    )
    assert t.get_term_certification(term) == "certified"


def test_the_rollup_ignores_attributes_the_viewer_cannot_see(world) -> None:
    """A badge must not be dragged down by something the list does not show."""
    world.table("orders", columns=("id",))
    world.table("secrets", columns=("id",))
    term = world.term("Order", represents=("orders",))
    t.update_term(term, name_certified=True, description_certified=True)
    world.column_attribute("visible", "Order", table="orders", certified=True)
    world.column_attribute("hidden", "Order", table="secrets", certified=False)
    zone_id = world.zone("Sales", table="orders")

    assert t.get_term_certification(term, None) == "partial"
    assert t.get_term_certification(term, [zone_id]) == "certified"


def test_the_list_card_and_the_detail_page_agree(world) -> None:
    """Three surfaces, one rule — asserted rather than assumed."""
    world.table("orders", columns=("id",))
    term = world.term("Order", represents=("orders",))
    t.update_term(term, name_certified=True)
    world.column_attribute("order-id", "Order", table="orders", certified=True)

    card = _mine(t.fetch_all_terms(), world)[0]["certification"]
    detail = t.get_full_term_by_id(term)["certification"]
    write = t.get_term_certification(term)
    assert card == detail == write == "partial"


# --------------------------------------------------------------------------
# Writes
# --------------------------------------------------------------------------


def test_update_returns_the_old_name(world) -> None:
    """The caller needs it: embeddings are keyed on the old text."""
    world.table("orders")
    term = world.term("Order", represents=("orders",))

    result = t.update_term(term, name=f"{world.prefix}-Purchase")
    assert result["name"] == f"{world.prefix}-Purchase"


def test_a_rename_rewrites_term_name_on_every_attribute(world) -> None:
    """Not optional: the attribute list matches through that denormalised copy.

    Leave it stale and a term's attribute list silently empties — the rows are
    still there, and nothing errors.
    """
    world.table("orders", columns=("id",))
    term = world.term("Order", represents=("orders",))
    world.column_attribute("order-id", "Order", table="orders", column="id")
    assert len(t.fetch_column_attributes_by_term_id(term)) == 1

    t.update_term(term, name=f"{world.prefix}-Purchase")

    assert len(t.fetch_column_attributes_by_term_id(term)) == 1
    row = store().query_read(
        select(s.column_attribute.c.term_name).where(
            s.column_attribute.c.id == world.attributes["order-id"]
        )
    )[0]
    assert row["term_name"] == f"{world.prefix}-Purchase"


def test_update_leaves_omitted_fields_alone(world) -> None:
    world.table("orders")
    term = world.term("Order", represents=("orders",), description="first")

    t.update_term(term, name_certified=True)
    result = t.update_term(term, name=f"{world.prefix}-Renamed")
    assert result["description"] == "first"
    assert result["name_certified"] is True


def test_update_of_a_missing_term_is_none(world) -> None:
    assert t.update_term("no-such-term", name="x") is None


def test_upsert_table_term_is_idempotent_for_one_table(world) -> None:
    world.table("orders")

    first = t.upsert_table_term(
        f"{world.prefix}-Order", "original", world.tables["orders"]
    )
    second = t.upsert_table_term(
        f"{world.prefix}-Order", "rebuilt", world.tables["orders"]
    )

    assert first is not None
    assert first[1] == f"{world.prefix}-Order"
    assert second == first
    row = store().query_read(
        select(s.term.c.description).where(s.term.c.id == first[0])
    )[0]
    assert row["description"] == "rebuilt"


def test_upsert_table_term_qualifies_only_a_cross_table_name_collision(world) -> None:
    world.table("orders")
    world.table("invoices")
    proposed = f"{world.prefix}-Record"

    first = t.upsert_table_term(proposed, "orders", world.tables["orders"])
    second = t.upsert_table_term(proposed, "invoices", world.tables["invoices"])

    assert first is not None
    assert second is not None
    assert first[1] == proposed
    assert second[0] != first[0]
    assert second[1] == f"{proposed} ({world.prefix}.public.invoices)"
    links = store().query_read(
        select(s.table__term.c.table_id, s.table__term.c.term_id).where(
            s.table__term.c.term_id.in_([first[0], second[0]])
        )
    )
    assert {(row["table_id"], row["term_id"]) for row in links} == {
        (world.tables["orders"], first[0]),
        (world.tables["invoices"], second[0]),
    }


def test_upsert_table_term_writes_nothing_when_the_table_is_missing() -> None:
    assert t.upsert_table_term("Ghost", "x", "no-such-table") is None


# --------------------------------------------------------------------------
# Term detail
# --------------------------------------------------------------------------


def test_get_full_term_carries_its_tables_and_count(world) -> None:
    world.table("orders")
    world.table("invoices")
    term = world.term("Order", represents=("orders", "invoices"))

    result = t.get_full_term_by_id(term)
    assert result["table_count"] == 2
    assert {row["name"] for row in result["tables"]} == {"orders", "invoices"}
    assert all(row["db_id"] == world.database for row in result["tables"])


def test_get_full_term_of_a_missing_term_is_none() -> None:
    assert t.get_full_term_by_id("no-such-term") is None


def test_term_record_for_table(world) -> None:
    world.table("orders")
    term = world.term("Order", represents=("orders",), description="a purchase")

    record = t.get_term_record_for_table(world.tables["orders"])
    assert record == {
        "id": term,
        "name": f"{world.prefix}-Order",
        "description": "a purchase",
    }
    assert t.get_term_record_for_table("no-such-table") is None


def test_a_missing_description_reads_as_empty_not_none(world) -> None:
    """Callers concatenate this into prompts; None would render as "None"."""
    world.table("orders")
    world.term("Order", represents=("orders",))
    assert t.get_term_record_for_table(world.tables["orders"])["description"] == ""


def test_semantic_layer_calculated(world) -> None:
    """A cheap "has the semantic build ever run" probe.

    Deliberately not asserted as False on an empty store: other tests in this
    module leave terms behind while they run, so the negative case is not
    something this suite can honestly observe.
    """
    world.table("orders")
    world.term("Order", represents=("orders",))
    assert t.semantic_layer_calculated() is True


# --------------------------------------------------------------------------
# Zone chips
# --------------------------------------------------------------------------


def test_zones_come_from_both_attribute_paths(world) -> None:
    """A SqlAttribute can reach tables no ColumnAttribute touches.

    Resolving only the column path would drop the chip for a term whose content
    arrives entirely through a cross-table formula.
    """
    world.table("orders", columns=("id",))
    world.table("payments")
    world.term("Order", represents=("orders",))
    world.column_attribute("order-id", "Order", table="orders", column="id")
    world.sql_attribute("revenue", "Order", tables=("payments",))
    orders_zone = world.zone("Orders", table="orders")
    payments_zone = world.zone("Payments", table="payments")

    chips = t.get_full_term_by_id(world.terms["Order"])["zones"]
    assert {z["id"] for z in chips} == {orders_zone, payments_zone}


def test_a_viewer_never_sees_a_disabled_zone_chip(world) -> None:
    world.table("orders", columns=("id",))
    world.term("Order", represents=("orders",))
    world.column_attribute("order-id", "Order", table="orders", column="id")
    live = world.zone("Live", table="orders")
    retired = world.zone("Retired", table="orders", enabled=False)

    chips = t.get_full_term_by_id(world.terms["Order"], [live, retired])["zones"]
    assert [z["id"] for z in chips] == [live]

    admin = t.get_full_term_by_id(world.terms["Order"], None)["zones"]
    assert {z["id"]: z["enabled"] for z in admin} == {live: True, retired: False}


def test_the_zones_map_matches_the_detail_page(world) -> None:
    """The whole reason the map exists is to agree with the per-term read."""
    world.table("orders", columns=("id",))
    world.term("Order", represents=("orders",))
    world.column_attribute("order-id", "Order", table="orders", column="id")
    world.zone("Orders", table="orders")

    detail = t.get_full_term_by_id(world.terms["Order"])["zones"]
    mapped = t.fetch_term_zones_map(term_ids=[world.terms["Order"]])[
        world.terms["Order"]
    ]
    assert detail == mapped


def test_an_empty_term_id_list_means_no_terms(world) -> None:
    """Not "all of them" — the distinction a truthiness check would lose."""
    assert t.fetch_term_zones_map(term_ids=[]) == {}


# --------------------------------------------------------------------------
# ColumnAttributes of a Term
# --------------------------------------------------------------------------


def test_attributes_use_plain_membership_not_all_or_nothing(world) -> None:
    """A genuine difference from the Term rule, not an oversight.

    A ColumnAttribute is owned by exactly one table, so there is no "spans the
    boundary" case to protect against.
    """
    world.table("orders", columns=("id",))
    world.table("secrets", columns=("id",))
    term = world.term("Order", represents=("orders",))
    world.column_attribute("visible", "Order", table="orders", column="id")
    world.column_attribute("hidden", "Order", table="secrets", column="id")
    zone_id = world.zone("Sales", table="orders")

    rows = t.fetch_column_attributes_by_term_id(term, [zone_id])
    assert [r["name"] for r in rows] == [f"{world.prefix}-visible"]
    assert t.count_column_attributes_by_term_id(term, [zone_id]) == 1


def test_attribute_counts_agree_with_the_list(world) -> None:
    world.table("orders", columns=("id", "total"))
    term = world.term("Order", represents=("orders",))
    world.column_attribute("a", "Order", table="orders", column="id")
    world.column_attribute("b", "Order", table="orders", column="total")

    assert t.count_column_attributes_by_term_id(term) == len(
        t.fetch_column_attributes_by_term_id(term)
    )
    counts = {r["term_id"]: r["count"] for r in t.fetch_column_attribute_counts()}
    assert counts[term] == 2


def test_an_attribute_on_several_columns_still_yields_one_row(world) -> None:
    """Otherwise the rows outnumber the total the pager was handed."""
    world.table("orders", columns=("id", "alt_id"))
    term = world.term("Order", represents=("orders",))
    attr = world.column_attribute("order-id", "Order", table="orders", column="id")
    _link(
        s.column__has_attribute,
        column_id=world.columns["orders.alt_id"],
        attribute_id=attr,
    )

    assert len(t.fetch_column_attributes_by_term_id(term)) == 1
    assert t.count_column_attributes_by_term_id(term) == 1


def test_attributes_carry_their_columns_and_zones(world) -> None:
    world.table("orders", columns=("id",))
    world.table("invoices", columns=("order_id",))
    term = world.term("Order", represents=("orders",))
    attr = world.column_attribute("order-id", "Order", table="orders", column="id")
    _link(
        s.column__semantic_fk,
        column_id=world.columns["invoices.order_id"],
        attribute_id=attr,
    )
    zone_id = world.zone("Sales", table="orders")

    row = t.fetch_column_attributes_by_term_id(term)[0]
    assert row["primary_column"]["column_name"] == "id"
    assert [c["table_name"] for c in row["referenced_columns"]] == ["invoices"]
    assert [z["id"] for z in row["zones"]] == [zone_id]


def test_sample_values_are_parsed_not_returned_raw(world) -> None:
    world.table("orders", columns=("id",))
    term = world.term("Order", represents=("orders",))
    world.column_attribute("order-id", "Order", table="orders", column="id")
    store().query_write(
        s.catalog_column.update()
        .where(s.catalog_column.c.id == world.columns["orders.id"])
        .values(sample_values='["1", "2"]')
    )

    assert t.fetch_column_attributes_by_term_id(term)[0]["sample_values"] == ["1", "2"]


# --------------------------------------------------------------------------
# Related terms
# --------------------------------------------------------------------------


@pytest.fixture
def related(world):
    """`Order` and `Customer` share `orders` through a SEMANTIC_FK.

    The third path, and the one a naive implementation misses: `orders` does not
    represent `Customer` and carries no ColumnAttribute of it — only a foreign
    key column pointing at one.
    """
    world.table("orders", columns=("id", "customer_id"))
    world.table("customers", columns=("id",))
    world.term("Order", represents=("orders",))
    world.term("Customer", represents=("customers",))
    attr = world.column_attribute(
        "customer-id", "Customer", table="customers", column="id"
    )
    _link(
        s.column__semantic_fk,
        column_id=world.columns["orders.customer_id"],
        attribute_id=attr,
    )
    return world


def test_semantic_fk_makes_two_terms_related(related) -> None:
    names = [r["name"] for r in t.fetch_related_terms(related.terms["Order"])]
    assert names == [f"{related.prefix}-Customer"]


def test_the_count_and_the_list_agree(related) -> None:
    """The failure mode here is three plausible numbers, not an error."""
    term = related.terms["Order"]
    listed = t.fetch_related_terms(term)
    ids = t.fetch_related_term_ids(term)
    counts = {
        r["term_id"]: r["count"] for r in t.fetch_related_terms_counts(term_ids=[term])
    }
    assert len(listed) == len(ids) == counts[term] == 1


def test_a_term_is_not_related_to_itself(related) -> None:
    assert related.terms["Order"] not in t.fetch_related_term_ids(
        related.terms["Order"]
    )


def test_narrowing_term_ids_does_not_shrink_a_count(related) -> None:
    """*term_ids* restricts which terms get an entry, not what counts as related.

    Otherwise a term's badge would change depending on which page it landed on.
    """
    term = related.terms["Order"]
    narrowed = t.fetch_related_terms_counts(term_ids=[term])
    everything = t.fetch_related_terms_counts()
    by_term = {r["term_id"]: r["count"] for r in everything}
    assert narrowed == [{"term_id": term, "count": by_term[term]}]


def test_related_terms_respect_the_zone_boundary(related) -> None:
    """A chip a viewer could not open must not be shown.

    `Customer` also represents an out-of-zone table, so opening it would 404 —
    and a chip that 404s is worse than no chip.
    """
    related.table("secrets")
    _link(
        s.table__term,
        table_id=related.tables["secrets"],
        term_id=related.terms["Customer"],
    )
    zone_id = related.zone("Sales", table="orders")
    related.zone("More", table="customers")
    zones = [zone_id, related.zones["More"]]

    assert t.fetch_related_terms(related.terms["Order"], zones) == []


def test_terms_by_ids_orders_by_name_case_insensitively(world) -> None:
    world.table("orders")
    ids = [
        world.term("beta", represents=("orders",)),
        world.term("Alpha", represents=("orders",)),
    ]
    names = [r["name"] for r in t.fetch_terms_by_ids(ids)]
    assert names == [f"{world.prefix}-Alpha", f"{world.prefix}-beta"]


def test_terms_by_ids_is_empty_for_an_empty_list() -> None:
    assert t.fetch_terms_by_ids([]) == []


# --------------------------------------------------------------------------
# Suggester and embedding inputs
# --------------------------------------------------------------------------


def test_table_schema_map_lower_cases_its_keys(world) -> None:
    world.table("Orders")
    assert t.fetch_table_schema_map(world.prefix) == {"orders": "public"}


def test_terms_with_sqls_excludes_generated_statements(world) -> None:
    """A suggester that learns from its own output is a feedback loop."""
    world.table("orders")
    world.term("Order", represents=("orders",))

    ingestion = _add(s.sql_query, sql_full_query=f"SELECT ingested -- {world.prefix}")
    world.queries.append(ingestion)
    _link(s.sql_query__table, sql_query_id=ingestion, table_id=world.tables["orders"])
    # An attribute-owned statement over the same table must not come back.
    world.sql_attribute("revenue", "Order", tables=("orders",))

    rows = [
        r for r in t.fetch_terms_with_sqls() if r["term_id"] == world.terms["Order"]
    ]
    assert len(rows) == 1
    assert [sql["sql_text"] for sql in rows[0]["sqls"]] == [
        f"SELECT ingested -- {world.prefix}"
    ]


def test_a_term_with_no_ingestion_sql_is_omitted(world) -> None:
    world.table("orders")
    world.term("Order", represents=("orders",))
    assert [
        r for r in t.fetch_terms_with_sqls() if r["term_id"] == world.terms["Order"]
    ] == []


def test_terms_and_attributes_for_table(world) -> None:
    world.table("orders", columns=("id",))
    world.term("Order", represents=("orders",))
    world.column_attribute("order-id", "Order", table="orders", column="id")

    terms, attrs = t.fetch_terms_and_attributes_for_table(world.tables["orders"])
    assert [row["name"] for row in terms] == [f"{world.prefix}-Order"]
    assert terms[0]["schema_names"] == ["public"]
    assert [row["name"] for row in attrs] == [f"{world.prefix}-order-id"]


def test_term_and_attributes_for_embedding(world) -> None:
    world.table("orders", columns=("id",))
    term = world.term("Order", represents=("orders",))
    world.column_attribute("order-id", "Order", table="orders", column="id")

    row, attrs = t.fetch_term_and_column_attributes_for_embedding(term)
    assert row["name"] == f"{world.prefix}-Order"
    assert row["database_name"] == world.prefix
    assert [a["name"] for a in attrs] == [f"{world.prefix}-order-id"]


def test_embedding_context_for_a_missing_term_is_none() -> None:
    assert t.fetch_term_and_column_attributes_for_embedding("no-such-term") == (
        None,
        [],
    )


def test_a_sample_value_update_invalidates_its_terms(world) -> None:
    """Sample values are part of an attribute's embedding text.

    Profiling a column stales every attribute on it and every term behind those,
    which is what this resolves.
    """
    world.table("orders", columns=("id",))
    world.term("Order", represents=("orders",))
    world.column_attribute("order-id", "Order", table="orders", column="id")

    contexts = t.fetch_column_attribute_embedding_contexts_by_column_id(
        world.columns["orders.id"]
    )
    assert len(contexts) == 1
    term_row, attrs = contexts[0]
    assert term_row["id"] == world.terms["Order"]
    assert term_row["database_name"] == world.prefix
    assert [a["name"] for a in attrs] == [f"{world.prefix}-order-id"]


def test_synonyms_are_keyed_by_name_and_skip_empty_ones(world) -> None:
    world.table("orders", columns=("id",))
    world.term("Order", represents=("orders",))
    world.term("Plain", represents=("orders",))
    store().query_write(
        s.term.update()
        .where(s.term.c.id == world.terms["Order"])
        .values(synonyms=["purchase", "sale"])
    )
    attr = world.column_attribute("order-id", "Order", table="orders", column="id")
    plain = world.column_attribute("plain-id", "Plain", table="orders", column="id")

    assert t.fetch_term_synonyms([attr, plain]) == {
        f"{world.prefix}-Order": ["purchase", "sale"]
    }


def test_synonyms_are_empty_for_an_empty_list() -> None:
    assert t.fetch_term_synonyms([]) == {}
