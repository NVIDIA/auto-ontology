# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Global search, against a real database.

The queries are the whole of this module, so mocking the connection would test
nothing: what can break here is a substring that stops matching, a Term that
should not have been visible, or a breadcrumb pointing at the wrong parent —
none of which a compiled statement shows.

Two properties get the most attention because they are the ones a reasonable
change quietly breaks:

* **Substring, not prefix.** ``ental`` has to find ``rental_count``. This is the
  reason the indexes are ``pg_trgm`` rather than ``tsvector``, and a port to
  full-text search would pass every other test in this file.
* **Name and description are matched whole.** Two tokens split across the two
  fields is not a hit, or every column named after a word in its table's
  description starts appearing.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator

import pytest

pytest.importorskip("sqlalchemy")

from auto_ontology.catalog.constants import Labels, TableTypes  # noqa: E402
from auto_ontology.dal import schema as s  # noqa: E402
from auto_ontology.dal import search  # noqa: E402
from auto_ontology.dal.session import store, write_transaction  # noqa: E402
from auto_ontology.semantic.constants import (  # noqa: E402
    LABEL_COLUMN_ATTRIBUTE,
    LABEL_TERM,
    SEMANTIC_SOURCE,
)

ALL_TYPES = set(search.SEARCH_OBJECT_TYPES)


@pytest.fixture(scope="module", autouse=True)
def require_schema():
    if not os.environ.get("POSTGRES_USER"):
        pytest.skip("POSTGRES_* not set")
    try:
        store().query_read("SELECT 1 FROM term LIMIT 1")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"auto_ontology schema unavailable (alembic upgrade head): {exc}")


def _add(table, **values) -> str:
    return store().query_write(table.insert().values(**values).returning(table.c.id))[
        0
    ]["id"]


class World:
    """A catalog and semantic layer whose names exist only for this test run.

    Every name carries the run's unique prefix, so a search for it cannot reach
    the seeded Pagila rows the developer database also holds, and two runs
    against the same database cannot see each other.
    """

    def __init__(self) -> None:
        self.prefix = f"gz{uuid.uuid4().hex[:8]}"
        p = self.prefix

        self.database = _add(s.catalog_database, name=f"{p}_warehouse")
        self.schema = _add(
            s.catalog_schema, database_id=self.database, name=f"{p}_public"
        )
        self.table = _add(
            s.catalog_table,
            schema_id=self.schema,
            name=f"{p}_rental",
            table_type=TableTypes.BASE_TABLE,
        )
        self.view = _add(
            s.catalog_table,
            schema_id=self.schema,
            name=f"{p}_rental_summary",
            table_type=TableTypes.VIEW,
        )
        self.column = _add(
            s.catalog_column,
            table_id=self.table,
            name=f"{p}_total_amount",
            description=f"{p} how much the customer paid",
        )
        self.term = _add(
            s.term,
            name=f"{p}_Revenue",
            source=SEMANTIC_SOURCE,
            synonyms=[f"{p} Business Unit", f"{p} Takings"],
            name_certified=True,
            description_certified=True,
        )
        # Visibility is a represented Term; an unrepresented one must not show.
        store().query_write(
            s.table__term.insert().values(table_id=self.table, term_id=self.term)
        )
        self.orphan_term = _add(
            s.term, name=f"{p}_Orphan", source=SEMANTIC_SOURCE, synonyms=[]
        )
        self.attribute = _add(
            s.column_attribute,
            name=f"{p}_net_total",
            source_column=f"{p}_total_amount",
            term_name=f"{p}_Revenue",
            table_id=self.table,
            certified=True,
        )
        store().query_write(
            s.column_attribute__term.insert().values(
                attribute_id=self.attribute, term_id=self.term
            )
        )
        # An attribute has no page of its own, so these two are as unreachable
        # as an unrepresented Term: one has no Term at all, the other's Term is
        # itself hidden.
        self.unlinked_attribute = _add(
            s.column_attribute,
            name=f"{p}_net_unlinked",
            source_column=f"{p}_total_amount",
            term_name=f"{p}_Nothing",
            table_id=self.table,
        )
        self.hidden_term_attribute = _add(
            s.column_attribute,
            name=f"{p}_net_hidden",
            source_column=f"{p}_total_amount",
            term_name=f"{p}_Orphan",
            table_id=self.table,
        )
        store().query_write(
            s.column_attribute__term.insert().values(
                attribute_id=self.hidden_term_attribute, term_id=self.orphan_term
            )
        )
        # Two tags over two kinds, and deliberately not over everything: the
        # view, the column, the attribute and the containers above them stay
        # unlabelled, which is what the ``(blanks)`` half of the filter has to
        # find.
        self.tag = _add(s.tag, name=f"{p}_pii")
        self.other_tag = _add(s.tag, name=f"{p}_curated")
        store().query_write(
            s.tag_target.insert().values(tag_id=self.tag, table_id=self.table)
        )
        store().query_write(
            s.tag_target.insert().values(tag_id=self.tag, term_id=self.term)
        )
        store().query_write(
            s.tag_target.insert().values(tag_id=self.other_tag, column_id=self.column)
        )


@pytest.fixture(scope="module")
def world() -> Iterator[World]:
    """Build the fixture once, and take it back out again.

    Deleting the database cascades to its schema, table, columns and the
    ``table__term`` link; the semantic rows hang off nothing and go by hand.
    The tags go the same way, and take their ``tag_target`` rows with them --
    every one of those columns is ``ondelete="CASCADE"`` from both ends.
    """
    with write_transaction():
        built = World()
    yield built
    with write_transaction():
        store().query_write(
            s.tag.delete().where(s.tag.c.id.in_([built.tag, built.other_tag]))
        )
        store().query_write(
            s.catalog_database.delete().where(s.catalog_database.c.id == built.database)
        )
        store().query_write(
            s.column_attribute.delete().where(
                s.column_attribute.c.id.in_(
                    [
                        built.attribute,
                        built.unlinked_attribute,
                        built.hidden_term_attribute,
                    ]
                )
            )
        )
        store().query_write(
            s.term.delete().where(s.term.c.id.in_([built.term, built.orphan_term]))
        )


def _find(world: World, term: str, types: set[str] | None = None, **kwargs) -> list:
    """Search for *term* the way the service does.

    Both token lists come from the same input, because the two are different
    readings of one query rather than separate arguments a caller chooses
    between — see ``auto_ontology.server.search.service._prepare_search``.
    """
    return search.fetch_global_search(
        search.search_tokens(term),
        types or ALL_TYPES,
        synonym_tokens=search.synonym_word_tokens(term),
        **kwargs,
    )


def _names(rows: list) -> set[str]:
    return {row["name"] for row in rows}


# --------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------


def test_matches_the_middle_of_a_word(world: World) -> None:
    """The property `tsvector` cannot provide, and the reason for `pg_trgm`."""
    hits = _find(world, "otal_amou", include_description=False)
    assert f"{world.prefix}_total_amount" in _names(hits)


def test_match_is_case_insensitive(world: World) -> None:
    assert _names(_find(world, "TOTAL_AMOUNT", include_description=False))


def test_every_token_must_match(world: World) -> None:
    both = _find(world, f"{world.prefix} amount", include_description=False)
    assert f"{world.prefix}_total_amount" in _names(both)

    neither = _find(world, f"{world.prefix} nonesuch", include_description=False)
    assert neither == []


def test_tokens_are_not_split_across_name_and_description(world: World) -> None:
    """``total`` is in the name and ``customer`` in the description.

    Matching the two together would be an ``OR`` per token instead of per field,
    which is how a search starts returning rows that contain neither phrase.
    """
    hits = _find(world, f"{world.prefix}_total customer", include_description=True)
    assert f"{world.prefix}_total_amount" not in _names(hits)


def test_description_is_searched_only_when_asked(world: World) -> None:
    term = f"{world.prefix} how much"
    assert _names(_find(world, term, include_description=True))
    assert _find(world, term, include_description=False) == []


def test_like_wildcards_in_the_query_are_literal(world: World) -> None:
    """``_`` is legal in an identifier and a wildcard to Postgres.

    Unescaped, this query would also match ``total_amount`` — the search would
    quietly answer a question the user did not ask.
    """
    assert _find(world, "total_am_unt", include_description=False) == []


def test_short_tokens_still_match_here(world: World) -> None:
    """The two-character floor is the service's rule, not the DAL's."""
    assert search.fetch_global_search(["am"], ALL_TYPES, include_description=False)


def test_no_tokens_matches_nothing() -> None:
    """An empty token list ANDs to true, which would return the whole catalog."""
    assert search.fetch_global_search([], ALL_TYPES, include_description=False) == []
    assert search.count_global_search([], ALL_TYPES, include_description=False) == {}


# --------------------------------------------------------------------------
# Object types
# --------------------------------------------------------------------------


def test_object_types_narrow_the_search(world: World) -> None:
    hits = _find(world, world.prefix, {Labels.COLUMN}, include_description=False)
    assert {row["label"] for row in hits} == {Labels.COLUMN}


def test_views_are_a_type_but_keep_the_table_label(world: World) -> None:
    """``View`` filters and counts; ``Table`` is still what the row says it is."""
    views = _find(
        world, world.prefix, {search.SEARCH_TYPE_VIEW}, include_description=False
    )
    assert _names(views) == {f"{world.prefix}_rental_summary"}
    assert {row["label"] for row in views} == {Labels.TABLE}
    assert search.search_object_type(Labels.TABLE, TableTypes.VIEW) == "View"


def test_tables_exclude_views(world: World) -> None:
    tables = _find(world, world.prefix, {Labels.TABLE}, include_description=False)
    assert _names(tables) == {f"{world.prefix}_rental"}


def test_tables_and_views_together_return_both(world: World) -> None:
    both = _find(
        world,
        world.prefix,
        {Labels.TABLE, search.SEARCH_TYPE_VIEW},
        include_description=False,
    )
    assert _names(both) == {
        f"{world.prefix}_rental",
        f"{world.prefix}_rental_summary",
    }


# --------------------------------------------------------------------------
# Tags
# --------------------------------------------------------------------------
#
# Two halves that combine rather than compete: the tag ids, satisfied by any
# one of them, and ``include_untagged`` for the objects carrying none. The
# service takes them apart from the one list the API carries — see
# ``auto_ontology.server.search.service.resolve_tag_filter``.


def test_a_tag_filter_keeps_only_what_carries_the_tag(world: World) -> None:
    hits = _find(world, world.prefix, include_description=False, tag_ids=[world.tag])
    assert _names(hits) == {f"{world.prefix}_rental", f"{world.prefix}_Revenue"}


def test_any_one_of_the_tags_is_enough(world: World) -> None:
    """Two tags is a union, not an intersection.

    Nothing in the fixture carries both, so an ``AND`` reading of this would
    return an empty list rather than a shorter one — and the count would have
    to agree with it, which is what makes the distinction visible on the tabs
    rather than only in the list.
    """
    names = _names(
        _find(
            world,
            world.prefix,
            include_description=False,
            tag_ids=[world.tag, world.other_tag],
        )
    )
    assert f"{world.prefix}_rental" in names
    assert f"{world.prefix}_total_amount" in names


def test_a_tag_filter_leaves_out_the_kinds_that_cannot_carry_one(
    world: World,
) -> None:
    """``tag_target`` has no column for a Database or a Schema.

    So neither can ever satisfy a tag, and a search narrowed to one has no
    reason to scan them — which is ``_tag_filter_rules_out`` dropping the
    branch rather than the union filtering its rows away.
    """
    hits = _find(world, world.prefix, include_description=False, tag_ids=[world.tag])
    assert {Labels.DB, Labels.SCHEMA}.isdisjoint({row["label"] for row in hits})


def test_the_untagged_sentinel_returns_what_carries_no_tag(world: World) -> None:
    """Including the kinds that cannot be tagged, which are untagged by nature."""
    names = _names(
        _find(world, world.prefix, include_description=False, include_untagged=True)
    )
    assert f"{world.prefix}_rental_summary" in names
    assert f"{world.prefix}_warehouse" in names
    assert f"{world.prefix}_rental" not in names
    assert f"{world.prefix}_Revenue" not in names


def test_tags_and_the_untagged_sentinel_together_narrow_nothing(
    world: World,
) -> None:
    """ "Tagged one of these, or tagged nothing" over every tag is everything.

    The two halves being unioned is what makes this hold; an ``AND`` of them
    would be the empty set, since nothing is both tagged and untagged.
    """
    both = _find(
        world,
        world.prefix,
        include_description=False,
        tag_ids=[world.tag, world.other_tag],
        include_untagged=True,
    )
    unfiltered = _find(world, world.prefix, include_description=False)
    assert _names(both) == _names(unfiltered)


def test_no_tag_filter_asks_nothing_about_tags(world: World) -> None:
    """The twin of the tests above: absent, the filter must not narrow at all."""
    names = _names(_find(world, world.prefix, include_description=False))
    assert f"{world.prefix}_rental" in names
    assert f"{world.prefix}_rental_summary" in names


def test_an_alias_only_term_respects_the_tag_filter(world: World) -> None:
    """The synonym branch is a second route to a Term, not an exemption.

    It is a separate statement so the cap cannot evict it (see
    ``_synonym_term_select``), and that is the reason it would be easy to leave
    the narrowing off — a tag-filtered search would then return untagged Terms
    its own count does not include.
    """
    alias = search.synonym_word_tokens(f"{world.prefix} Takings")
    tagged = search.fetch_global_search(
        search.search_tokens("zzzznomatch"),
        {LABEL_TERM},
        include_description=False,
        synonym_tokens=alias,
        tag_ids=[world.tag],
    )
    assert f"{world.prefix}_Revenue" in _names(tagged)

    untagged = search.fetch_global_search(
        search.search_tokens("zzzznomatch"),
        {LABEL_TERM},
        include_description=False,
        synonym_tokens=alias,
        include_untagged=True,
    )
    assert f"{world.prefix}_Revenue" not in _names(untagged)


def test_counts_respect_the_tag_filter(world: World) -> None:
    counts = search.count_global_search(
        search.search_tokens(world.prefix),
        ALL_TYPES,
        include_description=False,
        tag_ids=[world.tag],
    )
    assert set(counts) == {Labels.TABLE, LABEL_TERM}
    assert counts[Labels.TABLE] == 1


# --------------------------------------------------------------------------
# Tag as an object of its own
# --------------------------------------------------------------------------
#
# A tag is the one searchable kind with no ``description`` column, so it is
# also the one that would break the union by selecting a column that is not
# there -- see ``_LABELS_WITHOUT_DESCRIPTION``.


def test_a_tag_is_a_hit_in_its_own_right(world: World) -> None:
    hits = _find(
        world, f"{world.prefix}_curated", {search.LABEL_TAG}, include_description=False
    )
    assert [(row["name"], row["label"]) for row in hits] == [
        (f"{world.prefix}_curated", search.LABEL_TAG)
    ]


def test_searching_descriptions_does_not_break_the_tag_branch(world: World) -> None:
    """The column does not exist, so the flag has to be dropped for this kind.

    Left in, the branch either selects a missing column or matches against one
    -- and either way the failure is the whole union, not just the tag: every
    other kind's hits disappear with it.
    """
    hits = _find(world, world.prefix, {search.LABEL_TAG}, include_description=True)
    assert _names(hits) == {f"{world.prefix}_pii", f"{world.prefix}_curated"}
    assert all(row["description"] is None for row in hits)


def test_tags_are_counted_and_tabbed_like_any_other_kind(world: World) -> None:
    counts = search.count_global_search(
        search.search_tokens(world.prefix), ALL_TYPES, include_description=False
    )
    assert counts[search.LABEL_TAG] == 2


def test_the_object_filter_can_leave_tags_out(world: World) -> None:
    """Which is what the Objects picker does, and what every other tab does."""
    names = _names(
        _find(
            world,
            world.prefix,
            ALL_TYPES - {search.LABEL_TAG},
            include_description=False,
        )
    )
    assert f"{world.prefix}_pii" not in names
    assert f"{world.prefix}_rental" in names


# --------------------------------------------------------------------------
# The data filter
# --------------------------------------------------------------------------
#
# Narrowing to what lives under chosen databases and schemas. It gets a world
# of its own rather than extra rows in ``World``: the fixture above has one
# database, and giving it a second would change what every test searching the
# bare prefix finds -- which is most of this file.


class DataWorld:
    """Two databases under one prefix, so one search reaches both.

    The shape is the smallest that can tell the four mistakes apart: a filter
    that ignores the database, one that ignores the schema, one that cannot
    place a Column, and one that forgets a View is a table.
    """

    def __init__(self) -> None:
        self.prefix = f"dz{uuid.uuid4().hex[:8]}"
        p = self.prefix

        self.left = _add(s.catalog_database, name=f"{p}_left")
        self.sales = _add(s.catalog_schema, database_id=self.left, name=f"{p}_sales")
        self.finance = _add(
            s.catalog_schema, database_id=self.left, name=f"{p}_finance"
        )
        self.orders = _add(
            s.catalog_table,
            schema_id=self.sales,
            name=f"{p}_orders",
            table_type=TableTypes.BASE_TABLE,
        )
        self.ledger = _add(
            s.catalog_table,
            schema_id=self.finance,
            name=f"{p}_ledger",
            table_type=TableTypes.BASE_TABLE,
        )
        self.amount = _add(s.catalog_column, table_id=self.orders, name=f"{p}_amount")

        self.right = _add(s.catalog_database, name=f"{p}_right")
        self.archive = _add(
            s.catalog_schema, database_id=self.right, name=f"{p}_archive"
        )
        self.old_orders = _add(
            s.catalog_table,
            schema_id=self.archive,
            name=f"{p}_orders_old",
            table_type=TableTypes.BASE_TABLE,
        )
        self.old_view = _add(
            s.catalog_table,
            schema_id=self.archive,
            name=f"{p}_orders_view",
            table_type=TableTypes.VIEW,
        )

        # A Term matching the same prefix, and reachable by an alias as well as
        # by its name -- the two routes the filter has to close separately.
        self.term = _add(
            s.term,
            name=f"{p}_Orders",
            source=SEMANTIC_SOURCE,
            synonyms=[f"{p} Purchases"],
        )
        store().query_write(
            s.table__term.insert().values(table_id=self.orders, term_id=self.term)
        )


@pytest.fixture(scope="module")
def data_world() -> Iterator[DataWorld]:
    with write_transaction():
        built = DataWorld()
    yield built
    with write_transaction():
        store().query_write(
            s.catalog_database.delete().where(
                s.catalog_database.c.id.in_([built.left, built.right])
            )
        )
        store().query_write(s.term.delete().where(s.term.c.id == built.term))


def _under(data: DataWorld, *ids: str, types: set[str] | None = None) -> set[str]:
    """Names matching the whole prefix, narrowed to *ids*."""
    return _names(
        search.fetch_global_search(
            search.search_tokens(data.prefix),
            types or ALL_TYPES,
            include_description=False,
            synonym_tokens=search.synonym_word_tokens(data.prefix),
            data_ids=list(ids),
        )
    )


def test_a_database_narrows_to_everything_beneath_it(data_world: DataWorld) -> None:
    """Both schemas of the chosen database, and neither of the other's."""
    p = data_world.prefix
    assert _under(data_world, data_world.left) == {
        f"{p}_orders",
        f"{p}_ledger",
        f"{p}_amount",
    }


def test_a_schema_narrows_further_than_its_database(data_world: DataWorld) -> None:
    """The finance table drops out, which is the whole point of offering
    schemas in the tree rather than databases alone."""
    p = data_world.prefix
    assert _under(data_world, data_world.sales) == {f"{p}_orders", f"{p}_amount"}


def test_databases_and_schemas_mix_in_one_selection(data_world: DataWorld) -> None:
    """One list holds both, and an id matching either level is enough.

    This is why the two are not separate request fields: the tree they come
    from lets somebody tick a whole database and pick a schema out of another,
    and that is one selection, not two filters.
    """
    p = data_world.prefix
    assert _under(data_world, data_world.finance, data_world.right) == {
        f"{p}_ledger",
        f"{p}_orders_old",
        f"{p}_orders_view",
    }


def test_a_column_is_placed_by_the_table_it_sits_on(data_world: DataWorld) -> None:
    """A Column has no schema of its own, so it is reached through its table.

    Get that hop wrong and the column either escapes every filter or is caught
    by all of them -- neither of which the table beside it would show.
    """
    p = data_world.prefix
    assert _under(data_world, data_world.sales, types={Labels.COLUMN}) == {
        f"{p}_amount"
    }
    assert _under(data_world, data_world.right, types={Labels.COLUMN}) == set()


def test_a_view_is_placed_like_the_table_it_shares_a_row_with(
    data_world: DataWorld,
) -> None:
    p = data_world.prefix
    assert _under(data_world, data_world.archive, types={search.SEARCH_TYPE_VIEW}) == {
        f"{p}_orders_view"
    }


def test_the_filter_leaves_only_what_lives_under_a_schema(
    data_world: DataWorld,
) -> None:
    """Tables, Views and Columns, and nothing else -- including the containers.

    A Term is not part of any database, so it goes. So do the database and the
    schema themselves: returning the very thing somebody just ticked says
    nothing they did not already know, which is the line the reference
    implementation takes too.
    """
    p = data_world.prefix
    unfiltered = _under(data_world)
    assert f"{p}_Orders" in unfiltered
    assert f"{p}_left" in unfiltered
    assert f"{p}_sales" in unfiltered

    narrowed = _under(data_world, data_world.left)
    assert f"{p}_Orders" not in narrowed
    assert f"{p}_left" not in narrowed
    assert f"{p}_sales" not in narrowed


def test_an_alias_does_not_smuggle_a_term_past_the_filter(
    data_world: DataWorld,
) -> None:
    """The alias branch is a second route to a Term, not an exemption.

    It is a separate statement built by a separate function, so the narrowing
    has to be repeated there -- and this is the only test that would notice,
    since the Term's own name matches too and the main branch drops it either
    way.
    """
    hits = search.fetch_global_search(
        search.search_tokens(f"{data_world.prefix} purchases"),
        ALL_TYPES,
        include_description=False,
        synonym_tokens=search.synonym_word_tokens(f"{data_world.prefix} Purchases"),
        data_ids=[data_world.left],
    )
    assert hits == []


def test_an_id_belonging_to_nothing_narrows_to_nothing(
    data_world: DataWorld,
) -> None:
    """Database ids and schema ids are told apart by matching, not by the
    caller, so an id that is neither simply finds no home."""
    assert _under(data_world, data_world.orders) == set()


def test_the_object_filter_still_narrows_inside_the_data_filter(
    data_world: DataWorld,
) -> None:
    """The data filter never widens: asking only for Terms and also for a
    database gives nothing, rather than the tables under it."""
    assert _under(data_world, data_world.left, types={LABEL_TERM}) == set()


def test_counts_agree_with_the_list_about_the_data_filter(
    data_world: DataWorld,
) -> None:
    """The tab badge and the page under it are one match or neither is right."""
    counts = search.count_global_search(
        search.search_tokens(data_world.prefix),
        ALL_TYPES,
        include_description=False,
        synonym_tokens=search.synonym_word_tokens(data_world.prefix),
        data_ids=[data_world.right],
    )
    assert counts == {Labels.TABLE: 1, search.SEARCH_TYPE_VIEW: 1}


def test_the_matching_statements_narrow_by_data_too(data_world: DataWorld) -> None:
    """A rule saved over a filtered search labels what that search showed.

    Its own loop, so the narrowing has to be repeated -- and missed here the
    rule would quietly label every table in the catalog rather than the ones
    on screen.
    """
    selects = search.matching_id_selects(
        search.search_tokens(data_world.prefix),
        ALL_TYPES,
        include_description=False,
        data_ids=[data_world.sales],
    )
    found = {
        label: {row["id"] for row in store().query_read(q)}
        for label, q in selects.items()
    }
    assert found == {
        Labels.TABLE: {data_world.orders},
        Labels.COLUMN: {data_world.amount},
    }


# --------------------------------------------------------------------------
# Terms
# --------------------------------------------------------------------------


def test_a_term_no_table_represents_is_hidden(world: World) -> None:
    """It is a real row with no page behind it, so a hit would open onto nothing."""
    hits = _find(world, world.prefix, {LABEL_TERM}, include_description=False)
    assert f"{world.prefix}_Orphan" not in _names(hits)
    assert f"{world.prefix}_Revenue" in _names(hits)


def test_term_certification_is_the_three_state_string(world: World) -> None:
    hits = _find(
        world, f"{world.prefix}_Revenue", {LABEL_TERM}, include_description=False
    )
    assert [row["certified"] for row in hits] == ["certified"]


def test_an_attribute_no_term_owns_is_hidden(world: World) -> None:
    """The Term rule, for the entity that has no page except its Term's.

    ``hrefForGlobalSearchItem`` has no ``parent_id`` to build a link from and
    falls back to a bare ``/terms``, which cannot focus what was clicked.
    """
    names = _names(
        _find(world, world.prefix, {LABEL_COLUMN_ATTRIBUTE}, include_description=False)
    )
    assert f"{world.prefix}_net_unlinked" not in names
    assert f"{world.prefix}_net_total" in names


def test_an_attribute_whose_term_is_hidden_is_hidden_too(world: World) -> None:
    """A link is not enough — a hidden Term has no page to focus either."""
    names = _names(
        _find(world, world.prefix, {LABEL_COLUMN_ATTRIBUTE}, include_description=False)
    )
    assert f"{world.prefix}_net_hidden" not in names


def test_unreachable_attributes_are_left_out_of_the_counts(world: World) -> None:
    """The badge cannot promise attributes the list refuses to return."""
    counts = search.count_global_search(
        search.search_tokens(world.prefix),
        {LABEL_COLUMN_ATTRIBUTE},
        include_description=False,
    )
    assert counts[LABEL_COLUMN_ATTRIBUTE] == 1


def test_attribute_certification_stays_a_boolean(world: World) -> None:
    hits = _find(
        world,
        f"{world.prefix}_net_total",
        {LABEL_COLUMN_ATTRIBUTE},
        include_description=False,
    )
    assert [row["certified"] for row in hits] == [True]


def test_synonyms_match_whole_words_not_substrings(world: World) -> None:
    found = _find(
        world, f"{world.prefix} Unit", {LABEL_TERM}, include_description=False
    )
    assert f"{world.prefix}_Revenue" in _names(found)

    partial = _find(
        world, f"{world.prefix} Uni", {LABEL_TERM}, include_description=False
    )
    assert f"{world.prefix}_Revenue" not in _names(partial)


def test_all_tokens_must_land_in_one_synonym(world: World) -> None:
    """``Business`` and ``Takings`` are two different aliases of the same Term."""
    hits = search.fetch_global_search(
        search.search_tokens("zzzznomatch"),
        {LABEL_TERM},
        include_description=False,
        synonym_tokens=["business", "takings"],
    )
    assert f"{world.prefix}_Revenue" not in _names(hits)


def test_a_synonym_only_term_is_returned(world: World) -> None:
    """The name matches nothing; the alias is the only reason it is here."""
    hits = search.fetch_global_search(
        search.search_tokens("zzzznomatch"),
        {LABEL_TERM},
        include_description=False,
        synonym_tokens=search.synonym_word_tokens(f"{world.prefix} Takings"),
    )
    assert f"{world.prefix}_Revenue" in _names(hits)


def test_no_synonym_tokens_drops_the_synonym_only_term(world: World) -> None:
    """The twin of the test above, and what the ``synonyms`` filter rests on.

    ``include_synonyms=False`` in the service is not a flag this module knows
    about — it arrives as no synonym tokens at all. So the filter is only real
    if an empty list actually withholds the alias branch, which is what this
    asks: same query as above, same alias in the database, no tokens, no hit.
    """
    hits = search.fetch_global_search(
        search.search_tokens("zzzznomatch"),
        {LABEL_TERM},
        include_description=False,
        synonym_tokens=[],
    )
    assert f"{world.prefix}_Revenue" not in _names(hits)


def test_a_term_matching_name_and_synonym_is_one_hit(world: World) -> None:
    hits = search.fetch_global_search(
        search.search_tokens(world.prefix),
        {LABEL_TERM},
        include_description=False,
        synonym_tokens=search.synonym_word_tokens(world.prefix),
    )
    ids = [row["id"] for row in hits]
    assert ids.count(world.term) == 1


def _aliased_term(world: World, name: str, alias: str) -> str:
    """A visible Term called *name* whose only alias is *alias*, no description."""
    term_id = _add(
        s.term, name=name, source=SEMANTIC_SOURCE, synonyms=[alias], description=None
    )
    store().query_write(
        s.table__term.insert().values(table_id=world.table, term_id=term_id)
    )
    return term_id


def test_the_alias_cap_is_spent_on_terms_the_text_match_missed(world: World) -> None:
    """*synonym_limit* has to buy rescues, not duplicates of the page.

    The alias branch and the text branch both see a Term that matches its name
    *and* its alias. If the branches overlap, ``_list_rank`` puts that Term
    first in the alias branch too — it is a name hit — so a cap of one is spent
    re-fetching a row the page already has, the id pass drops it as a duplicate,
    and the Term that had nothing but an alias is lost.
    """
    needle = f"{world.prefix}kw"
    alias = f"{needle} alias"
    created: list[str] = []
    try:
        with write_transaction():
            created.append(_aliased_term(world, f"{needle}_named_term", alias))
            created.append(
                _aliased_term(world, f"{world.prefix}_zzz_alias_only", alias)
            )
        hits = search.fetch_global_search(
            search.search_tokens(needle),
            {LABEL_TERM},
            include_description=False,
            synonym_tokens=search.synonym_word_tokens(needle),
            synonym_limit=1,
        )
        assert _names(hits) == {
            f"{needle}_named_term",
            f"{world.prefix}_zzz_alias_only",
        }
    finally:
        with write_transaction():
            store().query_write(s.term.delete().where(s.term.c.id.in_(created)))


def test_an_alias_only_term_with_no_description_still_comes_back(
    world: World,
) -> None:
    """The alias branch negates the text match, and ``description`` is nullable.

    An unmatched name over a NULL description makes the text match NULL, not
    false, so a bare ``NOT`` is NULL and the WHERE clause drops the row. Every
    alias-only Term without a description would vanish, and only when the
    caller asked for descriptions — which is the default in the UI's All tab.
    """
    needle = f"{world.prefix}nd"
    created: list[str] = []
    try:
        with write_transaction():
            created.append(
                _aliased_term(world, f"{world.prefix}_zzz_nulldesc", f"{needle} alias")
            )
        hits = search.fetch_global_search(
            search.search_tokens(needle),
            {LABEL_TERM},
            include_description=True,
            synonym_tokens=search.synonym_word_tokens(needle),
        )
        assert f"{world.prefix}_zzz_nulldesc" in _names(hits)
    finally:
        with write_transaction():
            store().query_write(s.term.delete().where(s.term.c.id.in_(created)))


# --------------------------------------------------------------------------
# Breadcrumbs
# --------------------------------------------------------------------------


def test_a_column_carries_its_database_schema_and_table(world: World) -> None:
    hit = _find(
        world,
        f"{world.prefix}_total_amount",
        {Labels.COLUMN},
        include_description=False,
    )[0]
    assert hit["breadcrumbs"] == [
        {"id": world.database, "name": f"{world.prefix}_warehouse", "type": Labels.DB},
        {"id": world.schema, "name": f"{world.prefix}_public", "type": Labels.SCHEMA},
        {"id": world.table, "name": f"{world.prefix}_rental", "type": Labels.TABLE},
    ]


def test_a_table_stops_at_its_schema(world: World) -> None:
    hit = _find(
        world, f"{world.prefix}_rental", {Labels.TABLE}, include_description=False
    )[0]
    assert [crumb["type"] for crumb in hit["breadcrumbs"]] == [
        Labels.DB,
        Labels.SCHEMA,
    ]


def test_a_database_has_no_ancestors(world: World) -> None:
    hit = _find(
        world, f"{world.prefix}_warehouse", {Labels.DB}, include_description=False
    )[0]
    assert hit["breadcrumbs"] == []


def test_an_attribute_points_at_its_term(world: World) -> None:
    hit = _find(
        world,
        f"{world.prefix}_net_total",
        {LABEL_COLUMN_ATTRIBUTE},
        include_description=False,
    )[0]
    assert hit["breadcrumbs"] == [
        {"id": world.term, "name": f"{world.prefix}_Revenue", "type": LABEL_TERM}
    ]


# --------------------------------------------------------------------------
# Counts
# --------------------------------------------------------------------------


def test_counts_group_by_type_and_split_views_out(world: World) -> None:
    counts = search.count_global_search(
        search.search_tokens(world.prefix), ALL_TYPES, include_description=False
    )
    assert counts[Labels.DB] == 1
    assert counts[Labels.SCHEMA] == 1
    assert counts[Labels.TABLE] == 1
    assert counts[search.SEARCH_TYPE_VIEW] == 1
    assert counts[Labels.COLUMN] == 1
    assert counts[LABEL_TERM] == 1
    assert counts[LABEL_COLUMN_ATTRIBUTE] == 1


def test_list_limit_ranks_before_cutting_so_later_union_branches_survive(
    world: World,
) -> None:
    """The cap is the best *limit* hits, not the first *limit* UNION ALL branches.

    Table/View are last in ``_hit_selects``. Database and Schema come first, so
    an unordered ``LIMIT 2`` on this fixture would keep those two and drop the
    table even though its name is shorter — the same shape as All omitting
    tables while the Tables tab still has a count.

    Tags are left out of the types rather than out of the assertion: the
    fixture's own tag names are shorter still, so they would take both slots
    and the test would stop saying anything about where Table sits in the
    union.
    """
    hits = search.fetch_global_search(
        search.search_tokens(world.prefix),
        ALL_TYPES - {search.LABEL_TAG},
        include_description=False,
        limit=2,
    )
    assert [row["name"] for row in hits] == [
        f"{world.prefix}_public",
        f"{world.prefix}_rental",
    ]


def test_list_limit_keeps_name_hits_ahead_of_earlier_description_matches(
    world: World,
) -> None:
    """A Column branch full of description hits must not crowd Table/View out of All."""
    extra_ids: list[str] = []
    term = f"{world.prefix}_rental"
    try:
        with write_transaction():
            for i in range(5):
                extra_ids.append(
                    _add(
                        s.catalog_column,
                        table_id=world.table,
                        name=f"{world.prefix}_zzzz_col_{i}",
                        description=f"mentions {term} in passing",
                    )
                )
        hits = search.fetch_global_search(
            search.search_tokens(term),
            ALL_TYPES,
            include_description=True,
            limit=2,
        )
        assert [row["name"] for row in hits] == [
            f"{world.prefix}_rental",
            f"{world.prefix}_rental_summary",
        ]
    finally:
        if extra_ids:
            with write_transaction():
                store().query_write(
                    s.catalog_column.delete().where(
                        s.catalog_column.c.id.in_(extra_ids)
                    )
                )


def test_an_unrelated_alias_does_not_outrank_a_description_hit(world: World) -> None:
    """The middle rank bucket is a *matching* synonym, not merely having one.

    Almost every Term carries aliases, so ranking on their presence hands the
    bucket to Terms whose aliases have nothing to do with the query — and the
    cap then drops the description hits the count tab still reports. Both rows
    here match on description alone, so the shorter name has to win.
    """
    needle = f"{world.prefix}zneedle"
    extra_term: str | None = None
    extra_column: str | None = None
    try:
        with write_transaction():
            extra_term = _add(
                s.term,
                name=f"{world.prefix}_zzz_aliased_term",
                source=SEMANTIC_SOURCE,
                synonyms=[f"{world.prefix} Unrelated Alias"],
                description=f"mentions {needle} in passing",
            )
            store().query_write(
                s.table__term.insert().values(table_id=world.table, term_id=extra_term)
            )
            extra_column = _add(
                s.catalog_column,
                table_id=world.table,
                name=f"{world.prefix}_c",
                description=f"mentions {needle} in passing",
            )
        hits = _find(world, needle, include_description=True, limit=1)
        assert [row["name"] for row in hits] == [f"{world.prefix}_c"]
    finally:
        with write_transaction():
            if extra_column is not None:
                store().query_write(
                    s.catalog_column.delete().where(
                        s.catalog_column.c.id == extra_column
                    )
                )
            if extra_term is not None:
                store().query_write(s.term.delete().where(s.term.c.id == extra_term))


def test_counts_are_not_capped_by_the_list_limit(world: World) -> None:
    """The tab badge reports the total, not the size of the page."""
    counts = search.count_global_search(
        search.search_tokens(world.prefix), ALL_TYPES, include_description=False
    )
    listed = search.fetch_global_search(
        search.search_tokens(world.prefix),
        ALL_TYPES,
        include_description=False,
        limit=2,
    )
    assert sum(counts.values()) > len(listed)


def test_counts_respect_the_object_filter(world: World) -> None:
    counts = search.count_global_search(
        search.search_tokens(world.prefix), {Labels.COLUMN}, include_description=False
    )
    assert set(counts) == {Labels.COLUMN}


# --------------------------------------------------------------------------
# matching_id_selects — the match a rule replays
# --------------------------------------------------------------------------
#
# The third path, and the one with no page to fill: a rule labels everything
# its search finds. So the property under test throughout is that this agrees
# with the *count* -- which is uncapped for the same reason -- rather than with
# the list, which is a page and stops at 200.


def _matching_ids(
    world: World,
    term: str,
    types: set[str] | None = None,
    **kwargs,
) -> dict[str, set[str]]:
    """Run every statement :func:`matching_id_selects` built, as a rule would."""
    selects = search.matching_id_selects(
        search.search_tokens(term),
        types or ALL_TYPES,
        synonym_tokens=search.synonym_word_tokens(term),
        **kwargs,
    )
    return {
        label: {row["id"] for row in store().query_read(statement)}
        for label, statement in selects.items()
    }


def test_the_match_is_not_capped_at_the_list_limit(world: World) -> None:
    """The whole reason this exists beside ``fetch_global_search``.

    A rule matching more columns than a page can show has to label all of
    them: capped, it would label an arbitrary two hundred and leave the rest
    unlabelled however many times ingest re-ran it.
    """
    needle = f"{world.prefix}zbulk"
    extra: list[str] = []
    try:
        with write_transaction():
            for i in range(search.LIST_LIMIT + 50):
                extra.append(
                    _add(
                        s.catalog_column,
                        table_id=world.table,
                        name=f"{needle}_{i:04d}",
                    )
                )

        listed = search.fetch_global_search(
            search.search_tokens(needle), {Labels.COLUMN}, include_description=False
        )
        matched = _matching_ids(
            world, needle, {Labels.COLUMN}, include_description=False
        )

        assert len(listed) == search.LIST_LIMIT
        assert matched[Labels.COLUMN] == set(extra)
    finally:
        if extra:
            with write_transaction():
                store().query_write(
                    s.catalog_column.delete().where(s.catalog_column.c.id.in_(extra))
                )


def test_the_match_agrees_with_the_count(world: World) -> None:
    """Same rows as the tab badges, which are the uncapped truth about a query.

    ``View`` is folded into ``Table`` here and counted apart there, so the
    comparison is over the total rather than per key -- see
    :func:`matching_id_selects` on why a view is labelled as the table it is.
    """
    counts = search.count_global_search(
        search.search_tokens(world.prefix), ALL_TYPES, include_description=False
    )
    matched = _matching_ids(world, world.prefix, include_description=False)

    assert sum(len(ids) for ids in matched.values()) == sum(counts.values())


def test_a_view_is_matched_under_the_table_it_is(world: World) -> None:
    """One key, because a view is a ``catalog_table`` row and a tag on one is a
    tag on a Table."""
    matched = _matching_ids(
        world,
        f"{world.prefix}_rental",
        {Labels.TABLE, search.SEARCH_TYPE_VIEW},
        include_description=False,
    )

    assert set(matched) == {Labels.TABLE}
    assert matched[Labels.TABLE] == {world.table, world.view}


def test_asking_only_for_views_leaves_the_table_out(world: World) -> None:
    """The two tabs still filter, even sharing a key."""
    matched = _matching_ids(
        world,
        f"{world.prefix}_rental",
        {search.SEARCH_TYPE_VIEW},
        include_description=False,
    )

    assert matched[Labels.TABLE] == {world.view}


def test_a_term_reached_only_by_an_alias_is_matched(world: World) -> None:
    """Folded in with ``or_`` rather than branched off: there is no page for an
    alias-only Term to be evicted from."""
    matched = _matching_ids(
        world, f"{world.prefix} Takings", {LABEL_TERM}, include_description=False
    )

    assert matched[LABEL_TERM] == {world.term}


def test_aliases_are_ignored_when_the_rule_says_so(world: World) -> None:
    """``synonyms: false`` in a rule's filters reaches here as no tokens."""
    selects = search.matching_id_selects(
        search.search_tokens(f"{world.prefix} Takings"),
        {LABEL_TERM},
        include_description=False,
        synonym_tokens=[],
    )

    assert store().query_read(selects[LABEL_TERM]) == []


def test_an_unrepresented_term_is_not_matched(world: World) -> None:
    """Same visibility rule as the list. A rule labelling a Term no page can
    open would be a label nobody can see, let alone remove."""
    matched = _matching_ids(
        world, f"{world.prefix}_Orphan", {LABEL_TERM}, include_description=False
    )

    assert matched[LABEL_TERM] == set()


def test_an_attribute_with_no_visible_term_is_not_matched(world: World) -> None:
    matched = _matching_ids(
        world,
        f"{world.prefix}_net",
        {LABEL_COLUMN_ATTRIBUTE},
        include_description=False,
    )

    assert matched[LABEL_COLUMN_ATTRIBUTE] == {world.attribute}


def test_descriptions_are_matched_only_when_asked(world: World) -> None:
    """The rule's own ``description`` filter, on the path it replays."""
    needle = "how much the customer paid"

    assert (
        _matching_ids(world, needle, {Labels.COLUMN}, include_description=False)[
            Labels.COLUMN
        ]
        == set()
    )
    assert (
        world.column
        in _matching_ids(world, needle, {Labels.COLUMN}, include_description=True)[
            Labels.COLUMN
        ]
    )


def test_the_matching_statements_respect_the_tag_filter(world: World) -> None:
    """What a rule filtering on a tag would label, which is the tagged rows only.

    The third path has its own loop over the labels, so the narrowing has to be
    repeated there — and it is the path where getting it wrong is worst: this
    one is uncapped and it *writes*, so a rule saved over "tagged PII" would
    label the whole match instead.
    """
    ids = _matching_ids(
        world, world.prefix, include_description=False, tag_ids=[world.tag]
    )
    assert ids[Labels.TABLE] == {world.table}
    assert ids[LABEL_TERM] == {world.term}
    assert Labels.DB not in ids


def test_a_query_that_matches_nothing_builds_no_statements() -> None:
    """Read by a caller syncing labels as "this rule matches nothing", which is
    a legitimate state for a standing rule."""
    assert search.matching_id_selects([], ALL_TYPES, include_description=False) == {}
    assert search.matching_id_selects(["x"], set(), include_description=False) == {}
