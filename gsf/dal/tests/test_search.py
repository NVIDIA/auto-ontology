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

from gsf.catalog.constants import Labels, TableTypes  # noqa: E402
from gsf.dal import schema as s  # noqa: E402
from gsf.dal import search  # noqa: E402
from gsf.dal.session import store, write_transaction  # noqa: E402
from gsf.semantic.constants import (  # noqa: E402
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
        pytest.skip(f"gsf schema unavailable (alembic upgrade head): {exc}")


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


@pytest.fixture(scope="module")
def world() -> Iterator[World]:
    """Build the fixture once, and take it back out again.

    Deleting the database cascades to its schema, table, columns and the
    ``table__term`` link; the semantic rows hang off nothing and go by hand.
    """
    with write_transaction():
        built = World()
    yield built
    with write_transaction():
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
    between — see ``gsf.server.search.service._prepare_search``.
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
    """
    hits = search.fetch_global_search(
        search.search_tokens(world.prefix),
        ALL_TYPES,
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
