# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Turning a typed query into tokens, without touching a database.

The counterpart to ``auto_ontology/dal/tests/test_search.py``, which runs the queries
themselves and is skipped when no Postgres is configured. Everything here is a
pure function, so it runs everywhere — which matters, because these decide what
the query *means* before any index is consulted.

A search term is read twice, and the two readings deliberately disagree:
``name`` and ``description`` match on a substring, ``Term.synonyms`` on whole
words. See the module docstring in ``auto_ontology.dal.search`` for why.
"""

from __future__ import annotations

import pytest

from auto_ontology.catalog.constants import Labels, TableTypes
from auto_ontology.dal import schema as s
from auto_ontology.dal.search import (
    SEARCH_OBJECT_TYPES,
    SEARCH_TABLES,
    SEARCH_TYPE_VIEW,
    filter_special_characters,
    is_view_table_type,
    search_object_type,
    search_tokens,
    synonym_matches_tokens,
    synonym_word_tokens,
)


def test_filter_special_characters_replaces_separators_with_spaces() -> None:
    assert filter_special_characters("foo/bar*baz") == "foo bar baz"


@pytest.mark.parametrize(
    ("term", "expected"),
    [
        ("revenue", ["revenue"]),
        ("annual revenue", ["annual", "revenue"]),
        ("Annual Revenue", ["annual", "revenue"]),
        ("rev/enue", ["rev", "enue"]),
        # A hyphen splits, so `created-at` still reaches `created_at`.
        ("created-at", ["created", "at"]),
        ("2024-01-01", ["2024", "01", "01"]),
        ("(revenue)", ["revenue"]),
        # An underscore does not: splitting `total_amount` would make it match
        # every table with a total and every table with an amount.
        ("total_amount", ["total_amount"]),
        ("!!!", []),
        ("   ", []),
    ],
)
def test_search_tokens(term: str, expected: list[str]) -> None:
    assert search_tokens(term) == expected


def test_synonym_word_tokens_are_alphanumeric_whole_words() -> None:
    assert synonym_word_tokens("BU") == ["bu"]
    assert synonym_word_tokens("Business Unit") == ["business", "unit"]
    assert synonym_word_tokens("rev/enue") == ["rev", "enue"]


def test_synonym_word_tokens_drop_the_underscore_search_tokens_keep() -> None:
    """The two readings of one query, and the clearest case of them differing.

    A synonym is prose — ``Business Unit`` — so an underscore in the query is
    punctuation. A name is an identifier, where it is part of the word.
    """
    assert synonym_word_tokens("total_amount") == ["total", "amount"]
    assert search_tokens("total_amount") == ["total_amount"]


def test_synonym_matches_tokens_requires_whole_words_in_one_synonym() -> None:
    assert synonym_matches_tokens("BU", ["bu"]) is True
    assert synonym_matches_tokens("BU", ["b"]) is False
    assert synonym_matches_tokens("Business Unit", ["unit"]) is True
    assert synonym_matches_tokens("Business Unit", ["uni"]) is False
    assert synonym_matches_tokens("Business Unit", ["business", "unit"]) is True
    assert synonym_matches_tokens("Business", ["business", "unit"]) is False


def test_synonym_matches_nothing_when_there_are_no_tokens() -> None:
    """Zero tokens must not read as "every token matched"."""
    assert synonym_matches_tokens("Business Unit", []) is False


def test_search_object_type_uses_labels_and_the_view_table_type() -> None:
    assert search_object_type("Term") == "Term"
    assert search_object_type("ColumnAttribute") == "ColumnAttribute"
    assert search_object_type(Labels.CUSTOM_ANALYSIS) == "CustomAnalysis"
    assert search_object_type(Labels.TABLE, TableTypes.BASE_TABLE) == Labels.TABLE
    assert search_object_type(Labels.TABLE, "view") == SEARCH_TYPE_VIEW
    assert search_object_type(Labels.TABLE, "materialized view") == SEARCH_TYPE_VIEW
    assert search_object_type(None) is None


def test_view_table_types_remap_to_view() -> None:
    assert is_view_table_type("view") is True
    assert is_view_table_type("materialized view") is True
    assert is_view_table_type("VIEW") is True
    assert is_view_table_type("base table") is False
    assert is_view_table_type(None) is False


def test_object_types_are_the_labels_plus_view() -> None:
    assert set(SEARCH_OBJECT_TYPES) == set(SEARCH_TABLES) | {SEARCH_TYPE_VIEW}


def test_every_searched_table_has_a_trigram_index() -> None:
    """The queries and the indexes behind them must not drift apart.

    Adding a searchable entity without its index is not a failure anyone sees:
    the search still returns the right rows, by scanning the table on every
    keystroke.
    """
    assert set(SEARCH_TABLES.values()) == set(s.TRIGRAM_SEARCH_TABLES)
