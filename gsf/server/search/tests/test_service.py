# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for global-search orchestration."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from gsf.server.search.constants import MIN_SEARCH_LENGTH
from gsf.server.search.service import (
    SearchValidationError,
    global_search,
    global_search_count,
    rank_key,
    resolve_object_types,
)


def test_resolve_object_types_defaults_to_all() -> None:
    types = resolve_object_types(None)
    assert "Term" in types
    assert "Column" in types
    assert "View" in types


def test_resolve_object_types_empty_list_is_all() -> None:
    assert resolve_object_types([]) == resolve_object_types(None)


def test_resolve_object_types_rejects_unknown() -> None:
    with pytest.raises(SearchValidationError, match="metric"):
        resolve_object_types(["Term", "metric"])


def test_rank_key_prefers_names_containing_the_term_then_shorter() -> None:
    items = [
        {"name": "other"},
        {"name": "annual_revenue_total"},
        {"name": "revenue"},
        {"name": "Revenue_id"},
    ]
    items.sort(key=rank_key("revenue"))
    assert [row["name"] for row in items] == [
        "revenue",
        "Revenue_id",
        "annual_revenue_total",
        "other",
    ]


def test_rank_key_uses_tokens_not_the_raw_query() -> None:
    """Matching splits ``created-at``; ranking must too or every hit is bucket 2."""
    items = [
        {"name": "zz"},
        {"name": "created_at_timestamp"},
        {"name": "created_at"},
    ]
    items.sort(key=rank_key("created-at"))
    assert [row["name"] for row in items] == [
        "created_at",
        "created_at_timestamp",
        "zz",
    ]


def test_rank_key_orders_synonym_after_name_before_description_only() -> None:
    items = [
        {"name": "other"},
        {"name": "Cluster Passport", "synonyms": ["Customer"]},
        {"name": "Customer"},
    ]
    items.sort(key=rank_key("Customer"))
    assert [row["name"] for row in items] == [
        "Customer",
        "Cluster Passport",
        "other",
    ]


def test_short_query_skips_the_database_and_returns_empty() -> None:
    with patch("gsf.server.search.service.search_dal.fetch_global_search") as fetch:
        result = global_search(
            search_term="a",
            text_match_option="contains",
            objects=None,
            include_description=True,
        )
    assert result == {"data": [], "count": 0}
    fetch.assert_not_called()


def test_min_length_query_reaches_the_database() -> None:
    with patch("gsf.server.search.service.search_dal.fetch_global_search") as fetch:
        fetch.return_value = []
        global_search(
            search_term="x" * MIN_SEARCH_LENGTH,
            text_match_option="contains",
            objects=None,
            include_description=True,
        )
    fetch.assert_called_once()


def test_specials_only_query_is_treated_as_empty() -> None:
    with patch("gsf.server.search.service.search_dal.fetch_global_search") as fetch:
        result = global_search_count(
            search_term="**",
            text_match_option="contains",
            objects=None,
            include_description=False,
        )
    assert result == {"data": {}}
    fetch.assert_not_called()


def test_unsupported_match_option_raises() -> None:
    with pytest.raises(SearchValidationError, match="starts_with"):
        global_search(
            search_term="revenue",
            text_match_option="starts_with",
            objects=None,
            include_description=False,
        )


@patch("gsf.server.search.service.search_dal.fetch_global_search")
def test_global_search_normalizes_and_ranks(fetch: MagicMock) -> None:
    fetch.return_value = [
        {
            "id": "1",
            "name": "zz_revenue",
            "label": "Table",
            "table_type": "BASE TABLE",
            "description": None,
            "certified": None,
            "parent_id": None,
            "breadcrumbs": [
                {"name": "sales", "type": "Database"},
                {"name": None, "type": "Schema"},
            ],
        },
        {
            "id": "2",
            "name": "revenue",
            "label": "Term",
            "table_type": None,
            "description": "money",
            "certified": "pending",
            "parent_id": None,
            "breadcrumbs": [],
        },
    ]

    result = global_search(
        search_term="revenue",
        text_match_option="contains",
        objects=["Term", "Table"],
        include_description=True,
    )

    assert [item["id"] for item in result["data"]] == ["2", "1"]
    assert result["data"][1]["breadcrumbs"] == [{"name": "sales", "type": "Database"}]
    assert result["data"][1]["type"] == "Table"
    assert result["data"][1]["table_type"] == "BASE TABLE"
    assert result["data"][0]["type"] == "Term"
    assert result["count"] == 2
    fetch.assert_called_once()
    kwargs = fetch.call_args
    assert kwargs.args[0] == ["revenue"]
    assert kwargs.args[1] == {"Term", "Table"}
    assert kwargs.kwargs["include_description"] is True
    assert kwargs.kwargs["synonym_tokens"] == ["revenue"]


@patch("gsf.server.search.service.search_dal.fetch_global_search")
def test_global_search_keeps_breadcrumb_ids(fetch: MagicMock) -> None:
    fetch.return_value = [
        {
            "id": "col-1",
            "name": "customer_id",
            "label": "Column",
            "table_type": None,
            "description": None,
            "certified": None,
            "parent_id": "tbl-1",
            "breadcrumbs": [
                {"id": "db-1", "name": "sales", "type": "Database"},
                {"id": "sch-1", "name": "public", "type": "Schema"},
                {"id": "tbl-1", "name": "customers", "type": "Table"},
            ],
        },
    ]
    result = global_search(
        search_term="customer",
        text_match_option="contains",
        objects=["Column"],
        include_description=True,
    )
    assert result["data"][0]["breadcrumbs"] == [
        {"id": "db-1", "name": "sales", "type": "Database"},
        {"id": "sch-1", "name": "public", "type": "Schema"},
        {"id": "tbl-1", "name": "customers", "type": "Table"},
    ]
    assert result["data"][0]["parent_id"] == "tbl-1"


@patch("gsf.server.search.service.search_dal.count_global_search")
def test_global_search_count_passes_object_filter(count: MagicMock) -> None:
    count.return_value = {"Term": 2, "Column": 4}
    result = global_search_count(
        search_term="id",
        text_match_option="contains",
        objects=["Column"],
        include_description=False,
    )
    assert result == {"data": {"Term": 2, "Column": 4}}
    assert count.call_args.args[1] == {"Column"}
    assert count.call_args.kwargs["synonym_tokens"] == ["id"]


@patch("gsf.server.search.service.search_dal.fetch_global_search")
def test_global_search_keeps_matching_synonyms_only(fetch: MagicMock) -> None:
    fetch.return_value = [
        {
            "id": "term-1",
            "name": "Business Unit",
            "label": "Term",
            "table_type": None,
            "description": None,
            "certified": "pending",
            "parent_id": None,
            "breadcrumbs": [],
            "synonyms": ["BU", "Org"],
        },
    ]
    result = global_search(
        search_term="BU",
        text_match_option="contains",
        objects=["Term"],
        include_description=False,
    )
    assert result["data"][0]["synonyms"] == ["BU"]
    assert fetch.call_args.kwargs["synonym_tokens"] == ["bu"]


@patch("gsf.server.search.service.search_dal.fetch_global_search")
def test_global_search_keeps_table_label_for_views(fetch: MagicMock) -> None:
    fetch.return_value = [
        {
            "id": "v-1",
            "name": "orders_v",
            "label": "Table",
            "table_type": "view",
            "description": None,
            "certified": None,
            "parent_id": None,
            "breadcrumbs": [],
            "synonyms": [],
        },
    ]
    result = global_search(
        search_term="orders",
        text_match_option="contains",
        objects=["View"],
        include_description=False,
    )
    assert result["data"][0]["type"] == "Table"
    assert result["data"][0]["table_type"] == "view"
    assert fetch.call_args.args[1] == {"View"}


@patch("gsf.server.search.service.search_dal.LIST_LIMIT", 3)
@patch("gsf.server.search.service.search_dal.fetch_global_search")
def test_synonym_only_term_survives_list_cap(fetch: MagicMock) -> None:
    fetch.return_value = [
        {
            "id": "col-1",
            "name": "customer_a",
            "label": "Column",
            "table_type": None,
            "description": None,
            "certified": None,
            "parent_id": None,
            "breadcrumbs": [],
            "synonyms": [],
        },
        {
            "id": "col-2",
            "name": "customer_b",
            "label": "Column",
            "table_type": None,
            "description": None,
            "certified": None,
            "parent_id": None,
            "breadcrumbs": [],
            "synonyms": [],
        },
        {
            "id": "col-3",
            "name": "customer_c",
            "label": "Column",
            "table_type": None,
            "description": None,
            "certified": None,
            "parent_id": None,
            "breadcrumbs": [],
            "synonyms": [],
        },
        {
            "id": "term-syn",
            "name": "Cluster Passport",
            "label": "Term",
            "table_type": None,
            "description": None,
            "certified": "pending",
            "parent_id": None,
            "breadcrumbs": [],
            "synonyms": ["Customer"],
        },
    ]
    result = global_search(
        search_term="Customer",
        text_match_option="contains",
        objects=None,
        include_description=True,
    )
    ids = [item["id"] for item in result["data"]]
    assert "term-syn" in ids
    assert len(result["data"]) == 3
    assert result["data"][0]["id"] != "term-syn"


def _synonym_only_term(index: int) -> dict[str, Any]:
    return {
        "id": f"term-{index}",
        "name": f"Cluster Passport {index}",
        "label": "Term",
        "table_type": None,
        "description": None,
        "certified": "pending",
        "parent_id": None,
        "breadcrumbs": [],
        "synonyms": ["Customer"],
    }


def _name_hit_column(index: int) -> dict[str, Any]:
    return {
        "id": f"col-{index}",
        "name": f"customer_{index}",
        "label": "Column",
        "table_type": None,
        "description": None,
        "certified": None,
        "parent_id": None,
        "breadcrumbs": [],
        "synonyms": [],
    }


@patch("gsf.server.search.service.search_dal.LIST_LIMIT", 4)
@patch("gsf.server.search.service.search_dal.fetch_global_search")
def test_synonym_rescue_cannot_evict_the_whole_page(fetch: MagicMock) -> None:
    """Aliases get a bounded share of the page, not all of it.

    Unbounded, the rescue swaps one name hit out per alias-only Term and walks
    the page from the worst-ranked end to index 0 — so the exact match the user
    typed disappears behind a page of Terms whose names look nothing like it.
    """
    fetch.return_value = [
        {
            "id": "col-exact",
            "name": "customer",
            "label": "Column",
            "table_type": None,
            "description": None,
            "certified": None,
            "parent_id": None,
            "breadcrumbs": [],
            "synonyms": [],
        },
        *(_name_hit_column(index) for index in range(3)),
        *(_synonym_only_term(index) for index in range(6)),
    ]
    result = global_search(
        search_term="Customer",
        text_match_option="contains",
        objects=None,
        include_description=True,
    )
    ids = [item["id"] for item in result["data"]]
    assert len(ids) == 4
    assert ids[0] == "col-exact"
    assert sum(1 for item in result["data"] if item["synonyms"]) == 1


@patch("gsf.server.search.service.search_dal.LIST_LIMIT", 20)
@patch("gsf.server.search.service.search_dal.fetch_global_search")
def test_synonym_rescue_claims_only_its_share_of_the_page(fetch: MagicMock) -> None:
    """The bound is a share of the page, not the one slot ``max`` guarantees.

    At a page of four the budget floors at 1 either way, so that size cannot
    tell a proportion from a constant. Twenty can: 25 name hits and 10
    alias-only Terms have to come back as 18 and 2.
    """
    fetch.return_value = [
        *(_name_hit_column(index) for index in range(25)),
        *(_synonym_only_term(index) for index in range(10)),
    ]
    result = global_search(
        search_term="Customer",
        text_match_option="contains",
        objects=None,
        include_description=True,
    )
    assert len(result["data"]) == 20
    assert result["data"][0]["id"] == "col-0"
    assert sum(1 for item in result["data"] if item["synonyms"]) == 2


@patch("gsf.server.search.service.search_dal.fetch_global_search")
def test_an_unmatched_alias_does_not_claim_the_middle_rank(fetch: MagicMock) -> None:
    """``rank_key``'s middle bucket is a *matching* alias, not the presence of one.

    ``rank_key`` tests ``item["synonyms"]`` for truth, which reads like "has any
    alias". It is not, and only because ``_normalize_item`` has already reduced
    that list to the aliases this query matched. That reduction is the whole
    reason this function agrees with ``search._list_rank``, which asks Postgres
    for ``synonym_hit``. Lose it and every aliased Term outranks every
    description hit here while Postgres still ranks them below — and the cap
    then drops rows the count tab reports.

    Both rows below match on description alone, so the shorter name has to win.
    """
    fetch.return_value = [
        {
            "id": "term-aliased",
            "name": "zzzzzzzz",
            "label": "Term",
            "table_type": None,
            "description": "mentions customer in passing",
            "certified": "pending",
            "breadcrumbs": [],
            "synonyms": ["Totally Unrelated"],
        },
        {
            "id": "col-short",
            "name": "zz",
            "label": "Column",
            "table_type": None,
            "description": "mentions customer in passing",
            "certified": None,
            "breadcrumbs": [],
            "synonyms": [],
        },
    ]
    result = global_search(
        search_term="Customer",
        text_match_option="contains",
        objects=None,
        include_description=True,
    )
    assert [item["id"] for item in result["data"]] == ["col-short", "term-aliased"]
    assert result["data"][1]["synonyms"] == []
