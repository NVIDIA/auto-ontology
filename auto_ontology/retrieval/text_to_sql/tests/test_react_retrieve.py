# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pandas as pd
import pytest

from auto_ontology.retrieval.text_to_sql.react import retrieve as tool
from auto_ontology.retrieval.text_to_sql.react import siblings
from auto_ontology.retrieval.text_to_sql.react.context import ToolContext

ORDERS = {
    "id": "t-orders",
    "name": "orders",
    "schema_name": "sales",
    "database_name": "shop",
    "description": "One row per order",
    "pk": ["order_id"],
    "columns": [
        {"name": "order_id", "data_type": "INT"},
        {
            "name": "cust_ref",
            "data_type": "INT",
            "description": "Who placed the order",
            "semantic_fk": "crm.customers.customer_id",
        },
        {"name": "amount", "data_type": "REAL"},
        {"name": "discount", "data_type": "REAL"},
    ],
}
CUSTOMERS = {
    "id": "t-customers",
    "name": "customers",
    "schema_name": "crm",
    "database_name": "shop",
    "columns": [
        {"name": "customer_id", "data_type": "INT"},
        {
            "name": "status",
            "data_type": "TEXT",
            "sample_values": ["ACTIVE", "CHURNED", "PENDING"],
        },
    ],
}
ELSEWHERE = {**CUSTOMERS, "id": "t-other", "database_name": "other"}
TABLES = {t["id"]: t for t in (ORDERS, CUSTOMERS, ELSEWHERE)}

REVENUE = {
    "id": "a-revenue",
    "label": "SqlAttribute",
    "name": "Revenue",
    "description": "Money kept from paid orders",
    "sql": "SELECT SUM(amount - discount) FROM sales.orders",
    "term_name": "Order",
    "relevant_tables": [{"id": "t-orders"}],
}
STATUS_ATTRIBUTE = {"id": "a-status", "label": "ColumnAttribute", "name": "Status"}
AMOUNT_ATTRIBUTE = {"id": "a-amount", "label": "ColumnAttribute", "name": "Amount"}
CUSTOMER_TERM = {"id": "term-customer", "label": "Term", "name": "Customer"}
ATTRIBUTE_COLUMNS = {
    "a-status": {"col_name": "status", "table_id": "t-customers"},
    "a-amount": {"col_name": "amount", "table_id": "t-orders"},
    "a-elsewhere": {"col_name": "status", "table_id": "t-other"},
}


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[Any]]:
    """Stub every data-access call the tool makes, recording what it asked."""
    seen: dict[str, list[Any]] = {
        "meaning": [],
        "names": [],
        "name_scopes": [],
        "join": [],
    }
    meaning_hits: dict[str, list[dict]] = {
        "how is revenue calculated": [REVENUE],
        "customer status": [STATUS_ATTRIBUTE],
        "order amount": [AMOUNT_ATTRIBUTE, REVENUE, AMOUNT_ATTRIBUTE],
        "customers": [CUSTOMER_TERM],
    }

    def meaning(
        _retriever: object,
        query: str,
        list_of_semantic: list[str],
        database_name: str | None,
        per_label_k: dict[str, int],
    ) -> list[dict]:
        seen["meaning"].append((query, list_of_semantic, database_name))
        return [dict(h) for h in meaning_hits.get(query, [])]

    def names(
        tokens: list[str],
        object_types: set[str],
        *,
        include_description: bool,
        data_ids: list[str] | None,
        limit: int,
    ) -> list[dict]:
        seen["names"].append((tokens, include_description, object_types))
        seen["name_scopes"].append((set(object_types), data_ids))
        if tokens == ["cust_ref"] and "Column" in object_types:
            return [{"id": "c-cust-ref", "label": "Column", "name": "cust_ref"}]
        return []

    def term_tables(
        *, term_ids: list[str] | None, database_name: str | None
    ) -> list[dict]:
        return [
            {"term_id": "term-customer", "table_id": "t-orders", "path": "SEMANTIC_FK"},
            {
                "term_id": "term-customer",
                "table_id": "t-customers",
                "path": "REPRESENTS",
            },
        ]

    def expand(ids_and_labels: list[dict] | None) -> dict[str, dict]:
        return {
            "c-cust-ref": {"name": "cust_ref", "relevant_tables": [{"id": "t-orders"}]}
        }

    def join_path(anchor: str, dest: str) -> list[dict]:
        seen["join"].append((anchor, dest))
        return [
            {
                "hops": [
                    {
                        "source_schema": "sales",
                        "source_table": "orders",
                        "source_column": "cust_ref",
                        "target_schema": "crm",
                        "target_table": "customers",
                        "target_column": "customer_id",
                        "key": "Customer.customer_id",
                    }
                ],
                "default": True,
            }
        ]

    monkeypatch.setattr(tool, "_get_candidates_information", meaning)
    monkeypatch.setattr(tool, "fetch_global_search", names)
    monkeypatch.setattr(tool, "fetch_schema_ids_for_database", lambda _db: [])
    monkeypatch.setattr(tool, "expand_info", expand)
    monkeypatch.setattr(
        tool, "fetch_tables_by_ids", lambda ids: [TABLES[i] for i in ids if i in TABLES]
    )
    monkeypatch.setattr(
        tool,
        "find_column_id_by_table_and_name",
        lambda table, column, database: f"{table}.{column}",
    )
    monkeypatch.setattr(tool, "find_join_routes", join_path)
    monkeypatch.setattr(
        tool,
        "fetch_attr_column_contexts",
        lambda ids, **_: {
            i: ATTRIBUTE_COLUMNS[i] for i in ids if i in ATTRIBUTE_COLUMNS
        },
    )
    monkeypatch.setattr(tool, "fetch_term_table_pairs", term_tables)
    monkeypatch.setattr(siblings, "fetch_table_column_names", lambda _db: [])
    return seen


def _ctx() -> ToolContext:
    return ToolContext(retriever=object(), connectors=[], database_name="shop")


def test_definition_comes_back_with_its_sql_and_its_table(calls: dict) -> None:
    text = tool.retrieve(_ctx(), "how is revenue calculated")

    assert "ATTRIBUTE Revenue (of Order) — Money kept from paid orders" in text
    assert "SQL: SELECT SUM(amount - discount) FROM sales.orders" in text
    assert "TABLE sales.orders — One row per order" in text
    assert (
        "cust_ref INT — Who placed the order; references crm.customers.customer_id"
        in text
    )


def test_second_search_joins_new_tables_to_ones_already_shown(calls: dict) -> None:
    ctx = _ctx()
    tool.retrieve(ctx, "how is revenue calculated")
    text = tool.retrieve(ctx, "customer status")

    assert "TABLE crm.customers" in text
    assert (
        "status TEXT ← matches the query — values: 'ACTIVE', 'CHURNED', 'PENDING'"
        in text
    )
    assert (
        "JOIN crm.customers ↔ sales.orders: "
        "sales.orders.cust_ref = crm.customers.customer_id [key Customer.customer_id]"
    ) in text
    assert calls["join"] == [("customers.customer_id", "orders.order_id")]


def test_every_direct_join_is_listed_with_the_default_first(
    calls: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    def hop(column: str) -> dict:
        return {
            "source_schema": "sales",
            "source_table": "orders",
            "source_column": column,
            "target_schema": "crm",
            "target_table": "customers",
            "target_column": "customer_id",
            "key": "Customer.customer_id",
        }

    routes = [
        {"hops": [hop("cust_ref")], "default": True},
        {"hops": [hop("billing_ref")], "default": False},
    ]
    monkeypatch.setattr(tool, "find_join_routes", lambda _a, _d: routes)
    ctx = _ctx()
    tool.retrieve(ctx, "how is revenue calculated")
    text = tool.retrieve(ctx, "customer status")

    assert (
        "JOIN crm.customers ↔ sales.orders, one of:\n"
        "  sales.orders.cust_ref = crm.customers.customer_id"
        " [key Customer.customer_id] (default)\n"
        "  sales.orders.billing_ref = crm.customers.customer_id"
        " [key Customer.customer_id]"
    ) in text
    assert ctx.join_paths == [{"path": routes[0]["hops"]}]


def test_a_table_already_shown_is_not_repeated_in_full(calls: dict) -> None:
    ctx = _ctx()
    tool.retrieve(ctx, "customer status")
    text = tool.retrieve(ctx, "customer status")

    assert text == (
        "TABLE crm.customers (shown earlier; matched columns only)\n"
        "  status TEXT — values: 'ACTIVE', 'CHURNED', 'PENDING'"
    )


def test_identifier_words_are_also_searched_by_physical_name(calls: dict) -> None:
    text = tool.retrieve(_ctx(), "which orders have cust_ref set")

    by_name = [types for tokens, _, types in calls["names"] if tokens == ["cust_ref"]]
    assert by_name and {"Table", "Column"} <= by_name[0]
    assert "TABLE sales.orders" in text


def test_a_one_word_query_is_also_searched_by_physical_name(calls: dict) -> None:
    tool.retrieve(_ctx(), "custref")

    by_name = [types for tokens, _, types in calls["names"] if tokens == ["custref"]]
    assert any({"Table", "Column"} <= types for types in by_name)


def test_a_sentence_searches_only_the_semantic_layer_by_name(calls: dict) -> None:
    tool.retrieve(_ctx(), "customer status")

    assert calls["names"] == [
        (
            ["customer", "status"],
            True,
            {"Term", "ColumnAttribute", "SqlAttribute", "CustomAnalysis"},
        )
    ]


def test_an_attribute_brings_back_its_columns_table(calls: dict) -> None:
    text = tool.retrieve(_ctx(), "customer status")

    assert "TABLE crm.customers" in text
    assert (
        "status TEXT ← matches the query — values: 'ACTIVE', 'CHURNED', 'PENDING'"
        in text
    )
    assert "ATTRIBUTE" not in text


def test_a_term_brings_back_the_table_it_represents(calls: dict) -> None:
    text = tool.retrieve(_ctx(), "customers")

    assert "TABLE crm.customers" in text
    assert "TABLE sales.orders" not in text
    assert "TERM" not in text


def test_tables_in_another_database_are_left_out(
    calls: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        tool,
        "_get_candidates_information",
        lambda *a, **k: [
            {"id": "a-elsewhere", "label": "ColumnAttribute", "name": "Status"}
        ],
    )
    text = tool.retrieve(_ctx(), "status")

    assert text.startswith("No matches for 'status'")


def test_tables_in_another_database_do_not_use_up_the_limit(
    calls: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tool.config, "max_tables_per_retrieve", lambda: 1)
    monkeypatch.setattr(
        tool,
        "_get_candidates_information",
        lambda *a, **k: [
            {"id": "t-other", "label": "Table", "name": "customers"},
            {"id": "t-customers", "label": "Table", "name": "customers"},
        ],
    )

    text = tool.retrieve(_ctx(), "customers")

    assert "TABLE crm.customers" in text


def test_physical_names_are_searched_in_the_target_database_only(
    calls: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        tool,
        "fetch_schema_ids_for_database",
        lambda db: ["s-sales", "s-crm"] if db == "shop" else [],
    )

    tool.retrieve(_ctx(), "cust_ref", kinds=["term", "definition"])

    semantic = {tool.LABEL_TERM, tool.LABEL_SQL_ATTRIBUTE}
    assert calls["name_scopes"] == [
        (semantic, None),
        ({tool.Labels.TABLE}, ["s-sales", "s-crm"]),
        (semantic, None),
    ]


def test_names_are_searched_everywhere_without_a_target_database(
    calls: dict,
) -> None:
    ctx = ToolContext(retriever=object(), connectors=[])

    tool.retrieve(ctx, "customers")

    assert calls["name_scopes"]
    assert all(data_ids is None for _types, data_ids in calls["name_scopes"])


def test_kinds_narrow_the_search(calls: dict) -> None:
    tool.retrieve(_ctx(), "how is revenue calculated", kinds=["attribute"])
    tool.retrieve(_ctx(), "how is revenue calculated", kinds=["term", "definition"])

    assert calls["meaning"][0][1] == ["ColumnAttribute"]
    assert calls["meaning"][0][2] == "shop"
    assert calls["meaning"][1][1] == ["Term", "SqlAttribute"]


def test_unknown_kinds_search_everything() -> None:
    assert len(tool.resolve_kinds(["nonsense"])) == len(tool.resolve_kinds(None))


_OPAQUE_TABLE = {
    "schema_name": "crm",
    "name": "customers",
    "columns": [
        {"name": "c1", "data_type": "INT"},
        {"name": "c_nm", "data_type": "TEXT"},
        {"name": "fld_07", "data_type": "TEXT"},
    ],
}


def test_a_matched_column_is_tagged_and_the_others_are_not() -> None:
    text = tool._table_block(_OPAQUE_TABLE, {"C_NM"}, False)

    assert text.splitlines() == [
        "TABLE crm.customers",
        "  c1 INT",
        "  c_nm TEXT ← matches the query",
        "  fld_07 TEXT",
    ]


def test_no_column_is_tagged_when_the_search_matched_none() -> None:
    text = tool._table_block(_OPAQUE_TABLE, set(), False)

    assert "matches the query" not in text


def test_a_shortened_table_tags_its_matched_column() -> None:
    text = tool._short_table_block(_OPAQUE_TABLE, {"c_nm"})

    assert "  c_nm TEXT ← matches the query" in text.splitlines()
    assert "  c1 INT" not in text


def test_wide_tables_show_matched_key_and_reference_columns(
    calls: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEXT_TO_SQL_REACT_MAX_COLUMNS_PER_TABLE", "3")
    text = tool.retrieve(_ctx(), "how is revenue calculated")

    assert "order_id INT" in text
    assert "cust_ref INT" in text
    assert "amount REAL" not in text
    assert "… 2 more columns not shown" in text


def test_output_is_capped(calls: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEXT_TO_SQL_REACT_MAX_OBSERVATION_CHARS", "200")
    text = tool.retrieve(_ctx(), "how is revenue calculated")

    assert len(text) < 400
    assert "ALSO FOUND, not shown for space" in text


def test_what_does_not_fit_is_still_named(
    calls: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEXT_TO_SQL_REACT_MAX_OBSERVATION_CHARS", "120")
    text = tool.retrieve(_ctx(), "how is revenue calculated")

    assert len(text) <= 120
    assert text.endswith(
        "ALSO FOUND, not shown for space — retrieve one by name to see it: "
        "sales.orders, ATTRIBUTE Revenue (of Order)"
    )


def test_a_notice_longer_than_the_limit_is_cut_to_it(
    calls: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEXT_TO_SQL_REACT_MAX_OBSERVATION_CHARS", "60")
    text = tool.retrieve(_ctx(), "how is revenue calculated")

    assert len(text) <= 60
    assert text.startswith("ALSO FOUND")
    assert text.endswith("…")


def test_a_notice_is_cut_between_names(
    calls: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEXT_TO_SQL_REACT_MAX_OBSERVATION_CHARS", "90")
    text = tool.retrieve(_ctx(), "how is revenue calculated")

    assert len(text) <= 90
    assert text.endswith("see it: sales.orders…")


def test_the_also_found_line_fits_inside_the_limit(
    calls: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEXT_TO_SQL_REACT_MAX_OBSERVATION_CHARS", "200")
    text = tool.retrieve(_ctx(), "how is revenue calculated")

    assert len(text) <= 200
    assert "ALSO FOUND" in text


def test_a_table_shown_earlier_is_not_named_again(
    calls: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = _ctx()
    tool.retrieve(ctx, "order amount")
    monkeypatch.setenv("TEXT_TO_SQL_REACT_MAX_OBSERVATION_CHARS", "10")
    text = tool.retrieve(ctx, "order amount")

    assert "sales.orders" not in text


def test_tables_past_the_table_limit_are_named(
    calls: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEXT_TO_SQL_REACT_MAX_TABLES_PER_RETRIEVE", "1")
    monkeypatch.setattr(
        tool, "_resolve_tables", lambda ctx, hits: (["t-orders", "t-customers"], {})
    )
    text = tool.retrieve(_ctx(), "customers")

    assert "TABLE sales.orders" in text
    assert "TABLE crm.customers" not in text
    assert text.endswith("retrieve one by name to see it: crm.customers")


def test_a_table_shown_shortened_is_not_named_again_past_the_limit(
    calls: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEXT_TO_SQL_REACT_MAX_TABLES_PER_RETRIEVE", "1")
    monkeypatch.setattr(
        tool, "_resolve_tables", lambda ctx, hits: (["t-orders", "t-customers"], {})
    )
    ctx = _ctx()
    # What an earlier, cramped result leaves behind: the table was named in a
    # shortened block, so it is shown but not in ``seen_tables``.
    ctx.show_table("crm", "customers")

    text = tool.retrieve(ctx, "customers")

    assert "TABLE sales.orders" in text
    assert "TABLE crm.customers" not in text
    assert "ALSO FOUND" not in text


def test_empty_query_is_refused(calls: dict) -> None:
    assert tool.retrieve(_ctx(), "  ").startswith("Error: query is empty")


@pytest.fixture
def physical(calls: dict, monkeypatch: pytest.MonkeyPatch) -> dict[str, list]:
    """Two retrievers: the semantic one finds nothing, the data one finds a
    covered column, an uncovered foreign key and a covered table."""
    data_retriever = object()
    seen: dict[str, list] = {"coverage": []}

    def meaning(
        retriever: object,
        query: str,
        list_of_semantic: list[str],
        database_name: str | None,
        per_label_k: dict[str, int],
    ) -> list[dict]:
        if retriever is not data_retriever:
            return []
        seen["labels"] = list_of_semantic
        return [
            {
                "id": "c-status",
                "label": "Column",
                "relevant_tables": [{"id": "t-customers"}],
            },
            {
                "id": "c-cust-ref",
                "label": "Column",
                "name": "cust_ref",
                "relevant_tables": [{"id": "t-orders"}],
            },
            {"id": "t-customers", "label": "Table"},
        ]

    def coverage(
        column_ids: list[str], table_ids: list[str]
    ) -> tuple[set[str], set[str]]:
        seen["coverage"].append((column_ids, table_ids))
        return {"c-status"}, {"t-customers"}

    monkeypatch.setattr(tool, "_get_candidates_information", meaning)
    monkeypatch.setattr(tool, "fetch_semantic_coverage", coverage)
    seen["ctx"] = [
        ToolContext(
            retriever=object(),
            data_retriever=data_retriever,
            connectors=[],
            database_name="shop",
        )
    ]
    return seen


def test_physical_rows_fill_only_what_the_semantic_layer_lacks(physical: dict) -> None:
    text = tool.retrieve(physical["ctx"][0], "who placed the order")

    assert "TABLE sales.orders" in text
    assert "references crm.customers.customer_id" in text
    assert "TABLE crm.customers" not in text


def test_a_kind_searches_its_physical_counterpart(physical: dict) -> None:
    tool.retrieve(physical["ctx"][0], "order reference", kinds=["attribute"])

    assert physical["labels"] == ["Column"]


def test_tables_and_columns_show_only_their_own_names(calls: dict) -> None:
    text = tool.retrieve(_ctx(), "order amount")

    assert "TABLE sales.orders — One row per order" in text
    assert "  cust_ref INT — Who placed the order; references" in text
    assert "[" not in text.split("\n\n")[0]


def test_an_attributes_column_survives_the_wide_table_cut(
    calls: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEXT_TO_SQL_REACT_MAX_COLUMNS_PER_TABLE", "3")
    text = tool.retrieve(_ctx(), "order amount")

    assert "amount REAL" in text


def test_a_table_found_twice_is_shown_once(calls: dict) -> None:
    text = tool.retrieve(_ctx(), "order amount")

    assert text.count("TABLE sales.orders") == 1


def test_tables_come_before_definitions(calls: dict) -> None:
    text = tool.retrieve(_ctx(), "order amount")

    assert text.index("TABLE sales.orders") < text.index("ATTRIBUTE Revenue")


def test_a_table_the_limit_cut_comes_back_in_full(
    calls: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = _ctx()
    monkeypatch.setenv("TEXT_TO_SQL_REACT_MAX_OBSERVATION_CHARS", "60")
    cut = tool.retrieve(ctx, "customer status")
    monkeypatch.setenv("TEXT_TO_SQL_REACT_MAX_OBSERVATION_CHARS", "4000")
    again = tool.retrieve(ctx, "customer status")

    assert "TABLE crm.customers" not in cut
    assert "shown earlier" not in again
    assert (
        "status TEXT ← matches the query — values: 'ACTIVE', 'CHURNED', 'PENDING'"
        in again
    )


def test_a_table_too_long_for_the_limit_is_shown_short(
    calls: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    notes = [
        {"name": f"note_{i}", "data_type": "TEXT", "description": "A free note " * 6}
        for i in range(10)
    ]
    wide = {**ORDERS, "columns": [*ORDERS["columns"], *notes]}
    monkeypatch.setattr(tool, "fetch_tables_by_ids", lambda ids: [wide])
    monkeypatch.setenv("TEXT_TO_SQL_REACT_MAX_OBSERVATION_CHARS", "400")
    ctx = _ctx()
    text = tool.retrieve(ctx, "order amount")

    assert "TABLE sales.orders — One row per order (shortened to fit)" in text
    assert "  order_id INT\n  cust_ref INT — references crm.customers" in text
    assert "  amount REAL" in text
    assert "note_0" not in text
    assert "… 11 more columns not shown; retrieve sales.orders" in text
    assert ctx.was_shown("sales", "orders")
    assert "t-orders" not in ctx.seen_tables


def test_a_table_the_limit_cut_is_not_in_view(
    calls: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = _ctx()
    monkeypatch.setenv("TEXT_TO_SQL_REACT_MAX_OBSERVATION_CHARS", "60")
    tool.retrieve(ctx, "customer status")

    assert not ctx.was_shown("crm", "customers")


def test_a_bridge_on_a_join_route_is_in_view(
    calls: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    hop = {"source_column": "id", "target_column": "id"}
    monkeypatch.setattr(
        tool,
        "find_join_routes",
        lambda _a, _d: [
            {
                "hops": [
                    {**hop, "source_schema": "sales", "source_table": "orders"}
                    | {"target_schema": "sales", "target_table": "order_links"},
                    {**hop, "source_schema": "sales", "source_table": "order_links"}
                    | {"target_schema": "crm", "target_table": "customers"},
                ],
                "default": True,
            }
        ],
    )
    ctx = _ctx()
    tool.retrieve(ctx, "how is revenue calculated")
    tool.retrieve(ctx, "customer status")

    assert ctx.was_shown("sales", "order_links")


def _hop(left: tuple[str, str], right: tuple[str, str]) -> dict:
    return {
        "source_schema": left[0],
        "source_table": left[1],
        "source_column": "id",
        "target_schema": right[0],
        "target_table": right[1],
        "target_column": "id",
    }


ORDERS_T, CUSTOMERS_T = ("sales", "orders"), ("crm", "customers")
VIA_STAFF = [_hop(ORDERS_T, ("hr", "staff")), _hop(("hr", "staff"), CUSTOMERS_T)]
VIA_STORE = [_hop(ORDERS_T, ("ops", "store")), _hop(("ops", "store"), CUSTOMERS_T)]


@pytest.fixture
def ranked(calls: dict, monkeypatch: pytest.MonkeyPatch) -> dict[str, list[Any]]:
    """Ask for three routes between tables two hops apart; stub the ranked lookup
    with two."""
    seen: dict[str, list[Any]] = {"ranked": []}

    def find_join_paths(
        anchor: str, dest: str, *, k: int, boost_table_ids: set[str]
    ) -> list[dict]:
        seen["ranked"].append((anchor, dest, k, set(boost_table_ids)))
        return [{"path": VIA_STAFF}, {"path": VIA_STORE}]

    monkeypatch.setattr(tool.config, "join_routes", lambda: 3)
    monkeypatch.setattr(
        tool, "find_join_routes", lambda _a, _d: [{"hops": VIA_STAFF, "default": True}]
    )
    monkeypatch.setattr(tool, "find_join_paths", find_join_paths)
    return seen


def test_one_route_by_default_never_ranks(
    calls: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TEXT_TO_SQL_REACT_JOIN_ROUTES", raising=False)
    monkeypatch.setattr(
        tool, "find_join_paths", lambda *a, **k: pytest.fail("ranked lookup")
    )
    ctx = _ctx()
    tool.retrieve(ctx, "how is revenue calculated")
    text = tool.retrieve(ctx, "customer status")

    assert "routes, best ranked first" not in text
    assert len(calls["join"]) == 1


def test_directly_joined_tables_keep_their_direct_pairs_unranked(
    calls: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tool.config, "join_routes", lambda: 3)
    monkeypatch.setattr(
        tool, "find_join_paths", lambda *a, **k: pytest.fail("ranked lookup")
    )
    ctx = _ctx()
    tool.retrieve(ctx, "how is revenue calculated")
    text = tool.retrieve(ctx, "customer status")

    assert "routes, best ranked first" not in text
    assert len(calls["join"]) == 1


def test_tied_routes_are_offered_best_first(ranked: dict) -> None:
    ctx = _ctx()
    tool.retrieve(ctx, "how is revenue calculated")
    text = tool.retrieve(ctx, "customer status")

    head = "JOIN crm.customers ↔ sales.orders: 2 routes, best ranked first."
    assert head in text
    staff = text.index("  1. sales.orders.id = hr.staff.id")
    store = text.index("  2. sales.orders.id = ops.store.id")
    assert text.index(head) < staff < store


def test_every_offered_route_is_in_view_but_only_the_best_is_linked(
    ranked: dict,
) -> None:
    ctx = _ctx()
    tool.retrieve(ctx, "how is revenue calculated")
    tool.retrieve(ctx, "customer status")

    assert ctx.was_shown("hr", "staff")
    assert ctx.was_shown("ops", "store")
    assert ctx.join_paths == [{"path": VIA_STAFF}]


def test_tables_the_model_searched_for_boost_the_ranking(ranked: dict) -> None:
    ctx = _ctx()
    tool.retrieve(ctx, "how is revenue calculated")
    tool.retrieve(ctx, "customer status")

    [(_anchor, _dest, k, boost)] = ranked["ranked"]
    assert k == 3
    assert boost == {"t-orders", "t-customers"}


def test_a_single_ranked_route_reads_like_the_default(
    ranked: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tool, "find_join_paths", lambda *a, **k: [{"path": VIA_STAFF}])
    ctx = _ctx()
    tool.retrieve(ctx, "how is revenue calculated")
    text = tool.retrieve(ctx, "customer status")

    assert (
        "JOIN crm.customers ↔ sales.orders: "
        "sales.orders.id = hr.staff.id → hr.staff.id = crm.customers.id"
    ) in text


def test_the_best_loaded_route_is_linked_when_the_top_one_is_not(
    ranked: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    direct = [_hop(ORDERS_T, CUSTOMERS_T)]
    monkeypatch.setattr(
        tool, "find_join_paths", lambda *a, **k: [{"path": VIA_STAFF}, {"path": direct}]
    )
    ctx = _ctx()
    tool.retrieve(ctx, "how is revenue calculated")
    text = tool.retrieve(ctx, "customer status")

    assert "  1. sales.orders.id = hr.staff.id" in text
    assert ctx.join_paths == [{"path": direct}]


@pytest.mark.parametrize(("raw", "routes"), [("", 1), ("3", 3), ("0", 1), ("9", 5)])
def test_join_routes_setting_is_bounded(
    monkeypatch: pytest.MonkeyPatch, raw: str, routes: int
) -> None:
    monkeypatch.setenv("TEXT_TO_SQL_REACT_JOIN_ROUTES", raw)

    assert tool.config.join_routes() == routes


def test_a_table_only_a_definition_names_is_in_view(
    calls: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    refunds = {**REVENUE, "sql": "SELECT SUM(r.amount) FROM finance.refunds r"}
    monkeypatch.setattr(
        tool, "_get_candidates_information", lambda *a, **k: [dict(refunds)]
    )
    ctx = _ctx()
    tool.retrieve(ctx, "refunds")

    assert ctx.was_shown("finance", "refunds")
    assert ctx.was_shown("sales", "orders")


def test_a_quoted_table_name_ending_in_punctuation_is_in_view(
    calls: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    quoted = {**REVENUE, "sql": 'SELECT * FROM finance."refunds$"'}
    monkeypatch.setattr(
        tool, "_get_candidates_information", lambda *a, **k: [dict(quoted)]
    )
    ctx = _ctx()
    tool.retrieve(ctx, "refunds")

    assert ctx.was_shown("finance", "refunds$")
    assert not ctx.was_shown("finance", "refunds")


def test_a_table_named_only_past_the_clipped_sql_is_not_in_view(
    calls: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    padding = ", ".join(f"c{i}" for i in range(300))
    long_sql = (
        f"SELECT {padding} FROM finance.refunds r "
        "JOIN finance.hidden_ledger h ON h.id = r.id"
    )
    assert long_sql.index("hidden_ledger") > 500
    clipped = {**REVENUE, "sql": long_sql}
    monkeypatch.setattr(
        tool, "_get_candidates_information", lambda *a, **k: [dict(clipped)]
    )
    ctx = _ctx()

    text = tool.retrieve(ctx, "refunds")

    assert "hidden_ledger" not in text
    assert not ctx.was_shown("finance", "hidden_ledger")


ORDERS_ARCHIVE = {
    "id": "t-orders-archive",
    "name": "orders_archive",
    "schema_name": "sales",
    "database_name": "shop",
    "columns": ["order_id", "cust_ref", "amount", "discount", "archived_at"],
}


class CountingConnector:
    database_name = "shop"
    dialect = "mysql"
    supports_statement_timeout = True

    def __init__(self, counts: dict[str, int]) -> None:
        self.counts = counts
        self.sql: list[str] = []
        self.timeouts: list[float | None] = []

    def qualify(self, schema: str | None, table: str) -> str:
        return f"cat.`{schema}`.`{table}`"

    def execute(self, sql: str, timeout_s: float | None = None) -> pd.DataFrame:
        self.sql.append(sql)
        self.timeouts.append(timeout_s)
        table = sql.rsplit(".", 1)[-1].strip("`")
        return pd.DataFrame([[self.counts[table]]])


@pytest.fixture
def with_siblings(calls: dict, monkeypatch: pytest.MonkeyPatch) -> CountingConnector:
    catalog = [
        {**ORDERS, "columns": [c["name"] for c in ORDERS["columns"]]},
        ORDERS_ARCHIVE,
        {**CUSTOMERS, "columns": ["customer_id", "status"]},
        {**ORDERS_ARCHIVE, "id": "t-elsewhere", "database_name": "other"},
    ]
    monkeypatch.setattr(siblings, "fetch_table_column_names", lambda _db: catalog)
    return CountingConnector({"orders": 120, "orders_archive": 950})


def test_a_near_copy_is_named_with_rows_and_differences(
    with_siblings: CountingConnector,
) -> None:
    ctx = ToolContext(
        retriever=object(), connectors=[with_siblings], database_name="shop"
    )
    text = tool.retrieve(ctx, "how is revenue calculated")

    assert (
        "SIMILAR sales.orders (120 rows) ~ sales.orders_archive (950 rows): "
        "4 shared columns; sales.orders has no columns of its own; "
        "only sales.orders_archive has archived_at"
    ) in text
    assert "customers (" not in text
    assert text.count("SIMILAR") == 1
    assert ctx.was_shown("sales", "orders_archive")


def test_a_near_copy_is_named_once_per_episode(
    with_siblings: CountingConnector,
) -> None:
    ctx = ToolContext(
        retriever=object(), connectors=[with_siblings], database_name="shop"
    )
    tool.retrieve(ctx, "how is revenue calculated")
    ctx.seen_tables.clear()
    again = tool.retrieve(ctx, "how is revenue calculated")

    assert "SIMILAR" not in again
    assert len(with_siblings.sql) == 2


def test_a_count_is_qualified_by_the_connector_and_capped_in_time(
    with_siblings: CountingConnector, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEXT_TO_SQL_REACT_SIBLING_COUNT_TIMEOUT_S", "3")
    ctx = ToolContext(
        retriever=object(), connectors=[with_siblings], database_name="shop"
    )
    tool.retrieve(ctx, "how is revenue calculated")

    assert with_siblings.sql == [
        "SELECT COUNT(*) FROM cat.`sales`.`orders`",
        "SELECT COUNT(*) FROM cat.`sales`.`orders_archive`",
    ]
    assert with_siblings.timeouts == [3, 3]


def test_a_table_the_connector_cannot_qualify_is_named_without_a_count(
    with_siblings: CountingConnector, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(schema: str | None, table: str) -> str:
        raise ValueError(f"Trino requires a schema to qualify {table!r}")

    monkeypatch.setattr(with_siblings, "qualify", refuse)
    ctx = ToolContext(
        retriever=object(), connectors=[with_siblings], database_name="shop"
    )
    text = tool.retrieve(ctx, "how is revenue calculated")

    assert "SIMILAR sales.orders ~ sales.orders_archive: 4 shared columns" in text
    assert with_siblings.sql == []


def test_counts_past_the_limit_are_left_off_and_not_cached(
    with_siblings: CountingConnector, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEXT_TO_SQL_REACT_MAX_SIBLING_COUNTS", "1")
    ctx = ToolContext(
        retriever=object(), connectors=[with_siblings], database_name="shop"
    )
    text = tool.retrieve(ctx, "how is revenue calculated")

    assert "SIMILAR sales.orders (120 rows) ~ sales.orders_archive: " in text
    assert len(with_siblings.sql) == 1
    assert "t-orders-archive" not in ctx.row_counts


def test_without_a_database_the_line_has_no_counts(with_siblings: None) -> None:
    text = tool.retrieve(_ctx(), "how is revenue calculated")

    assert "SIMILAR sales.orders ~ sales.orders_archive: 4 shared columns" in text


def test_siblings_can_be_turned_off(
    with_siblings: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEXT_TO_SQL_REACT_SIBLINGS", "false")

    assert "SIMILAR" not in tool.retrieve(_ctx(), "how is revenue calculated")


def test_tables_sharing_only_a_few_columns_are_not_siblings() -> None:
    table = {"id": "a", "columns": ["id", "name", "x1", "x2", "x3", "x4"]}
    other = {"id": "b", "columns": ["id", "name", "y1", "y2", "y3", "y4"]}

    assert siblings.find_siblings(table, [other]) == []


def test_a_lookup_copied_into_a_wide_table_is_not_a_sibling() -> None:
    lookup = {"id": "a", "columns": ["dept_code", "dept_name", "school", "head"]}
    wide = {
        "id": "b",
        "columns": ["dept_code", "dept_name", "school", "head"]
        + [f"measure_{i}" for i in range(10)],
    }

    assert siblings.find_siblings(lookup, [wide]) == []


def test_a_concept_the_limit_dropped_is_shown_by_a_later_search(
    calls: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    members = [
        {"table_id": "t-orders", "schema_name": "sales", "table_name": "orders"},
        {"table_id": "t-customers", "schema_name": "crm", "table_name": "customers"},
    ]
    concept = {
        "concept_id": "concept-1",
        "concept_name": "Customer",
        "members": members,
        "keys": [],
    }
    monkeypatch.setattr(
        tool,
        "fetch_concept_members",
        lambda ids: {i: concept for i in ids if i in {"t-orders", "t-customers"}},
    )
    limit = {"chars": 10}
    monkeypatch.setattr(tool.config, "max_observation_chars", lambda: limit["chars"])
    ctx = _ctx()

    first = tool.retrieve(ctx, "how is revenue calculated")
    assert "CONCEPT Customer" not in first
    assert "concept-1" not in ctx.shown_concepts

    limit["chars"] = 100_000
    second = tool.retrieve(ctx, "customer status")
    assert "CONCEPT Customer: sales.orders + crm.customers" in second
    assert "concept-1" in ctx.shown_concepts
