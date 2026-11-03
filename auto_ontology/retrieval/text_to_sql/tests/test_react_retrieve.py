# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest

from auto_ontology.retrieval.text_to_sql.react import retrieve as tool
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
STATUS_COLUMN = {
    "id": "c-status",
    "label": "Column",
    "name": "status",
    "relevant_tables": [{"id": "t-customers"}],
}


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[Any]]:
    """Stub every data-access call the tool makes, recording what it asked."""
    seen: dict[str, list[Any]] = {"meaning": [], "names": [], "join": []}
    meaning_hits: dict[str, list[dict]] = {
        "how is revenue calculated": [REVENUE],
        "customer status": [STATUS_COLUMN],
    }

    def meaning(_retriever, query, list_of_semantic, database_name, per_label_k):
        seen["meaning"].append((query, list_of_semantic, database_name))
        return [dict(h) for h in meaning_hits.get(query, [])]

    def names(tokens, object_types, *, include_description, limit):
        seen["names"].append((tokens, include_description))
        if tokens == ["cust_ref"]:
            return [{"id": "c-cust-ref", "label": "Column", "name": "cust_ref"}]
        return []

    def expand(ids_and_labels):
        return {
            "c-cust-ref": {"name": "cust_ref", "relevant_tables": [{"id": "t-orders"}]}
        }

    def join_path(anchor, dest):
        seen["join"].append((anchor, dest))
        return [
            {
                "source_schema": "sales",
                "source_table": "orders",
                "source_column": "cust_ref",
                "target_schema": "crm",
                "target_table": "customers",
                "target_column": "customer_id",
            }
        ]

    monkeypatch.setattr(tool, "_get_candidates_information", meaning)
    monkeypatch.setattr(tool, "fetch_global_search", names)
    monkeypatch.setattr(tool, "expand_info", expand)
    monkeypatch.setattr(
        tool, "fetch_tables_by_ids", lambda ids: [TABLES[i] for i in ids if i in TABLES]
    )
    monkeypatch.setattr(
        tool,
        "find_column_id_by_table_and_name",
        lambda table, column, database: f"{table}.{column}",
    )
    monkeypatch.setattr(tool, "find_join_path", join_path)
    monkeypatch.setattr(tool, "fetch_attr_column_contexts", lambda ids, **_: {})
    monkeypatch.setattr(tool, "fetch_term_table_pairs", lambda **_: [])
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
    assert "status TEXT — values: 'ACTIVE', 'CHURNED', 'PENDING'" in text
    assert (
        "JOIN crm.customers ↔ sales.orders: "
        "sales.orders.cust_ref = crm.customers.customer_id"
    ) in text
    assert calls["join"] == [("customers.customer_id", "orders.order_id")]


def test_a_table_already_shown_is_not_repeated_in_full(calls: dict) -> None:
    ctx = _ctx()
    tool.retrieve(ctx, "customer status")
    text = tool.retrieve(ctx, "customer status")

    assert text == (
        "TABLE crm.customers (shown earlier; matched columns only)\n"
        "  status TEXT — values: 'ACTIVE', 'CHURNED', 'PENDING'"
    )


def test_identifier_words_are_also_searched_by_name(calls: dict) -> None:
    text = tool.retrieve(_ctx(), "which orders have cust_ref set")

    assert (["cust_ref"], False) in calls["names"]
    assert "TABLE sales.orders" in text


def test_tables_in_another_database_are_left_out(
    calls: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        tool,
        "_get_candidates_information",
        lambda *a, **k: [
            {
                "id": "c-x",
                "label": "Column",
                "name": "status",
                "relevant_tables": [{"id": "t-other"}],
            }
        ],
    )
    text = tool.retrieve(_ctx(), "status")

    assert text.startswith("No matches for 'status'")


def test_kinds_narrow_the_search(calls: dict) -> None:
    tool.retrieve(_ctx(), "how is revenue calculated", kinds=["attribute"])

    assert calls["meaning"][0][1] == ["ColumnAttribute", "SqlAttribute"]
    assert calls["meaning"][0][2] == "shop"


def test_unknown_kinds_search_everything() -> None:
    assert len(tool.resolve_kinds(["nonsense"])) == len(tool.resolve_kinds(None))


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

    assert len(text) < 300
    assert "more results not shown" in text


def test_empty_query_is_refused(calls: dict) -> None:
    assert tool.retrieve(_ctx(), "  ").startswith("Error: query is empty")


AMOUNT_ATTRIBUTE = {
    "id": "a-amount",
    "label": "ColumnAttribute",
    "name": "Order Amount",
}


@pytest.fixture
def attribute_hits(calls: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """The search finds the orders table and, twice, an attribute on it."""
    monkeypatch.setattr(
        tool,
        "_get_candidates_information",
        lambda *a, **k: [dict(AMOUNT_ATTRIBUTE), dict(REVENUE), dict(AMOUNT_ATTRIBUTE)],
    )
    monkeypatch.setattr(
        tool,
        "fetch_attr_column_contexts",
        lambda ids, **_: {
            "a-amount": {
                "col_name": "amount",
                "table_name": "orders",
                "schema_name": "sales",
                "table_id": "t-orders",
            }
        },
    )


def test_attribute_line_names_its_table(attribute_hits: None) -> None:
    text = tool.retrieve(_ctx(), "order amount")

    assert "ATTRIBUTE Order Amount\n  stored in sales.orders.amount" in text


def test_a_repeated_line_is_shown_once(attribute_hits: None) -> None:
    text = tool.retrieve(_ctx(), "order amount")

    assert text.count("ATTRIBUTE Order Amount") == 1


def test_tables_come_before_definitions_and_short_lines(attribute_hits: None) -> None:
    text = tool.retrieve(_ctx(), "order amount")

    assert (
        text.index("TABLE sales.orders")
        < text.index("ATTRIBUTE Revenue")
        < text.index("ATTRIBUTE Order Amount")
    )


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
    assert "status TEXT — values: 'ACTIVE', 'CHURNED', 'PENDING'" in again
