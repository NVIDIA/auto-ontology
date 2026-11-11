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
    seen: dict[str, list[Any]] = {"meaning": [], "names": [], "join": []}
    meaning_hits: dict[str, list[dict]] = {
        "how is revenue calculated": [REVENUE],
        "customer status": [STATUS_ATTRIBUTE],
        "order amount": [AMOUNT_ATTRIBUTE, REVENUE, AMOUNT_ATTRIBUTE],
        "customers": [CUSTOMER_TERM],
    }

    def meaning(_retriever, query, list_of_semantic, database_name, per_label_k):
        seen["meaning"].append((query, list_of_semantic, database_name))
        return [dict(h) for h in meaning_hits.get(query, [])]

    def names(tokens, object_types, *, include_description, limit):
        seen["names"].append((tokens, include_description, object_types))
        if tokens == ["cust_ref"] and "Column" in object_types:
            return [{"id": "c-cust-ref", "label": "Column", "name": "cust_ref"}]
        return []

    def term_tables(*, term_ids, database_name):
        return [
            {"term_id": "term-customer", "table_id": "t-orders", "path": "SEMANTIC_FK"},
            {
                "term_id": "term-customer",
                "table_id": "t-customers",
                "path": "REPRESENTS",
            },
        ]

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


def test_identifier_words_are_also_searched_by_physical_name(calls: dict) -> None:
    text = tool.retrieve(_ctx(), "which orders have cust_ref set")

    by_name = [types for tokens, _, types in calls["names"] if tokens == ["cust_ref"]]
    assert by_name and {"Table", "Column"} <= by_name[0]
    assert "TABLE sales.orders" in text


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
    assert "status TEXT — values: 'ACTIVE', 'CHURNED', 'PENDING'" in text
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


def test_kinds_narrow_the_search(calls: dict) -> None:
    tool.retrieve(_ctx(), "how is revenue calculated", kinds=["attribute"])
    tool.retrieve(_ctx(), "how is revenue calculated", kinds=["term", "definition"])

    assert calls["meaning"][0][1] == ["ColumnAttribute"]
    assert calls["meaning"][0][2] == "shop"
    assert calls["meaning"][1][1] == ["Term", "SqlAttribute"]


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


@pytest.fixture
def physical(calls: dict, monkeypatch: pytest.MonkeyPatch) -> dict[str, list]:
    """Two retrievers: the semantic one finds nothing, the data one finds a
    covered column, an uncovered foreign key and a covered table."""
    data_retriever = object()
    seen: dict[str, list] = {"coverage": []}

    def meaning(retriever, query, list_of_semantic, database_name, per_label_k):
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

    def coverage(column_ids, table_ids):
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


def test_terms_and_attributes_are_shown_in_brackets(
    calls: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    named = {
        **ORDERS,
        "terms": ["Purchase Order"],
        "columns": [
            {"name": "amount", "data_type": "REAL", "attribute": "Order Amount"},
            *ORDERS["columns"][1:2],
        ],
    }
    monkeypatch.setattr(tool, "fetch_tables_by_ids", lambda ids: [named])
    text = tool.retrieve(_ctx(), "order amount")

    assert "TABLE sales.orders [Purchase Order] — One row per order" in text
    assert "  amount REAL [Order Amount]" in text
    assert "  cust_ref INT — Who placed the order; references" in text


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
    assert "status TEXT — values: 'ACTIVE', 'CHURNED', 'PENDING'" in again


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
        "find_join_path",
        lambda _a, _d: [
            {**hop, "source_schema": "sales", "source_table": "orders"}
            | {"target_schema": "sales", "target_table": "order_links"},
            {**hop, "source_schema": "sales", "source_table": "order_links"}
            | {"target_schema": "crm", "target_table": "customers"},
        ],
    )
    ctx = _ctx()
    tool.retrieve(ctx, "how is revenue calculated")
    tool.retrieve(ctx, "customer status")

    assert ctx.was_shown("sales", "order_links")


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

    def __init__(self, counts: dict[str, int]):
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
