# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest

from auto_ontology.retrieval.text_to_sql.react import information as info_module
from auto_ontology.retrieval.text_to_sql.react.context import ToolContext
from auto_ontology.retrieval.text_to_sql.react.information import information

TABLES = [
    {
        "id": "tab-orders",
        "name": "orders",
        "schema_name": "shop",
        "database_name": "shop",
        "description": "Each purchase.",
    },
    {
        "id": "tab-customers",
        "name": "customers",
        "schema_name": "shop",
        "database_name": "shop",
        "description": "Who buys.",
    },
]
TERMS = [
    {"id": "t1", "name": "Order", "description": "A purchase.", "synonyms": ["Sale"]},
    {"id": "t2", "name": "Customer", "description": "A buyer.", "synonyms": []},
    {"id": "t3", "name": "Patient", "description": "Elsewhere.", "synonyms": []},
]
PAIRS = [
    {"term_id": "t1", "table_id": "tab-orders"},
    {"term_id": "t2", "table_id": "tab-customers"},
]


@pytest.fixture
def ctx(monkeypatch: pytest.MonkeyPatch) -> ToolContext:
    def setattr(name: str, value: Any) -> None:
        monkeypatch.setattr(info_module, name, value)

    setattr("fetch_schema_ids_for_database", lambda _: ["s1"])
    setattr("fetch_tables_for_schema", lambda _: TABLES)
    setattr("fetch_all_terms", lambda: TERMS)
    setattr(
        "fetch_term_table_pairs",
        lambda database_name=None, term_ids=None: [
            p for p in PAIRS if not term_ids or p["term_id"] in term_ids
        ],
    )
    setattr("fetch_tables_by_ids", lambda ids: [t for t in TABLES if t["id"] in ids])
    setattr(
        "fetch_column_attributes_by_term_id",
        lambda term_id: (
            [
                {
                    "name": "Order Amount",
                    "description": "What was paid.",
                    "table_id": "tab-orders",
                    "source_column": "amount",
                    "primary_column": {"column_name": "amount", "table_name": "orders"},
                }
            ]
            if term_id == "t1"
            else []
        ),
    )
    setattr(
        "fetch_sql_attributes_by_term_id",
        lambda term_id: (
            [
                {
                    "name": "Revenue",
                    "description": "Paid minus discount.",
                    "sql": "SUM(amount)",
                }
            ]
            if term_id == "t1"
            else []
        ),
    )
    setattr(
        "fetch_attr_column_contexts",
        lambda ids, database_name=None: {
            "a1": {
                "attr_name": "Customer Status",
                "attr_description": "Active or not.",
                "term_name": "Customer",
                "schema_name": "shop",
                "table_name": "customers",
                "database_name": "shop",
                "col_name": "status",
            },
            # What the real lookup returns for an attribute whose column is in
            # another database: names filled in, ``database_name`` empty.
            "a-other": {
                "attr_name": "Patient Status",
                "attr_description": "Elsewhere.",
                "term_name": "",
                "schema_name": "clinic",
                "table_name": "patients",
                "database_name": "",
                "col_name": "status",
            },
        },
    )
    for search in ("_uncovered_physical_hits", "_name_hits"):
        setattr(search, lambda *_: [])
    return ToolContext(retriever=None, connectors=[], database_name="shop")


def _found(monkeypatch: pytest.MonkeyPatch, hits: list[dict]) -> None:
    monkeypatch.setattr(info_module, "_meaning_hits", lambda *_: hits)


def test_an_empty_query_lists_every_table(ctx: ToolContext) -> None:
    assert information(ctx, "") == (
        "TABLE shop.customers — Who buys.\nTABLE shop.orders — Each purchase."
    )


def test_an_empty_query_on_terms_lists_this_databases_terms(ctx: ToolContext) -> None:
    assert information(ctx, "", ["term"]) == (
        "TERM Customer — A buyer.\nTERM Order — A purchase. (also: Sale)"
    )


def test_a_term_is_shown_with_its_tables_attributes_and_definitions(
    ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    _found(monkeypatch, [{"id": "t1", "label": "Term"}])

    assert information(ctx, "order") == "\n".join(
        [
            "TERM Order — A purchase. (also: Sale)",
            "  Stored in: shop.orders",
            "  Attributes:",
            "    Order Amount (shop.orders.amount) — What was paid.",
            "  Definitions:",
            "    Revenue — Paid minus discount.",
            "      SQL: SUM(amount)",
        ]
    )


def test_an_attribute_is_shown_with_its_column(
    ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    _found(monkeypatch, [{"id": "a1", "label": "ColumnAttribute"}])

    assert information(ctx, "status") == (
        "ATTRIBUTE Customer Status (of Customer) — column shop.customers.status: "
        "Active or not."
    )


def test_an_attribute_of_another_database_is_not_shown(
    ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    _found(monkeypatch, [{"id": "a-other", "label": "ColumnAttribute"}])

    assert information(ctx, "status").startswith("No matches for 'status'.")


def test_an_attribute_of_a_term_shown_in_full_is_not_repeated(
    ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    _found(
        monkeypatch,
        [{"id": "t2", "label": "Term"}, {"id": "a1", "label": "ColumnAttribute"}],
    )

    assert information(ctx, "customer") == (
        "TERM Customer — A buyer.\n  Stored in: shop.customers"
    )


def _column_hit(database_name: str) -> dict:
    return {
        "id": "col-customer-id",
        "label": "Column",
        "name": "customer_id",
        "relevant_tables": [
            {
                "id": "tab-orders",
                "name": "orders",
                "schema_name": "shop",
                "database_name": database_name,
                "columns": [
                    {"name": "amount", "data_type": "decimal", "description": None},
                    {"name": "customer_id", "data_type": "int", "description": None},
                ],
            }
        ],
    }


def test_an_undescribed_foreign_key_column_is_shown_with_its_table(
    ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        info_module, "_uncovered_physical_hits", lambda *_: [_column_hit("shop")]
    )
    _found(monkeypatch, [])

    assert information(ctx, "customer id") == "COLUMN shop.orders.customer_id (int)"


def test_a_column_of_another_database_is_not_shown(
    ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(info_module, "_name_hits", lambda *_: [_column_hit("other")])
    _found(monkeypatch, [])

    assert information(ctx, "customer_id").startswith("No matches for 'customer_id'.")


def test_a_term_of_another_database_is_not_shown(
    ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    _found(monkeypatch, [{"id": "t3", "label": "Term"}])

    assert information(ctx, "patient").startswith("No matches for 'patient'.")


def test_a_term_too_long_for_the_limit_is_cut_not_dropped(
    ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEXT_TO_SQL_REACT_MAX_OBSERVATION_CHARS", "200")
    monkeypatch.setattr(
        info_module,
        "fetch_column_attributes_by_term_id",
        lambda term_id: [
            {
                "name": f"Attribute {n:02}",
                "description": "d" * 20,
                "table_id": "tab-orders",
                "source_column": "amount",
                "primary_column": {"column_name": "amount", "table_name": "orders"},
            }
            for n in range(20)
        ],
    )
    _found(monkeypatch, [{"id": "t1", "label": "Term"}])

    out = information(ctx, "order")

    assert out.startswith("TERM Order — A purchase.")
    assert len(out) <= 200
    assert out.endswith("…")
    assert "not shown" not in out


def test_a_long_list_is_cut_and_counted(
    ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEXT_TO_SQL_REACT_MAX_OBSERVATION_CHARS", "200")
    many = [
        {"name": f"table_{n:03}", "schema_name": "shop", "description": "x" * 30}
        for n in range(50)
    ]
    monkeypatch.setattr(info_module, "fetch_tables_for_schema", lambda _: many)

    out = information(ctx, "")

    assert len(out) <= 200
    assert out.endswith("more tables not shown; search with words to narrow them.")
