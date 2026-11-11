# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the pure helpers in auto_ontology.semantic.embed."""

from __future__ import annotations

from auto_ontology.semantic.embed import _build_rows, _sample_values


def test_sample_values_handles_legacy_json_string_and_native_list() -> None:
    # Legacy Column nodes still store sample_values as a JSON string.
    assert _sample_values('["a", "b"]') == ["a", "b"]
    # Current writers store a native list.
    assert _sample_values(["a", "b"]) == ["a", "b"]


def test_sample_values_filters_long_values_and_handles_empty() -> None:
    long_value = "x" * 31
    assert _sample_values([long_value, "ok"]) == ["ok"]
    assert _sample_values(None) == []
    assert _sample_values([]) == []
    assert _sample_values(["a", None, "b"]) == ["a", "b"]
    assert _sample_values('["a", null, "b"]') == ["a", "b"]


TERM = {"name": "Purchase Order", "description": "An order placed", "synonyms": ["PO"]}
STATUS = {
    "id": "a-status",
    "name": "Order Status",
    "term_name": "Purchase Order",
    "source_column": "ord_st",
    "column_name": "ord_st",
    "description": "Where the order is — samples: OPEN, SHIPPED",
    "column_description": "Where the order is.",
    "column_data_type": "varchar",
    "sample_values": ["OPEN", "SHIPPED"],
    "is_unique": False,
    "table_name": "orders",
    "schema_name": "sales",
}


def _texts(term: dict, attrs: list[dict]) -> list[str]:
    return [row["text"] for row in _build_rows("shop", term, attrs)]


def test_term_text_is_shaped_like_its_tables() -> None:
    attrs = [
        {**STATUS, "table_description": "One row per order", "database_name": "shop"}
    ]

    assert _texts(TERM, attrs)[0] == (
        "term: Purchase Order, synonyms: PO, term_description: An order placed, "
        "database_name: shop, schema_name: sales, table_name: orders, "
        "table_description: One row per order, attributes: {name: Order Status, "
        "column: ord_st, data_type: varchar, description: Where the order is}"
    )


def test_a_term_over_several_tables_names_them_all() -> None:
    archived = {**STATUS, "id": "a-2", "table_name": "orders_archive"}
    text = _texts(TERM, [STATUS, archived])[0]

    assert "tables: {table: sales.orders},{table: sales.orders_archive}" in text
    assert "{name: Order Status, table: sales.orders_archive, column: ord_st" in text


def test_a_term_without_attributes_still_has_its_own_fields() -> None:
    assert _texts(TERM, [])[0] == (
        "term: Purchase Order, synonyms: PO, term_description: An order placed, "
        "database_name: shop"
    )


def test_attribute_text_says_what_column_it_represents() -> None:
    text = _texts(TERM, [STATUS])[1]

    assert text == (
        "Order Status is an attribute of Purchase Order (also: PO). It represents "
        "column ord_st of table sales.orders in database shop. Description: Where "
        "the order is. data type: varchar, unique: no, sample values: OPEN, SHIPPED."
    )


def test_samples_and_format_appear_once() -> None:
    dated = {
        **STATUS,
        "description": "Day the order was placed — format: YYMMDD",
        "column_description": None,
        "sample_values": None,
        "column_data_type": "text",
    }
    text = _texts(TERM, [STATUS, dated])

    assert text[1].count("OPEN") == 1
    assert text[2].count("YYMMDD") == 1
    assert "Description: Day the order was placed." in text[2]


def test_a_different_column_description_is_kept_too() -> None:
    attr = {**STATUS, "column_description": "Lifecycle code"}

    assert "Column description: Lifecycle code." in _texts({}, [attr])[0]


def test_synonyms_come_from_the_attribute_when_no_term_is_passed() -> None:
    attr = {**STATUS, "term_synonyms": ["PO"]}

    assert _texts({}, [attr])[0].startswith(
        "Order Status is an attribute of Purchase Order (also: PO)."
    )
