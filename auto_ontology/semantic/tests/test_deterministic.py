# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for deterministic column mapping."""

from __future__ import annotations

import pytest

from unittest.mock import patch


from auto_ontology.semantic.deterministic import (
    column_attribute_specs,
    fill_missing_column_descriptions,
    fk_source_columns,
    fk_target_table_names,
    lacks_description,
    to_term_name,
)

_MODULE = "auto_ontology.semantic.deterministic"


def test_excludes_suggested_fk_columns() -> None:
    columns = [
        {"name": "id", "data_type": "integer"},
        {"name": "vendor_id", "data_type": "integer"},
        {"name": "amount", "data_type": "numeric"},
    ]
    specs = column_attribute_specs(
        columns,
        [],
        suggested_fk_columns={"vendor_id"},
    )
    assert {s.source_column for s in specs} == {"id", "amount"}


def test_excludes_fk_columns() -> None:
    columns = [
        {"name": "id", "data_type": "integer"},
        {"name": "customer_id", "data_type": "integer"},
        {"name": "amount", "data_type": "numeric"},
    ]
    fks = [{"source_column": "customer_id", "target_table": "customers"}]
    specs = column_attribute_specs(columns, fks)
    assert {s.source_column for s in specs} == {"id", "amount"}
    assert fk_source_columns(fks) == {"customer_id"}


def test_a_spec_carries_the_columns_own_description_only() -> None:
    columns = [
        {"name": "id", "data_type": "integer", "description": " Primary key. "},
        {"name": "status", "data_type": "text", "description": "STATUS"},
        {
            "name": "amount",
            "data_type": "numeric",
            "description": "Borrowed from the attribute",
            "own_description": None,
        },
    ]
    specs = {s.source_column: s for s in column_attribute_specs(columns, [])}
    assert specs["id"].column_description == "Primary key."
    assert specs["status"].column_description is None
    assert specs["amount"].column_description is None


@pytest.mark.parametrize(
    ("col", "lacks"),
    [
        ({"name": "amount"}, True),
        ({"name": "amount", "description": "  "}, True),
        ({"name": "order_id", "description": "ORDER_ID"}, True),
        ({"name": "amount", "description": "Total paid"}, False),
        ({"name": "amount", "description": "Total paid", "own_description": ""}, True),
    ],
)
def test_lacks_description(col: dict, lacks: bool) -> None:
    assert lacks_description(col) is lacks


def test_fill_stores_only_undocumented_columns_and_refreshes_them() -> None:
    columns = [
        {"id": "c1", "name": "id", "data_type": "integer", "description": "Key"},
        {"id": "c2", "name": "amount", "data_type": "numeric"},
        {"id": "c3", "name": "note", "data_type": "text", "description": "note"},
    ]
    with (
        patch(
            f"{_MODULE}._generate_column_descriptions",
            return_value={"amount": "Total paid.", "note": "Free-text remark."},
        ) as generate,
        patch(f"{_MODULE}.store_column_descriptions", return_value=["c2"]) as store,
        patch(f"{_MODULE}.refresh_catalog_embeddings") as refresh,
    ):
        written = fill_missing_column_descriptions("t1", columns)

    assert [c["name"] for c in generate.call_args.args[0]] == ["amount", "note"]
    store.assert_called_once_with(
        "t1", {"amount": "Total paid.", "note": "Free-text remark."}
    )
    # Only what the store accepted is reflected back onto the column dicts.
    assert written == ["amount"]
    assert columns[1]["description"] == "Total paid."
    assert columns[2]["description"] == "note"
    refresh.assert_called_once_with(["c2", "t1"])


def test_fill_without_undocumented_columns_calls_nothing() -> None:
    columns = [{"id": "c1", "name": "id", "description": "Key"}]
    with (
        patch(f"{_MODULE}._generate_column_descriptions", return_value={}),
        patch(f"{_MODULE}.store_column_descriptions") as store,
    ):
        assert fill_missing_column_descriptions("t1", columns) == []
    store.assert_not_called()


def test_to_term_name() -> None:
    assert to_term_name("purchase_orders") == "Purchase Orders"
    assert to_term_name("delta_lite_event") == "Delta Lite Event"
    assert to_term_name("gtl_imaging_event") == "Gtl Imaging Event"
    assert to_term_name("gtl_ui_event") == "Gtl Ui Event"
    assert to_term_name("GtlImagingEvent") == "Gtl Imaging Event"
    assert to_term_name("XMLHttpRequest") == "XML Http Request"


@pytest.mark.parametrize(
    ("table", "expected"),
    [
        ("고객", "고객"),
        ("주문_내역", "주문 내역"),
        ("TB_고객", "TB 고객"),
        ("顧客マスタ", "顧客マスタ"),
        ("клиенты", "Клиенты"),
        ("Société_Générale", "Société Générale"),
        # Decomposed accents (as macOS tooling often writes them) stay in the word.
        ("Socie\u0301te\u0301", "Société"),
        # Devanagari vowel signs and virama are marks, not separators.
        ("ग्राहक_सूची", "ग्राहक सूची"),
        # Nothing word-like left: the name itself, never "".
        ("___", "___"),
    ],
)
def test_to_term_name_keeps_words_in_any_script(table: str, expected: str) -> None:
    assert to_term_name(table) == expected


def test_fk_targets_deduped() -> None:
    fks = [
        {"target_table": "customers"},
        {"target_table": "customers"},
        {"target_table": "products"},
    ]
    assert fk_target_table_names(fks) == ["customers", "products"]
