# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the pure helpers in auto_ontology.semantic.embed."""

from __future__ import annotations

from auto_ontology.semantic.embed import _build_rows, _format_sample_values


def test_format_sample_values_handles_legacy_json_string_and_native_list() -> None:
    # Legacy Column nodes still store sample_values as a JSON string.
    assert _format_sample_values('["a", "b"]') == " Sample values: a, b."
    # Current writers store a native list.
    assert _format_sample_values(["a", "b"]) == " Sample values: a, b."


def test_format_sample_values_filters_long_values_and_handles_empty() -> None:
    long_value = "x" * 31
    assert _format_sample_values([long_value, "ok"]) == " Sample values: ok."
    assert _format_sample_values(None) == ""
    assert _format_sample_values([]) == ""
    assert _format_sample_values(["a", None, "b"]) == " Sample values: a, b."
    assert _format_sample_values('["a", null, "b"]') == " Sample values: a, b."


def _attr_text(**attr: object) -> str:
    base = {"id": "a1", "name": "Total", "term_name": "Order", "source_column": "t"}
    (row,) = _build_rows("shop", {}, [{**base, **attr}])
    return row["text"]


def test_the_columns_description_is_embedded_not_the_attributes() -> None:
    text = _attr_text(
        description="The long attribute description",
        column_description="Order total",
        sample_values="[10, 20]",
    )

    assert (
        text
        == "ColumnAttribute: Total of Term Order. Order total Sample values: 10, 20."
    )


def test_the_family_is_named_after_the_term() -> None:
    (row,) = _build_rows(
        "shop",
        {"synonyms": ["Purchase"]},
        [
            {
                "id": "a1",
                "name": "Total",
                "term_name": "Order",
                "column_description": "Order total",
                "family_prefix": "NEW_",
            }
        ],
    )

    assert row["text"] == (
        "ColumnAttribute: Total of Term Order (Purchase), family NEW_. Order total"
    )


def test_an_undocumented_column_embeds_its_name_and_samples() -> None:
    assert _attr_text(sample_values='["a"]') == (
        "ColumnAttribute: Total of Term Order.  Sample values: a."
    )
