# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the pure helpers in auto_ontology.semantic.embed."""

from __future__ import annotations

from auto_ontology.semantic.embed import (
    _build_relationship_rows,
    _format_sample_values,
)


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


def test_build_relationship_rows_carries_edge_endpoints_and_usage() -> None:
    rows = _build_relationship_rows(
        "card_games",
        [
            {
                "source_column_id": "cards-set-code",
                "source_table_id": "cards",
                "source_table_name": "cards",
                "target_attribute_id": "set-code-attribute",
                "target_column_id": "sets-code",
                "target_table_id": "sets",
                "target_table_name": "sets",
                "role": "card belongs to card set",
                "how_to_use": "Traverse from a card to its owning set.",
                "when_to_use": ["a card question needs set properties"],
                "avoid_when": ["the question asks only for the card name"],
            }
        ],
    )

    assert len(rows) == 1
    assert rows[0]["metadata"]["label"] == "SemanticRelationship"
    assert rows[0]["metadata"]["source_table_id"] == "cards"
    assert rows[0]["metadata"]["target_table_id"] == "sets"
    assert "card belongs to card set" in rows[0]["text"]
