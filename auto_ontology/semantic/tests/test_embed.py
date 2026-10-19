# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the pure helpers in auto_ontology.semantic.embed."""

from __future__ import annotations

from unittest.mock import MagicMock

from auto_ontology.semantic.embed import SemanticEmbedder, _format_sample_values


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


def test_relationship_embedding_cleans_stale_rows_when_none_remain() -> None:
    vdb = MagicMock()
    embedder = SemanticEmbedder(
        database_name="chemistry",
        embed_params=MagicMock(),
        vdb=vdb,
    )

    assert embedder.embed_relationships([]) == 0
    vdb.delete_by_database_and_label.assert_called_once_with(
        "chemistry", "SemanticRelationship"
    )
