# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the pure helpers in gsf.semantic.embed."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from gsf.semantic.embed import SemanticEmbedder, _format_sample_values


def test_format_sample_values_handles_legacy_json_string_and_native_list() -> None:
    # Legacy Column nodes still store sample_values as a JSON string.
    assert _format_sample_values('["a", "b"]') == " Sample values: a, b."
    # Current writers store a native list.
    assert _format_sample_values(["a", "b"]) == " Sample values: a, b."


def test_similar_term_search_is_scoped_to_the_compiling_database() -> None:
    embedder = object.__new__(SemanticEmbedder)
    embedder.database_name = "sales_catalog"
    retriever = MagicMock()
    hits = [
        {
            "name": "Customer",
            "text": "Term: Customer. A buyer.",
            "score": 0.2,
        }
    ]

    with (
        patch(
            "gsf.utils.retriever.get_semantic_objects_retriever",
            return_value=retriever,
        ),
        patch(
            "gsf.retrieval.data_access.semantic_search.search_semantic_index",
            return_value=hits,
        ) as search,
    ):
        candidates = embedder.search_similar_terms("Buyer", "A customer")

    assert candidates == [
        {
            "name": "Customer",
            "content": "Term: Customer. A buyer.",
            "score": 0.2,
        }
    ]
    assert search.call_args.kwargs["database_name"] == "sales_catalog"


def test_format_sample_values_filters_long_values_and_handles_empty() -> None:
    long_value = "x" * 31
    assert _format_sample_values([long_value, "ok"]) == " Sample values: ok."
    assert _format_sample_values(None) == ""
    assert _format_sample_values([]) == ""
    assert _format_sample_values(["a", None, "b"]) == " Sample values: a, b."
    assert _format_sample_values('["a", null, "b"]') == " Sample values: a, b."
