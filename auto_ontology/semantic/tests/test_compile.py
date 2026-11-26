# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the unified semantic compilation entry point."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from auto_ontology.semantic import compile as semantic_compile


def test_fallback_report_runs_after_unmatched_fk_can_create_a_term(
    caplog: pytest.LogCaptureFixture,
) -> None:
    events: list[str] = []

    def attach_unmatched(*_args: object, **_kwargs: object) -> int:
        events.append("unmatched")
        return 1

    def fetch_fallbacks(_database_name: str) -> list[str]:
        events.append("report")
        return ["Order Customer Map"]

    with (
        patch.object(semantic_compile, "load_domain_summary", return_value=MagicMock()),
        patch.object(
            semantic_compile, "build_semantic_embedder", return_value=MagicMock()
        ),
        patch.object(semantic_compile, "compile_semantic_layer", return_value=1),
        patch.object(semantic_compile, "is_cancelled", return_value=False),
        patch.object(semantic_compile, "resolve_term_families", return_value=0),
        patch.object(semantic_compile, "resolve_semantic_fks", return_value=0),
        patch.object(
            semantic_compile,
            "attach_unmatched_fk_columns",
            side_effect=attach_unmatched,
        ),
        patch(
            "auto_ontology.semantic.sql_attribute_suggester.suggest_sql_attributes",
            return_value=0,
        ),
        patch.object(
            semantic_compile, "build_bridge_tables_sql_attributes", return_value=0
        ),
        patch.object(semantic_compile, "describe_column_attributes", return_value=0),
        patch.object(
            semantic_compile,
            "fetch_fallback_term_names_for_database",
            side_effect=fetch_fallbacks,
        ),
        caplog.at_level("WARNING", logger="auto_ontology.semantic.compile"),
    ):
        assert semantic_compile._run_semantic_compilation("dbx") == 1

    assert events == ["unmatched", "report"]
    assert "1 placeholder Term(s)" in caplog.text
    assert "Order Customer Map" in caplog.text


def test_a_failed_fallback_count_does_not_fail_reporting(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with (
        patch.object(
            semantic_compile,
            "fetch_fallback_term_names_for_database",
            side_effect=RuntimeError("store down"),
        ),
        caplog.at_level("WARNING", logger="auto_ontology.semantic.compile"),
    ):
        semantic_compile._report_fallback_terms("dbx")

    assert "Could not count placeholder Terms" in caplog.text


def test_fallback_report_limits_the_names_on_one_log_line(
    caplog: pytest.LogCaptureFixture,
) -> None:
    fallback_terms = [f"Term {index:02d}" for index in range(25)]
    with (
        patch.object(
            semantic_compile,
            "fetch_fallback_term_names_for_database",
            return_value=fallback_terms,
        ),
        caplog.at_level("WARNING", logger="auto_ontology.semantic.compile"),
    ):
        semantic_compile._report_fallback_terms("dbx")

    assert "25 placeholder Term(s)" in caplog.text
    assert "Term 19" in caplog.text
    assert "and 5 more" in caplog.text
    assert "Term 20" not in caplog.text
