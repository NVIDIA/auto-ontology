# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the known_types fallback in find_jsonb_path_mismatches, and for
the datatype tags added to format_semantic_context.
"""

from gsf.retrieval.text_to_sql.db_probe.executor import ProbeExecutor
from gsf.retrieval.text_to_sql.db_probe.jsonb_path_check import (
    build_jsonb_path_repair_error,
    find_jsonb_path_mismatches,
)
from gsf.retrieval.text_to_sql.formatters_util import format_semantic_context


def _dead_executor() -> ProbeExecutor:
    """A ProbeExecutor with no connector — every .run() fails gracefully
    (ok=False), simulating "the live probe couldn't confirm anything" without
    needing a real database.
    """
    return ProbeExecutor(connector=None)


def test_wrong_type_flagged_when_known_types_says_not_json():
    """A ->> used on a column known (via known_types) to not be JSON gets
    flagged as wrong_type, instead of silently skipped as before.
    """
    sql = "SELECT notes->>'summary' FROM aliens"
    mismatches = find_jsonb_path_mismatches(
        _dead_executor(),
        "postgres",
        sql,
        known_types={("aliens", "notes"): "text"},
    )
    assert len(mismatches) == 1
    assert mismatches[0]["wrong_type"] == "text"
    assert mismatches[0]["table"] == "aliens"
    assert mismatches[0]["column"] == "notes"


def test_no_mismatch_when_known_types_absent():
    """Without known_types (or without an entry for this column), stay
    silent rather than guess — preserves prior behavior.
    """
    sql = "SELECT notes->>'summary' FROM aliens"
    mismatches = find_jsonb_path_mismatches(_dead_executor(), "postgres", sql)
    assert mismatches == []


def test_no_mismatch_when_known_type_is_jsonb():
    """A column known to actually be jsonb must never be flagged as
    wrong_type just because the probe itself failed (e.g. no connector) —
    that's a probe-availability issue, not evidence of a type mismatch.
    """
    sql = "SELECT notes->>'summary' FROM aliens"
    mismatches = find_jsonb_path_mismatches(
        _dead_executor(),
        "postgres",
        sql,
        known_types={("aliens", "notes"): "jsonb"},
    )
    assert mismatches == []


def test_repair_error_renders_wrong_type_distinctly_from_key_mismatch():
    key_mismatch = {
        "table": "aliens",
        "column": "notes",
        "container": None,
        "used_key": "summary",
        "available_keys": ["overview", "detail"],
    }
    type_mismatch = {
        "table": "aliens",
        "column": "age",
        "container": None,
        "used_key": "value",
        "wrong_type": "integer",
    }

    key_only = build_jsonb_path_repair_error([key_mismatch])
    assert "does not exist and returns NULL" in key_only
    assert "isn't JSONB-typed" not in key_only

    type_only = build_jsonb_path_repair_error([type_mismatch])
    assert "isn't JSONB-typed" in type_only
    assert "does not exist and returns NULL" not in type_only

    both = build_jsonb_path_repair_error([key_mismatch, type_mismatch])
    assert "does not exist and returns NULL" in both
    assert "isn't JSONB-typed" in both


def test_format_semantic_context_includes_datatype_tags():
    primary_attribute = {
        "schema_name": "",
        "table_name": "aliens",
        "col_name": "notes",
        "attr_name": "AlienNotes",
        "datatype": "jsonb",
    }
    attribute_join_paths = [
        {
            "attr_name": "SpeciesName",
            "col_name": "name",
            "schema_name": "",
            "table_name": "species",
            "datatype": "text",
            "path": [],
        }
    ]

    result = format_semantic_context(primary_attribute, attribute_join_paths)

    assert "[jsonb]" in result
    assert "[text]" in result


def test_format_semantic_context_omits_tag_when_datatype_unknown():
    """Attrs not yet re-ingested with datatype must render exactly as before
    — no stray tag/bracket artifacts.
    """
    primary_attribute = {
        "schema_name": "",
        "table_name": "aliens",
        "col_name": "notes",
        "attr_name": "AlienNotes",
    }

    result = format_semantic_context(primary_attribute, [])

    assert "[" not in result
