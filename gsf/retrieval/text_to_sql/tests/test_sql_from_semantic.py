# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the shared table/column prompt formatting in sql_from_semantic.py."""

from __future__ import annotations

from gsf.retrieval.text_to_sql.formatters_util import (
    format_important_columns_for_prompt,
    format_tables_for_prompt,
)
from gsf.retrieval.text_to_sql.prompts import create_sql_user_prompt


def test_prompt_renders_date_format_when_present() -> None:
    rendered = format_tables_for_prompt(
        [
            {
                "name": "matches",
                "database_name": "cricket",
                "schema_name": "public",
                "columns": [
                    {
                        "name": "Match_Date",
                        "data_type": "text",
                        "description": "Date the match was played.",
                        "format": "YYMMDD",
                    }
                ],
            }
        ]
    )
    assert "format: YYMMDD" in rendered


def test_prompt_omits_format_when_absent() -> None:
    rendered = format_tables_for_prompt(
        [
            {
                "name": "matches",
                "schema_name": "public",
                "columns": [
                    {"name": "team_name", "data_type": "text", "description": ""}
                ],
            }
        ]
    )
    assert "format:" not in rendered


def test_prompt_forbids_unused_joins() -> None:
    assert "menu of valid options" in create_sql_user_prompt
    assert "If removing a join would not change the answer, omit it" in (
        create_sql_user_prompt
    )
    assert "Never join a table solely because its path is listed" in (
        create_sql_user_prompt
    )


def test_prompt_preserves_requested_projection_shape() -> None:
    assert "never add IDs unless the user explicitly asks for them" in (
        create_sql_user_prompt
    )
    assert "project explicitly requested outputs from left to right" in (
        create_sql_user_prompt
    )


def test_important_columns_include_full_details_and_ignore_structural_paths() -> None:
    rendered = format_important_columns_for_prompt(
        {
            "attr_name": "Charter Number",
            "col_name": "CharterNum",
            "table_name": "schools",
            "schema_name": "main",
            "database_name": "california_schools",
            "datatype": "text",
        },
        [
            {
                "attr_name": "District Name",
                "col_name": "District",
                "table_name": "schools",
                "schema_name": "main",
                "database_name": "california_schools",
                "datatype": "text",
                "path": [],
            },
            {
                "attr_name": "Removed Table Value",
                "col_name": "value",
                "table_name": "filtered_out",
                "schema_name": "main",
                "database_name": "california_schools",
                "datatype": "text",
                "path": [],
            },
            {"path": [{"source_table": "schools", "target_table": "districts"}]},
        ],
        [
            {
                "name": "schools",
                "schema_name": "main",
                "database_name": "california_schools",
                "columns": [
                    {
                        "name": "CharterNum",
                        "data_type": "text",
                        "description": "Four-character charter number.",
                        "sample_values": ["0040", "0728"],
                        "format": "NNNN",
                    },
                    {
                        "name": "District",
                        "data_type": "text",
                        "description": "District overseeing the school.",
                    },
                ],
            }
        ],
    )

    assert "Semantic match: Charter Number" in rendered
    assert "california_schools.main.schools" in rendered
    assert "Four-character charter number." in rendered
    assert "sample values: 0040, 0728" in rendered
    assert "format: NNNN" in rendered
    assert "Semantic match: District Name" in rendered
    assert "Removed Table Value" not in rendered
    assert "structural bridge" not in rendered
