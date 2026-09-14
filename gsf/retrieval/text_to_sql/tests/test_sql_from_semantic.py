# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the shared table/column prompt formatting in sql_from_semantic.py."""

from __future__ import annotations

from gsf.retrieval.text_to_sql.formatters_util import format_tables_for_prompt


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
