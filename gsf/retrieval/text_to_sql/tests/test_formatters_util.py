# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from gsf.retrieval.text_to_sql.formatters_util import (
    format_tables_for_prompt,
    qualify_table,
)


def test_collapses_duplicate_database_and_schema() -> None:
    """MySQL reports TABLE_SCHEMA as the database, so db.db.table is invalid SQL."""
    assert qualify_table("dw", "dw", "SIS_DEPARTMENT") == "dw.SIS_DEPARTMENT"


def test_collapse_ignores_identifier_case() -> None:
    assert qualify_table("analytics", "ANALYTICS", "orders") == "ANALYTICS.orders"


def test_keeps_three_parts_for_distinct_schema() -> None:
    assert qualify_table("mydb", "public", "orders") == "mydb.public.orders"


def test_omits_missing_parts() -> None:
    assert qualify_table("", "public", "orders") == "public.orders"
    assert qualify_table("db", "", "orders") == "db.orders"
    assert qualify_table("", "", "orders") == "orders"


def test_prompt_renders_two_level_name_for_mysql_tables() -> None:
    rendered = format_tables_for_prompt(
        [
            {
                "name": "SIS_DEPARTMENT",
                "database_name": "dw",
                "schema_name": "dw",
                "columns": [{"name": "DEPARTMENT_NAME", "data_type": "varchar"}],
            }
        ]
    )
    assert "TABLE: dw.SIS_DEPARTMENT" in rendered
    assert "dw.dw." not in rendered


def test_prompt_spells_out_sample_values_without_list_punctuation() -> None:
    rendered = format_tables_for_prompt(
        [
            {
                "name": "orders",
                "columns": [
                    {
                        "name": "status",
                        "data_type": "varchar",
                        "sample_values": ["open", "closed"],
                    },
                    # Legacy nodes still hold a JSON-encoded string.
                    {
                        "name": "channel",
                        "data_type": "varchar",
                        "sample_values": '["web", "store"]',
                    },
                ],
            }
        ]
    )
    assert "status (varchar) | sample values: open, closed" in rendered
    assert "channel (varchar) | sample values: web, store" in rendered
    assert "['open'" not in rendered


def test_prompt_omits_sample_values_when_absent() -> None:
    rendered = format_tables_for_prompt(
        [
            {
                "name": "orders",
                "columns": [
                    {"name": "id", "data_type": "int", "sample_values": None},
                    {"name": "note", "data_type": "text"},
                ],
            }
        ]
    )
    assert "sample values" not in rendered


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
