# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from gsf.retrieval.text_to_sql.formatters_util import (
    format_semantic_context,
    format_tables_for_prompt,
    qualify_table,
)


def test_collapses_duplicate_database_and_schema() -> None:
    """MySQL reports TABLE_SCHEMA as the database, so db.db.table is invalid SQL."""
    assert qualify_table("dw", "dw", "SIS_DEPARTMENT") == "dw.SIS_DEPARTMENT"


def test_collapse_ignores_identifier_case() -> None:
    assert qualify_table("analytics", "ANALYTICS", "orders") == "ANALYTICS.orders"


def test_keeps_catalog_when_schema_shares_its_name_on_spark() -> None:
    """A Kyuubi catalog and schema may share a name and stay distinct.

    Collapsing yields ``lakehouse.events``, which Spark resolves as a schema
    under ``spark_catalog`` and rejects with TABLE_OR_VIEW_NOT_FOUND.
    """
    assert (
        qualify_table("lakehouse", "lakehouse", "events", "spark")
        == "lakehouse.lakehouse.events"
    )


def test_keeps_catalog_when_schema_shares_its_name_on_trino() -> None:
    assert qualify_table("hive", "hive", "orders", "trino") == "hive.hive.orders"


def test_catalog_qualified_dialect_check_ignores_case() -> None:
    assert (
        qualify_table("lakehouse", "lakehouse", "orders", "SPARK")
        == "lakehouse.lakehouse.orders"
    )


def test_collapses_for_a_dialect_that_binds_its_database() -> None:
    """Databricks and friends scope the session, so two parts still resolve."""
    assert qualify_table("dw", "dw", "orders", "databricks") == "dw.orders"


def test_distinct_schema_is_unaffected_by_dialect() -> None:
    assert qualify_table("mydb", "public", "orders", "spark") == "mydb.public.orders"


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


def test_prompt_renders_catalog_qualified_name_for_spark_tables() -> None:
    """The model copies these names verbatim, so the catalog has to survive."""
    rendered = format_tables_for_prompt(
        [
            {
                "name": "events",
                "database_name": "lakehouse",
                "schema_name": "lakehouse",
                "columns": [{"name": "cluster_id", "data_type": "string"}],
            }
        ],
        dialect="spark",
    )
    assert "TABLE: lakehouse.lakehouse.events" in rendered


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


def test_semantic_context_keeps_catalog_on_spark() -> None:
    """The anchor, the attribute lines and the join hops all carry the catalog.

    The prompt calls these paths authoritative and tells the model to copy the
    join conditions, so any name here can end up in SQL.
    """
    primary_attribute = {
        "database_name": "lakehouse",
        "schema_name": "lakehouse",
        "table_name": "clusters",
        "col_name": "cluster_id",
        "attr_name": "ClusterId",
    }
    join_paths = [
        {
            "attr_name": "EventName",
            "col_name": "name",
            "database_name": "lakehouse",
            "schema_name": "lakehouse",
            "table_name": "events",
            "path": [
                {
                    "source_database": "lakehouse",
                    "source_schema": "lakehouse",
                    "source_table": "clusters",
                    "source_column": "cluster_id",
                    "target_database": "lakehouse",
                    "target_schema": "lakehouse",
                    "target_table": "events",
                    "target_column": "cluster_id",
                }
            ],
        }
    ]

    rendered = format_semantic_context(
        primary_attribute, join_paths, target_db="lakehouse", dialect="spark"
    )

    assert "Table: lakehouse.lakehouse.clusters" in rendered
    assert "lakehouse.lakehouse.events.name" in rendered
    assert (
        "lakehouse.lakehouse.clusters.cluster_id"
        " = lakehouse.lakehouse.events.cluster_id" in rendered
    )


def test_semantic_context_falls_back_to_target_db_for_bridge_hops() -> None:
    """find_table_bridge omits the per-hop database, so target_db stands in."""
    rendered = format_semantic_context(
        {"schema_name": "lakehouse", "table_name": "clusters", "col_name": "id"},
        [
            {
                "attr_name": "A",
                "col_name": "id",
                "schema_name": "lakehouse",
                "table_name": "events",
                "path": [
                    {
                        "source_schema": "lakehouse",
                        "source_table": "clusters",
                        "source_column": "id",
                        "target_schema": "lakehouse",
                        "target_table": "events",
                        "target_column": "id",
                    }
                ],
            }
        ],
        target_db="lakehouse",
        dialect="spark",
    )

    assert "lakehouse.lakehouse.clusters.id = lakehouse.lakehouse.events.id" in rendered


def test_semantic_context_matches_table_section_spelling() -> None:
    """The hint and the schema context must name the same table identically."""
    table = {
        "name": "clusters",
        "database_name": "lakehouse",
        "schema_name": "lakehouse",
        "columns": [{"name": "cluster_id", "data_type": "string"}],
    }
    tables_section = format_tables_for_prompt([table], dialect="spark")
    hint = format_semantic_context(
        {
            "database_name": "lakehouse",
            "schema_name": "lakehouse",
            "table_name": "clusters",
            "col_name": "cluster_id",
            "attr_name": "ClusterId",
        },
        [],
        dialect="spark",
    )

    assert "TABLE: lakehouse.lakehouse.clusters" in tables_section
    assert "Table: lakehouse.lakehouse.clusters" in hint


def test_prompt_falls_back_to_target_db_when_table_has_no_database() -> None:
    """Legacy catalog rows carry no database_name; target_db supplies it.

    Without the fallback the table renders as ``lakehouse.events`` on Spark —
    a schema under spark_catalog, which does not resolve.
    """
    rendered = format_tables_for_prompt(
        [
            {
                "name": "events",
                "schema_name": "lakehouse",
                "columns": [{"name": "cluster_id", "data_type": "string"}],
            }
        ],
        target_db="lakehouse",
        dialect="spark",
    )

    assert "TABLE: lakehouse.lakehouse.events" in rendered


def test_table_own_database_wins_over_target_db() -> None:
    rendered = format_tables_for_prompt(
        [{"name": "events", "database_name": "other", "schema_name": "lakehouse"}],
        target_db="lakehouse",
        dialect="spark",
    )

    assert "TABLE: other.lakehouse.events" in rendered


def test_legacy_rows_spell_the_same_table_in_both_sections() -> None:
    """The cross-section invariant has to hold without database_name too."""
    tables_section = format_tables_for_prompt(
        [{"name": "clusters", "schema_name": "lakehouse"}],
        target_db="lakehouse",
        dialect="spark",
    )
    hint = format_semantic_context(
        {"schema_name": "lakehouse", "table_name": "clusters", "col_name": "id"},
        [],
        target_db="lakehouse",
        dialect="spark",
    )

    assert "TABLE: lakehouse.lakehouse.clusters" in tables_section
    assert "Table: lakehouse.lakehouse.clusters" in hint
