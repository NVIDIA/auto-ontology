# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from auto_ontology.retrieval.text_to_sql.agents.sql_parse_validation import (
    quote_known_mixed_case_identifiers,
    strip_sqlite_database_qualifiers,
)

_TABLES = [
    {
        "name": "events",
        "columns": [{"name": "kubeType"}, {"name": "clusterId"}],
    }
]


def test_quotes_with_backticks_on_spark() -> None:
    """Spark reads a double-quoted token as a string literal, not an identifier.

    Rendering ``ci."kubeType"`` makes the statement fail to parse outright, so
    an unmapped dialect is worse here than doing nothing.
    """
    sql = "SELECT ci.kubeType FROM lakehouse.lakehouse.events AS ci"
    quoted = quote_known_mixed_case_identifiers(sql, _TABLES, "spark")
    assert "`kubeType`" in quoted
    assert '"kubeType"' not in quoted


def test_quotes_with_backticks_on_databricks() -> None:
    sql = "SELECT ci.kubeType FROM main.gold.events AS ci"
    quoted = quote_known_mixed_case_identifiers(sql, _TABLES, "databricks")
    assert "`kubeType`" in quoted
    assert '"kubeType"' not in quoted


def test_quotes_with_double_quotes_on_postgres() -> None:
    sql = "SELECT ci.kubeType FROM public.events AS ci"
    quoted = quote_known_mixed_case_identifiers(sql, _TABLES, "postgres")
    assert '"kubeType"' in quoted


def test_leaves_all_lowercase_sql_untouched() -> None:
    sql = "SELECT ci.id FROM lakehouse.lakehouse.events AS ci"
    assert quote_known_mixed_case_identifiers(sql, _TABLES, "spark") == sql


def test_sqlite_database_qualifiers_are_dropped() -> None:
    sql = (
        "SELECT s.cds, schools_db.main.satscores.score "
        "FROM schools_db.main.satscores "
        "JOIN schools_db.main.schools AS s ON 1 = 1"
    )
    stripped = strip_sqlite_database_qualifiers(sql, "sqlite")
    assert "schools_db" not in stripped
    assert "FROM main.satscores" in stripped
    assert "main.satscores.score" in stripped


def test_qualifier_strip_leaves_other_sql_alone() -> None:
    sql = "SELECT a FROM main.t"
    assert strip_sqlite_database_qualifiers(sql, "sqlite") is sql
    three_part = "SELECT a FROM db.public.t"
    assert strip_sqlite_database_qualifiers(three_part, "postgres") is three_part
