# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from gsf.retrieval.text_to_sql.agents.sql_parse_validation import (
    quote_known_mixed_case_identifiers,
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
