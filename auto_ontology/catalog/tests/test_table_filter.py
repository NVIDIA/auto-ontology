# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regex allow/deny filtering of relations at catalog-extraction time."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from auto_ontology.catalog.extract import create_dataframe
from auto_ontology.catalog.table_filter import (
    InvalidTableFilterError,
    TableFilter,
    from_connection,
)


def _connector(tables: list[dict], columns: list[dict], **kwargs) -> SimpleNamespace:
    return SimpleNamespace(
        get_tables=lambda: pd.DataFrame(tables),
        get_columns=lambda: pd.DataFrame(columns),
        get_views=lambda: pd.DataFrame(),
        get_queries=lambda: pd.DataFrame(),
        get_pks=lambda: pd.DataFrame(),
        get_fks=lambda: pd.DataFrame(),
        **kwargs,
    )


def test_deny_drops_matching_relations() -> None:
    f = from_connection({"table_deny_regex": "^tmp_"})
    assert f.keep("orders") is True
    assert f.keep("tmp_orders") is False


def test_allow_keeps_only_matching_relations() -> None:
    f = from_connection({"table_allow_regex": "^fact_"})
    assert f.keep("fact_sales") is True
    assert f.keep("dim_date") is False


def test_deny_wins_over_allow() -> None:
    """So "this family, except these" is expressible."""
    f = from_connection(
        {"table_allow_regex": "^fact_", "table_deny_regex": "_deprecated$"}
    )
    assert f.keep("fact_sales") is True
    assert f.keep("fact_sales_deprecated") is False


def test_patterns_are_unanchored_and_case_sensitive() -> None:
    f = from_connection({"table_deny_regex": "stage"})
    assert f.keep("pg_perfbot_flat_all_stage") is False  # substring match
    assert f.keep("pg_perfbot_flat_all_STAGE") is True  # case-sensitive


def test_blank_and_missing_patterns_are_inactive() -> None:
    assert from_connection({}).active is False
    assert from_connection({"table_allow_regex": "   "}).active is False
    assert from_connection(None).active is False


def test_invalid_regex_raises_rather_than_being_ignored() -> None:
    """Silently dropping it would ingest everything the operator excluded."""
    with pytest.raises(InvalidTableFilterError) as exc:
        from_connection({"table_deny_regex": "([unclosed"})
    assert "Table denylist" in str(exc.value)


def test_create_dataframe_applies_the_connectors_filter() -> None:
    tables = [
        {"table_schema": "public", "table_name": "orders"},
        {"table_schema": "public", "table_name": "tmp_orders"},
    ]
    columns = [
        {"table_schema": "public", "table_name": "orders", "column_name": "id"},
        {"table_schema": "public", "table_name": "tmp_orders", "column_name": "id"},
    ]
    connector = _connector(
        tables,
        columns,
        table_filter=TableFilter(
            deny=from_connection({"table_deny_regex": "^tmp_"}).deny
        ),
    )

    out_tables, out_columns, *_ = create_dataframe(connector)

    assert list(out_tables["table_name"]) == ["orders"]
    # Columns must be filtered in lockstep, or the coverage check below would
    # see a described relation that is no longer listed.
    assert list(out_columns["table_name"]) == ["orders"]


def test_create_dataframe_without_a_filter_is_unchanged() -> None:
    tables = [{"table_schema": "public", "table_name": "orders"}]
    columns = [{"table_schema": "public", "table_name": "orders", "column_name": "id"}]

    out_tables, out_columns, *_ = create_dataframe(_connector(tables, columns))

    assert list(out_tables["table_name"]) == ["orders"]
    assert list(out_columns["table_name"]) == ["orders"]


def test_filtering_does_not_trip_the_column_coverage_guard() -> None:
    """Dropping a table whose columns remain would raise; it must not."""
    tables = [
        {"table_schema": "s", "table_name": "keep"},
        {"table_schema": "s", "table_name": "drop_me"},
    ]
    columns = [
        {"table_schema": "s", "table_name": "keep", "column_name": "a"},
        {"table_schema": "s", "table_name": "drop_me", "column_name": "b"},
    ]
    connector = _connector(
        tables, columns, table_filter=from_connection({"table_deny_regex": "drop_me"})
    )

    out_tables, _, *_ = create_dataframe(connector)  # must not raise

    assert list(out_tables["table_name"]) == ["keep"]
