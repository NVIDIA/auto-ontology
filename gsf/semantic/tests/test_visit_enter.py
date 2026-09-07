"""Tests for process_table taxonomy compilation."""

from __future__ import annotations

import re
from datetime import date, datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

import pytest

from gsf.semantic.visit_enter import (
    _json_ready_sample,
    _keep_persisted_sample,
    _quoted_identifier,
    _sample_key,
    calculate_columns_profiling,
    process_table,
)


@pytest.mark.parametrize(
    ("dialect", "expected"),
    [
        ("sqlite", '"Sales Orders"'),
        ("postgres", '"Sales Orders"'),
        ("mysql", "`Sales Orders`"),
        ("spark", "`Sales Orders`"),
        (None, '"Sales Orders"'),
        ("not-a-dialect", '"Sales Orders"'),
    ],
)
def test_quoted_identifier_per_dialect(dialect: str | None, expected: str) -> None:
    # Backticks on MySQL/Spark, double quotes elsewhere; an unknown dialect
    # still quotes, since a bare identifier is what loses the table.
    assert _quoted_identifier("Sales Orders", dialect) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (np.int64(7), 7),
        (np.float64(1.5), 1.5),
        (np.bool_(True), True),
        (Decimal("10"), 10),
        (Decimal("10.5"), 10.5),
        (datetime(2020, 1, 2, 3, 4, 5), "2020-01-02T03:04:05"),
        (date(2020, 1, 2), "2020-01-02"),
        (b"bytes", "bytes"),
        (["a", np.int64(1)], ["a", 1]),
        ({"k": Decimal("2")}, {"k": 2}),
        ("plain", "plain"),
        (None, None),
    ],
)
def test_json_ready_sample_converts_driver_types(raw: object, expected: object) -> None:
    converted = _json_ready_sample(raw)
    assert converted == expected
    # numpy scalars compare equal to their Python counterparts, so identity of
    # the type is the only thing that shows the unwrap actually happened.
    assert type(converted) is type(expected)


def test_sample_key_separates_values_a_str_key_would_merge() -> None:
    # str() would render all three as "1"/"True" collisions across types.
    assert len({_sample_key(1), _sample_key("1"), _sample_key(True)}) == 3
    # Unhashable values still get a key, and equal ones share it.
    assert _sample_key(["a", "b"]) == _sample_key(["a", "b"])
    assert _sample_key({"k": 1}) != _sample_key({"k": 2})


def test_keep_persisted_sample_caps_text_only() -> None:
    assert _keep_persisted_sample("short") is True
    assert _keep_persisted_sample("x" * 31) is False
    assert _keep_persisted_sample(10**40) is True
    assert _keep_persisted_sample(1.5) is True
    assert _keep_persisted_sample(False) is True
    # Non-scalars are judged by how long they render.
    assert _keep_persisted_sample(["a", "b"]) is True
    assert _keep_persisted_sample(["x" * 40]) is False


@patch("gsf.semantic.visit_enter.store_column_date_formats")
@patch("gsf.semantic.visit_enter.store_column_uniqueness")
@patch("gsf.semantic.visit_enter.store_column_sample_values")
def test_calculate_columns_profiling_quotes_name_with_space(
    mock_store_samples: MagicMock,
    mock_store_unique: MagicMock,
    mock_store_dates: MagicMock,
) -> None:
    # Unquoted "main.Sales Orders" parses as table main.Sales, so the probe
    # raised "no such table" and every column of the table was dropped.
    df = pd.DataFrame({"OrderDate": ["5/31/18", "6/1/18", "12/4/18"]})
    connector = MagicMock()
    connector.dialect = "sqlite"
    connector.execute.return_value = df

    table = {"id": "t1", "name": "Sales Orders", "schema_name": "main"}
    columns = [{"name": "OrderDate", "data_type": "TEXT"}]

    result = calculate_columns_profiling(table, columns, connector)

    sql = connector.execute.call_args_list[0][0][0]
    assert '"main"."Sales Orders"' in sql
    assert result["OrderDate"]["format"] == "M/D/YY"


@patch("gsf.semantic.visit_enter.store_column_date_formats")
@patch("gsf.semantic.visit_enter.store_column_uniqueness")
@patch("gsf.semantic.visit_enter.store_column_sample_values")
def test_calculate_columns_profiling_unhashable_values(
    mock_store_samples: MagicMock,
    mock_store_unique: MagicMock,
    mock_store_dates: MagicMock,
) -> None:
    # Postgres array / JSON columns come back as Python lists/dicts, which are
    # unhashable — profiling must not crash on them.
    df = pd.DataFrame(
        {
            "id": [1, 2],
            "tags": [["a", "b"], ["a", "b"]],
            "meta": [{"k": 1}, {"k": 2}],
        }
    )
    connector = MagicMock()
    connector.execute.return_value = df

    table = {"id": "t1", "name": "orders", "schema_name": "public"}
    columns = [
        {"name": "id", "data_type": "integer"},
        {"name": "tags", "data_type": "ARRAY"},
        {"name": "meta", "data_type": "jsonb"},
    ]

    result = calculate_columns_profiling(table, columns, connector)

    assert set(result) == {"id", "tags", "meta"}
    assert result["tags"]["is_unique"] is False  # ["a","b"] repeated
    assert result["meta"]["is_unique"] is True
    assert result["id"]["is_unique"] is True

    # Containers stay containers rather than being flattened to their repr.
    assert result["tags"]["sample_values"] == [["a", "b"]]
    assert result["meta"]["sample_values"] == [{"k": 1}, {"k": 2}]

    stored = mock_store_samples.call_args[0][1]
    assert stored["id"] == [1, 2]
    assert stored["tags"] == [["a", "b"]]


@patch("gsf.semantic.visit_enter.store_column_date_formats")
@patch("gsf.semantic.visit_enter.store_column_uniqueness")
@patch("gsf.semantic.visit_enter.store_column_sample_values")
def test_calculate_columns_profiling(
    mock_store_samples: MagicMock,
    mock_store_unique: MagicMock,
    mock_store_dates: MagicMock,
) -> None:
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "status": ["open", "open", "closed", "open"],
            "created_at": ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"],
            "token": ["a" * 40, "b" * 40, "a" * 40, "c" * 40],
        }
    )

    def _fake_execute(sql: str) -> pd.DataFrame:
        # The main profiling query (SELECT *) gets the full table. A
        # per-column DISTINCT probe (see _distinct_values_if_low_cardinality)
        # fires for non-unique text columns ("status", "token" here) — a
        # real connector would return only that column, so the mock must
        # too, or df.iloc[:, 0] silently reads whichever column comes first
        # instead of the one actually queried.
        match = re.search(r'SELECT DISTINCT "([^"]+)"', sql)
        if not match:
            return df
        col = match.group(1)
        return pd.DataFrame({col: df[col].dropna().unique().tolist()})

    connector = MagicMock()
    connector.dialect = "postgres"
    connector.execute.side_effect = _fake_execute

    table = {"id": "t1", "name": "orders", "schema_name": "public"}
    columns = [
        {"name": "id", "data_type": "integer"},
        {"name": "status", "data_type": "text"},
        {"name": "created_at", "data_type": "timestamp"},
        {"name": "token", "data_type": "text"},
    ]

    result = calculate_columns_profiling(table, columns, connector)

    # The main profiling query is the first connector.execute call — later
    # calls (e.g. a per-column DISTINCT check for long-string columns like
    # "token") would otherwise shadow it if we looked at call_args (last call).
    sql = connector.execute.call_args_list[0][0][0]
    assert '"public"."orders"' in sql
    assert "LIMIT 1000" in sql
    assert connector.execute.call_count == 3

    # Returned dict includes every column (dates, long strings included),
    # each with its top-5 sample values and is_unique flag.
    assert set(result) == {"id", "status", "created_at", "token"}
    assert result["status"]["sample_values"][0] == "open"
    assert len(result["created_at"]["sample_values"]) == 4
    assert result["id"]["is_unique"] is True
    assert result["status"]["is_unique"] is False
    assert result["created_at"]["is_unique"] is True
    assert result["token"]["is_unique"] is False
    assert result["created_at"]["format"] == "YYYY-MM-DD"
    mock_store_dates.assert_called_once()
    assert mock_store_dates.call_args[0][1] == {"created_at": "YYYY-MM-DD"}

    # Uniqueness persisted for every column.
    uniqueness = mock_store_unique.call_args[0][1]
    assert uniqueness == {
        "id": True,
        "status": False,
        "created_at": True,
        "token": False,
    }

    # Stored sample values exclude the date column and the >30-char token values.
    stored = mock_store_samples.call_args[0][1]
    assert "created_at" not in stored
    assert "token" not in stored
    assert stored["status"][0] == "open"
    assert stored["id"] == [1, 2, 3, 4]
    assert all(type(v) is int for v in stored["id"])


@patch("gsf.semantic.visit_enter.store_column_date_formats")
@patch("gsf.semantic.visit_enter.store_column_uniqueness")
@patch("gsf.semantic.visit_enter.store_column_sample_values")
def test_calculate_columns_profiling_preserves_scalar_types(
    mock_store_samples: MagicMock,
    mock_store_unique: MagicMock,
    mock_store_dates: MagicMock,
) -> None:
    """Numbers and booleans profile as themselves, not as their str() form."""
    df = pd.DataFrame(
        {
            "amount": [10, 20, 10, 30],
            "ratio": [1.5, 2.5, 1.5, 3.5],
            "active": [True, False, True, True],
            # Wider than the 30-char cap once rendered — the cap is about prose,
            # so a number is kept however many digits it has.
            "big": [10**40, 10**41, 10**40, 10**42],
            "label": ["a" * 40, "b", "a" * 40, "c"],
        }
    )
    connector = MagicMock()
    connector.dialect = "postgres"
    # Only `label` is text with fewer than 5 distinct values, so it is the one
    # column that triggers a DISTINCT probe; return empty to keep the top-N set.
    connector.execute.side_effect = [df, pd.DataFrame({"label": []})]

    table = {"id": "t1", "name": "orders", "schema_name": "public"}
    columns = [
        {"name": "amount", "data_type": "integer"},
        {"name": "ratio", "data_type": "numeric"},
        {"name": "active", "data_type": "boolean"},
        {"name": "big", "data_type": "numeric"},
        {"name": "label", "data_type": "text"},
    ]

    result = calculate_columns_profiling(table, columns, connector)

    assert result["amount"]["sample_values"] == [10, 20, 30]
    assert result["ratio"]["sample_values"] == [1.5, 2.5, 3.5]
    assert result["active"]["sample_values"] == [True, False]
    assert all(isinstance(v, int) for v in result["amount"]["sample_values"])
    assert all(isinstance(v, float) for v in result["ratio"]["sample_values"])
    assert all(isinstance(v, bool) for v in result["active"]["sample_values"])

    stored = mock_store_samples.call_args[0][1]
    assert stored["amount"] == [10, 20, 30]
    assert stored["ratio"] == [1.5, 2.5, 3.5]
    assert stored["active"] == [True, False]
    # The length cap applies to text only: the huge numbers survive, the
    # 40-char string does not.
    assert stored["big"] == [10**40, 10**41, 10**42]
    assert stored["label"] == ["b", "c"]


@patch("gsf.semantic.visit_enter.store_column_date_formats")
@patch("gsf.semantic.visit_enter.store_column_uniqueness")
@patch("gsf.semantic.visit_enter.store_column_sample_values")
def test_calculate_columns_profiling_infers_date_format(
    mock_store_samples: MagicMock,
    mock_store_unique: MagicMock,
    mock_store_dates: MagicMock,
) -> None:
    # Every column here is unique, so no DISTINCT-cardinality probe fires —
    # kept simple/isolated from test_calculate_columns_profiling, which
    # exercises that probe (via non-unique "status"/"token" columns) instead.
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "created_at": ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"],
        }
    )
    connector = MagicMock()
    connector.execute.return_value = df

    table = {"id": "t1", "name": "orders", "schema_name": "public"}
    columns = [
        {"name": "id", "data_type": "integer"},
        {"name": "created_at", "data_type": "timestamp"},
    ]

    result = calculate_columns_profiling(table, columns, connector)

    assert result["created_at"]["format"] == "YYYY-MM-DD"
    mock_store_dates.assert_called_once()
    assert mock_store_dates.call_args[0][1] == {"created_at": "YYYY-MM-DD"}
    # Declared date/timestamp type: sample_values stay excluded regardless of
    # the format inference (pre-existing _is_excluded_sample_type behavior).
    stored = mock_store_samples.call_args[0][1]
    assert "created_at" not in stored


@patch("gsf.semantic.visit_enter.store_column_date_formats")
@patch("gsf.semantic.visit_enter.store_column_uniqueness")
@patch("gsf.semantic.visit_enter.store_column_sample_values")
def test_text_column_with_date_format_keeps_sample_values(
    mock_store_samples: MagicMock,
    mock_store_unique: MagicMock,
    mock_store_dates: MagicMock,
) -> None:
    """A TEXT-typed date column, unlike a declared date/timestamp type, keeps
    its persisted sample_values alongside the inferred format — samples stay
    available to semantic_fk.py's SQL-probe fallback (which reads persisted
    Column.sample_values, not this call's in-memory profiling dict), and date
    values are short enough that keeping them costs little.
    """
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "signup_date": ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"],
        }
    )
    connector = MagicMock()
    connector.execute.return_value = df

    table = {"id": "t1", "name": "users", "schema_name": "public"}
    columns = [
        {"name": "id", "data_type": "integer"},
        {"name": "signup_date", "data_type": "text"},
    ]

    result = calculate_columns_profiling(table, columns, connector)

    assert result["signup_date"]["format"] == "YYYY-MM-DD"
    stored = mock_store_samples.call_args[0][1]
    assert "signup_date" in stored
    assert set(stored["signup_date"]) == {
        "2020-01-01",
        "2020-01-02",
        "2020-01-03",
        "2020-01-04",
    }


@patch("gsf.semantic.visit_enter.store_column_date_formats")
@patch("gsf.semantic.visit_enter.store_column_uniqueness")
@patch("gsf.semantic.visit_enter.store_column_sample_values")
def test_calculate_columns_profiling_distinct_only_when_under_top_n(
    mock_store_samples: MagicMock,
    mock_store_unique: MagicMock,
    mock_store_dates: MagicMock,
) -> None:
    """DISTINCT probes run only for text columns with fewer than 5 sample values."""
    sample_df = pd.DataFrame(
        {
            # 5+ distinct values in the sample → no DISTINCT probe.
            "city": ["a", "b", "c", "d", "e", "a"],
            # Fewer than 5 distinct values → DISTINCT probe for rare enums.
            "status": ["open", "open", "closed", "open", "open", "open"],
        }
    )
    status_distinct_df = pd.DataFrame({"status": ["open", "closed", "banned"]})

    connector = MagicMock()
    connector.execute.side_effect = [sample_df, status_distinct_df]

    table = {"id": "t1", "name": "orders", "schema_name": "public"}
    columns = [
        {"name": "city", "data_type": "string"},
        {"name": "status", "data_type": "text"},
    ]

    result = calculate_columns_profiling(table, columns, connector)

    assert connector.execute.call_count == 2
    distinct_sql = connector.execute.call_args_list[1][0][0]
    assert "SELECT DISTINCT" in distinct_sql
    assert '"status"' in distinct_sql
    assert '"city"' not in distinct_sql

    assert result["city"]["sample_values"] == ["a", "b", "c", "d", "e"]
    assert "banned" in result["status"]["sample_values"]
    assert "open" in result["status"]["sample_values"]
    assert "closed" in result["status"]["sample_values"]


@patch("gsf.semantic.visit_enter.merge_column_attribute")
@patch("gsf.semantic.visit_enter.merge_term")
@patch("gsf.semantic.visit_enter.suggest_potential_foreign_keys")
@patch("gsf.semantic.visit_enter.extract_term")
def test_process_table_writes_term_and_attributes(
    mock_term: MagicMock,
    mock_fk_suggest: MagicMock,
    mock_merge_term: MagicMock,
    mock_merge_col_attr: MagicMock,
) -> None:
    from gsf.semantic.models import (
        PotentialFkResult,
        TableTermsResult,
        TermAttributeAssignment,
        TermProposal,
    )

    mock_fk_suggest.return_value = PotentialFkResult()
    mock_term.return_value = TableTermsResult(
        terms=[
            TermProposal(
                name="Order",
                description="Order entity",
                attributes=[
                    TermAttributeAssignment(
                        source_column="amount",
                        display_name="Total Amount",
                    )
                ],
            )
        ]
    )

    table = {"id": "t1", "name": "orders", "description": "", "schema_name": "public"}
    ctx = {
        "columns": [{"name": "amount", "data_type": "numeric"}],
        "fks": [],
    }

    process_table(table, ctx, domain_summary=None)

    mock_merge_term.assert_called_once_with("Order", "Order entity", "t1", synonyms=[])
    mock_merge_col_attr.assert_called_once()


@patch("gsf.semantic.visit_enter.merge_column_attribute")
@patch("gsf.semantic.visit_enter.merge_term")
@patch("gsf.semantic.visit_enter.suggest_potential_foreign_keys")
@patch("gsf.semantic.visit_enter.extract_term")
def test_process_table_skips_fk_columns(
    mock_term: MagicMock,
    mock_fk_suggest: MagicMock,
    mock_merge_term: MagicMock,
    mock_merge_col_attr: MagicMock,
) -> None:
    from gsf.semantic.models import (
        PotentialFkResult,
        TableTermsResult,
        TermProposal,
    )

    mock_fk_suggest.return_value = PotentialFkResult()
    mock_term.return_value = TableTermsResult(
        terms=[TermProposal(name="Order", description="", attributes=[])]
    )

    table = {"id": "t1", "name": "orders", "description": "", "schema_name": ""}
    ctx = {
        "columns": [{"name": "customer_id", "data_type": "integer"}],
        "fks": [
            {
                "source_column": "customer_id",
                "target_table": "customers",
                "target_table_id": "t2",
            }
        ],
    }

    process_table(table, ctx, domain_summary=None)

    mock_merge_term.assert_not_called()


@patch("gsf.semantic.visit_enter.merge_column_attribute")
@patch("gsf.semantic.visit_enter.merge_term")
@patch("gsf.semantic.visit_enter.suggest_potential_foreign_keys")
@patch("gsf.semantic.visit_enter.extract_term")
def test_process_table_multiple_terms(
    mock_term: MagicMock,
    mock_fk_suggest: MagicMock,
    mock_merge_term: MagicMock,
    mock_merge_col_attr: MagicMock,
) -> None:
    from gsf.semantic.models import (
        PotentialFkResult,
        TableTermsResult,
        TermAttributeAssignment,
        TermProposal,
    )

    mock_fk_suggest.return_value = PotentialFkResult()
    mock_term.return_value = TableTermsResult(
        terms=[
            TermProposal(
                name="Order",
                description="Core order",
                attributes=[
                    TermAttributeAssignment(
                        source_column="amount", display_name="Total Amount"
                    )
                ],
            ),
            TermProposal(
                name="Audit Metadata",
                description="Audit fields",
                attributes=[
                    TermAttributeAssignment(
                        source_column="created_at", display_name="Created At"
                    )
                ],
            ),
        ]
    )

    table = {"id": "t1", "name": "orders", "description": "", "schema_name": "public"}
    ctx = {
        "columns": [
            {"name": "amount", "data_type": "numeric"},
            {"name": "created_at", "data_type": "timestamp"},
        ],
        "fks": [],
    }

    process_table(table, ctx, domain_summary=None)

    assert mock_merge_term.call_count == 2
    assert mock_merge_col_attr.call_count == 2
