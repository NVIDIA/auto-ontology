"""Tests for process_table taxonomy compilation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd

from gsf.semantic.visit_enter import calculate_columns_profiling, process_table


@patch("gsf.semantic.visit_enter.store_column_uniqueness")
@patch("gsf.semantic.visit_enter.store_column_sample_values")
def test_calculate_columns_profiling_unhashable_values(
    mock_store_samples: MagicMock,
    mock_store_unique: MagicMock,
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


@patch("gsf.semantic.visit_enter.store_column_uniqueness")
@patch("gsf.semantic.visit_enter.store_column_sample_values")
def test_calculate_columns_profiling(
    mock_store_samples: MagicMock,
    mock_store_unique: MagicMock,
) -> None:
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "status": ["open", "open", "closed", "open"],
            "created_at": ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"],
            "token": ["a" * 40, "b" * 40, "a" * 40, "c" * 40],
        }
    )
    # status and token each have <5 distinct sample values, so each gets a
    # DISTINCT probe. Return empty so the sample top-N values are kept as-is.
    connector = MagicMock()
    connector.execute.side_effect = [
        df,
        pd.DataFrame({"status": []}),
        pd.DataFrame({"token": []}),
    ]

    table = {"id": "t1", "name": "orders", "schema_name": "public"}
    columns = [
        {"name": "id", "data_type": "integer"},
        {"name": "status", "data_type": "text"},
        {"name": "created_at", "data_type": "timestamp"},
        {"name": "token", "data_type": "text"},
    ]

    result = calculate_columns_profiling(table, columns, connector)

    sql = connector.execute.call_args_list[0][0][0]
    assert "public.orders" in sql
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


@patch("gsf.semantic.visit_enter.store_column_uniqueness")
@patch("gsf.semantic.visit_enter.store_column_sample_values")
def test_calculate_columns_profiling_distinct_only_when_under_top_n(
    mock_store_samples: MagicMock,
    mock_store_unique: MagicMock,
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
