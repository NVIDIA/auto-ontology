"""Tests for process_table taxonomy compilation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from gsf.semantic.visit_enter import process_table


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
