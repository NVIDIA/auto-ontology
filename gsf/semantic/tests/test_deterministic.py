"""Tests for deterministic column mapping."""

from __future__ import annotations

from unittest.mock import patch

from gsf.semantic.deterministic import (
    column_attribute_specs,
    fk_source_columns,
    fk_target_table_names,
    to_term_name,
)


@patch("gsf.semantic.deterministic._generate_column_descriptions", return_value={})
def test_excludes_suggested_fk_columns(_mock_desc) -> None:
    columns = [
        {"name": "id", "data_type": "integer"},
        {"name": "vendor_id", "data_type": "integer"},
        {"name": "amount", "data_type": "numeric"},
    ]
    specs = column_attribute_specs(
        columns,
        [],
        suggested_fk_columns={"vendor_id"},
    )
    assert {s.source_column for s in specs} == {"id", "amount"}


@patch("gsf.semantic.deterministic._generate_column_descriptions", return_value={})
def test_excludes_fk_columns(_mock_desc) -> None:
    columns = [
        {"name": "id", "data_type": "integer"},
        {"name": "customer_id", "data_type": "integer"},
        {"name": "amount", "data_type": "numeric"},
    ]
    fks = [{"source_column": "customer_id", "target_table": "customers"}]
    specs = column_attribute_specs(columns, fks)
    assert {s.source_column for s in specs} == {"id", "amount"}
    assert fk_source_columns(fks) == {"customer_id"}


@patch(
    "gsf.semantic.deterministic._generate_column_descriptions",
    return_value={"amount": "The monetary amount of the order."},
)
def test_llm_description_used_as_fallback(_mock_desc) -> None:
    columns = [
        {"name": "id", "data_type": "integer", "description": "Primary key."},
        {"name": "amount", "data_type": "numeric"},
    ]
    specs = {s.source_column: s for s in column_attribute_specs(columns, [])}
    # Existing description wins over the LLM one.
    assert specs["id"].description == "Primary key."
    # LLM description fills in when the column has none.
    assert specs["amount"].description == "The monetary amount of the order."


def test_to_term_name() -> None:
    assert to_term_name("purchase_orders") == "PurchaseOrders"


def test_fk_targets_deduped() -> None:
    fks = [
        {"target_table": "customers"},
        {"target_table": "customers"},
        {"target_table": "products"},
    ]
    assert fk_target_table_names(fks) == ["customers", "products"]
