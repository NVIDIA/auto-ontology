# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for deterministic column mapping."""

from __future__ import annotations

from auto_ontology.semantic.deterministic import (
    column_attribute_specs,
    fk_source_columns,
    fk_target_table_names,
    to_term_name,
)


def test_excludes_suggested_fk_columns() -> None:
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


def test_excludes_fk_columns() -> None:
    columns = [
        {"name": "id", "data_type": "integer"},
        {"name": "customer_id", "data_type": "integer"},
        {"name": "amount", "data_type": "numeric"},
    ]
    fks = [{"source_column": "customer_id", "target_table": "customers"}]
    specs = column_attribute_specs(columns, fks)
    assert {s.source_column for s in specs} == {"id", "amount"}
    assert fk_source_columns(fks) == {"customer_id"}


def test_only_existing_descriptions_are_used() -> None:
    columns = [
        {"name": "id", "data_type": "integer", "description": "Primary key."},
        {"name": "amount", "data_type": "numeric"},
    ]
    profiling = {"amount": {"sample_values": ["10.5", "3"]}}
    specs = {
        s.source_column: s
        for s in column_attribute_specs(
            columns, [], columns_profiling_samples=profiling
        )
    }
    assert specs["id"].description == "Primary key."
    assert specs["amount"].description is None


def test_date_column_description_states_stored_notation() -> None:
    columns = [
        {
            "name": "game_date",
            "data_type": "text",
            "description": "Date the match was played.",
        }
    ]
    profiling = {"game_date": {"format": "YYMMDD", "sample_values": []}}
    specs = column_attribute_specs(columns, [], columns_profiling_samples=profiling)
    assert specs[0].description == ("Date the match was played. — format: YYMMDD")


def test_undescribed_date_column_gets_only_the_notation() -> None:
    columns = [{"name": "game_date", "data_type": "date"}]
    profiling = {"game_date": {"format": "YYYY-MM-DD"}}
    specs = column_attribute_specs(columns, [], columns_profiling_samples=profiling)
    assert specs[0].description == "format: YYYY-MM-DD"


def test_to_term_name() -> None:
    assert to_term_name("purchase_orders") == "Purchase Orders"
    assert to_term_name("delta_lite_event") == "Delta Lite Event"
    assert to_term_name("gtl_imaging_event") == "Gtl Imaging Event"
    assert to_term_name("gtl_ui_event") == "Gtl Ui Event"
    assert to_term_name("GtlImagingEvent") == "Gtl Imaging Event"
    assert to_term_name("XMLHttpRequest") == "XML Http Request"


def test_fk_targets_deduped() -> None:
    fks = [
        {"target_table": "customers"},
        {"target_table": "customers"},
        {"target_table": "products"},
    ]
    assert fk_target_table_names(fks) == ["customers", "products"]
