# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the stage that describes ColumnAttributes after FK resolution."""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest

from auto_ontology.semantic import attribute_descriptions as stage

_MODULE = "auto_ontology.semantic.attribute_descriptions"


def _row(table: str, column: str, **extra: object) -> dict:
    return {
        "attribute_id": f"attr:{table}.{column}",
        "attribute_name": column,
        "term_name": table.title(),
        "term_description": "A buyer",
        "table_id": table,
        "table_name": table,
        "table_description": None,
        "column_id": f"col:{table}.{column}",
        "column_name": column,
        "column_description": None,
        "data_type": "integer",
        "sample_values": None,
        "format": None,
        "family_prefix": None,
        **extra,
    }


@pytest.fixture
def dal() -> Iterator[dict[str, MagicMock]]:
    with (
        patch(f"{_MODULE}.fetch_attributes_to_describe") as rows,
        patch(f"{_MODULE}.fetch_semantic_fk_ends", return_value=[]) as ends,
        patch(
            f"{_MODULE}.store_attribute_descriptions",
            side_effect=lambda descriptions: len(descriptions),
        ) as store,
        patch(f"{_MODULE}.describe_columns") as describe,
    ):
        yield {"rows": rows, "ends": ends, "store": store, "describe": describe}


def _prompt(dal: dict[str, MagicMock]) -> str:
    return dal["describe"].call_args.args[1]


def test_nothing_to_describe_asks_nothing(dal) -> None:
    dal["rows"].return_value = []

    assert stage.describe_column_attributes("shop") == 0
    dal["describe"].assert_not_called()


def test_every_fact_reaches_the_prompt(dal) -> None:
    dal["rows"].return_value = [
        _row(
            "customers",
            "id",
            column_description="Customer number",
            sample_values="[1, 2]",
            family_prefix="NEW_",
        ),
        _row(
            "customers",
            "joined",
            data_type="text",
            sample_values='["240101"]',
            format="YYMMDD",
        ),
    ]
    dal["ends"].return_value = [
        {
            "source_column_id": "col:orders.customer_id",
            "source_table_name": "orders",
            "source_column_name": "customer_id",
            "target_attribute_id": "attr:customers.id",
            "target_table_name": "customers",
            "target_column_name": "id",
        }
    ]
    dal["describe"].return_value = {
        "id": "Identifies the customer.",
        "joined": "When the customer signed up, as YYMMDD.",
    }

    assert stage.describe_column_attributes("shop") == 2

    prompt = _prompt(dal)
    assert "Term: Customers — A buyer" in prompt
    assert "Family: NEW_" in prompt
    assert (
        "  - id (integer) — description: Customer number; samples: 1, 2; "
        "referenced by: orders.customer_id"
    ) in prompt
    assert "  - joined (text) — samples: 240101; format: YYMMDD" in prompt
    assert dal["describe"].call_args.args[2] == ["id", "joined"]
    dal["store"].assert_called_once_with(
        {
            "attr:customers.id": "Identifies the customer.",
            "attr:customers.joined": "When the customer signed up, as YYMMDD.",
        }
    )


def test_a_reference_out_of_the_column_is_named(dal) -> None:
    dal["rows"].return_value = [_row("employees", "manager_id")]
    dal["ends"].return_value = [
        {
            "source_column_id": "col:employees.manager_id",
            "source_table_name": "employees",
            "source_column_name": "manager_id",
            "target_attribute_id": "attr:employees.id",
            "target_table_name": "employees",
            "target_column_name": "id",
        }
    ]
    dal["describe"].return_value = {}

    stage.describe_column_attributes("hr")

    assert "  - manager_id (integer) — references: employees.id" in _prompt(dal)


def test_identical_descriptions_are_not_repeated(dal) -> None:
    dal["rows"].return_value = [
        _row("customers", "id", table_description="a buyer", column_description="ID"),
        _row("customers", "note", column_description="A BUYER"),
    ]
    dal["describe"].return_value = {}

    stage.describe_column_attributes("shop")

    prompt = _prompt(dal)
    assert "Table: customers\n" in prompt
    assert "  - id (integer)\n" in prompt
    assert prompt.endswith("  - note (integer)")


def test_date_columns_show_their_format_not_their_samples(dal) -> None:
    dal["rows"].return_value = [
        _row(
            "orders",
            "placed",
            data_type="date",
            sample_values='["2024-01-01"]',
            format="YYYY-MM-DD",
        )
    ]
    dal["describe"].return_value = {}

    stage.describe_column_attributes("shop")

    assert "  - placed (date) — format: YYYY-MM-DD" in _prompt(dal)


def test_each_table_is_its_own_prompt_and_one_failure_spares_the_rest(dal) -> None:
    dal["rows"].return_value = [_row("customers", "id"), _row("orders", "total")]

    def describe(_system: str, _prompt: str, names: list[str]) -> dict[str, str]:
        if names == ["id"]:
            raise RuntimeError("LLM down")
        return {"total": "What the order cost."}

    dal["describe"].side_effect = describe

    assert stage.describe_column_attributes("shop") == 1
    dal["store"].assert_called_once_with({"attr:orders.total": "What the order cost."})


def test_cancellation_stops_describing(dal) -> None:
    dal["rows"].return_value = [_row("customers", "id")]
    with patch(f"{_MODULE}.is_cancelled", return_value=True):
        assert stage.describe_column_attributes("shop") == 0
    dal["describe"].assert_not_called()
