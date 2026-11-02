# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Catalog descriptions for columns the source left undescribed."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from auto_ontology.semantic import catalog_descriptions as cd
from auto_ontology.semantic.models import ColumnDescription, ColumnDescriptionsResult


@pytest.mark.parametrize(
    ("column", "expected"),
    [
        ({"name": "status", "description": None}, True),
        ({"name": "status", "description": "   "}, True),
        ({"name": "customer_id", "description": "Customer ID"}, True),
        ({"name": "Order Date", "description": "order_date."}, True),
        ({"name": "status", "description": "Lifecycle state of the order."}, False),
        ({"name": "", "description": None}, False),
    ],
)
def test_needs_description(column: dict[str, Any], expected: bool) -> None:
    assert cd.needs_description(column) is expected


def _result(*names: str) -> ColumnDescriptionsResult:
    return ColumnDescriptionsResult(
        columns=[
            ColumnDescription(column_name=n, description=f"{n} text") for n in names
        ]
    )


def _prompt(invoke: MagicMock, call: int = 0) -> str:
    return invoke.call_args_list[call].args[1][1].content


@patch("auto_ontology.semantic.catalog_descriptions.get_llm_client")
@patch("auto_ontology.semantic.catalog_descriptions.invoke_with_structured_output")
def test_only_undescribed_columns_are_asked_for(
    invoke: MagicMock, _client: MagicMock
) -> None:
    invoke.return_value = _result("status", "cust_id")
    columns = [
        {"name": "order_id", "data_type": "INTEGER", "description": "Order key."},
        {"name": "status", "data_type": "TEXT", "description": None},
        {"name": "cust_id", "data_type": "TEXT", "description": "Cust ID"},
    ]

    out = cd.describe_catalog_columns(
        {"name": "orders"},
        columns,
        {"status": {"sample_values": ["open", "closed"]}},
    )

    assert out == {"status": "status text", "cust_id": "cust_id text"}
    to_describe = _prompt(invoke).split("Columns to describe:\n", 1)[1]
    assert "- status (TEXT) — samples: open, closed" in to_describe
    assert "- cust_id (TEXT)" in to_describe
    assert "order_id" not in to_describe
    assert "- order_id (INTEGER): Order key." in _prompt(invoke)


@patch("auto_ontology.semantic.catalog_descriptions.get_llm_client")
@patch("auto_ontology.semantic.catalog_descriptions.invoke_with_structured_output")
def test_nothing_to_describe_makes_no_call(
    invoke: MagicMock, _client: MagicMock
) -> None:
    columns = [{"name": "status", "description": "Lifecycle state."}]

    assert cd.describe_catalog_columns({"name": "orders"}, columns) == {}
    invoke.assert_not_called()


@patch("auto_ontology.semantic.catalog_descriptions.get_llm_client")
@patch("auto_ontology.semantic.catalog_descriptions.invoke_with_structured_output")
def test_a_failed_batch_is_skipped_and_the_rest_kept(
    invoke: MagicMock, _client: MagicMock
) -> None:
    columns = [{"name": f"c{i}", "data_type": "TEXT"} for i in range(20)]
    invoke.side_effect = [RuntimeError("boom"), _result("c15", "c16")]

    out = cd.describe_catalog_columns({"name": "wide"}, columns)

    assert invoke.call_count == 2
    assert out == {"c15": "c15 text", "c16": "c16 text"}


@patch("auto_ontology.semantic.catalog_descriptions.get_llm_client")
@patch("auto_ontology.semantic.catalog_descriptions.invoke_with_structured_output")
def test_dates_show_their_format_not_their_samples(
    invoke: MagicMock, _client: MagicMock
) -> None:
    invoke.return_value = _result("shipped")
    columns = [{"name": "shipped", "data_type": "DATE"}]

    cd.describe_catalog_columns(
        {"name": "orders"},
        columns,
        {"shipped": {"sample_values": ["2020-01-02"], "format": "YYYY-MM-DD"}},
    )

    assert "- shipped (DATE) — format: YYYY-MM-DD\n" in _prompt(invoke) + "\n"
    assert "2020-01-02" not in _prompt(invoke)
