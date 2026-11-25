# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Table-derived SqlAttribute extraction: prompt input and post-LLM filters."""

from __future__ import annotations

from typing import Any

import pytest
from pytest import MonkeyPatch

from auto_ontology.semantic import sql_attribute_extractor as extractor
from auto_ontology.semantic.models import (
    SqlAttributeExtractionResult,
    SqlAttributeProposal,
)

TABLE = {"id": "t1", "name": "orderlines"}
TERM = {"name": "Order Line", "description": "One line of an order"}
COLUMNS = [
    {"name": "unitprice", "data_type": "numeric", "sample_values": [10, 12.5]},
    {"name": "unitcost", "data_type": "numeric"},
    {"name": "status", "data_type": "text", "sample_values": '["Shipped", "Open"]'},
]


def _proposal(name: str, expression: str) -> SqlAttributeProposal:
    return SqlAttributeProposal(
        question=f"What is the {name}?",
        name=name,
        description=name,
        expression=expression,
    )


def _stub_llm(
    monkeypatch: MonkeyPatch, proposals: list[SqlAttributeProposal]
) -> list[Any]:
    """Return the messages each LLM call received."""
    calls: list[Any] = []
    monkeypatch.setattr(extractor, "get_llm_client", lambda **_k: object())

    def _invoke(_llm: Any, messages: Any, _schema: Any) -> Any:
        calls.append(messages)
        return SqlAttributeExtractionResult(metrics=proposals)

    monkeypatch.setattr(extractor, "invoke_with_structured_output", _invoke)
    return calls


def _run(columns: list[dict[str, Any]] = COLUMNS) -> list[str]:
    return [
        p.name for p in extractor.extract_sql_attributes(TABLE, columns, "sales", TERM)
    ]


def test_fewer_than_two_columns_skips_the_llm(monkeypatch: MonkeyPatch) -> None:
    calls = _stub_llm(monkeypatch, [_proposal("x", "SELECT 1")])

    assert _run(COLUMNS[:1]) == []
    assert calls == []


def test_prompt_lists_columns_with_native_and_json_samples(
    monkeypatch: MonkeyPatch,
) -> None:
    calls = _stub_llm(monkeypatch, [])

    _run()

    prompt = calls[0][1].content
    assert "Table: sales.orderlines" in prompt
    assert "unitprice (numeric) [sample values: 10, 12.5]" in prompt
    assert "status (text) [sample values: 'Shipped', 'Open']" in prompt


def test_good_metric_survives(monkeypatch: MonkeyPatch) -> None:
    _stub_llm(
        monkeypatch,
        [
            _proposal(
                "Order Line Gross Margin",
                "SELECT (sales.orderlines.unitprice - sales.orderlines.unitcost) "
                '/ NULLIF(sales.orderlines.unitprice, 0) AS "Gross Margin" '
                "FROM sales.orderlines",
            )
        ],
    )

    assert _run() == ["Order Line Gross Margin"]


def test_duplicate_expressions_are_dropped(monkeypatch: MonkeyPatch) -> None:
    sql = "SELECT a.b.unitprice - a.b.unitcost AS m FROM a.b"
    _stub_llm(monkeypatch, [_proposal("one", sql), _proposal("two", sql.upper())])

    assert _run() == ["one"]


@pytest.mark.parametrize(
    "expression",
    [
        'SELECT unitprice AS "Price" FROM sales.orderlines',
        "SELECT LENGTH(status) - LENGTH(unitcost) AS x FROM sales.orderlines",
        "SELECT unitprice / year AS x FROM sales.orderlines",
        "SELECT unitprice / EXTRACT(month FROM orderdate) AS x FROM sales.orderlines",
        "SELECT validto - valid_from AS x FROM sales.orderlines",
        "SELECT unitprice - unitcost AS x FROM sales.orderlines "
        "WHERE status = 'Cancelled'",
    ],
    ids=[
        "single-column-alias",
        "length-arithmetic",
        "year-divisor",
        "month-divisor",
        "validfrom-validto",
        "unsampled-where-literal",
    ],
)
def test_filtered_proposals_are_dropped(
    monkeypatch: MonkeyPatch, expression: str
) -> None:
    _stub_llm(monkeypatch, [_proposal("bad", expression)])

    assert _run() == []


def test_where_literal_backed_by_sample_values_survives(
    monkeypatch: MonkeyPatch,
) -> None:
    _stub_llm(
        monkeypatch,
        [
            _proposal(
                "Shipped Margin",
                "SELECT unitprice - unitcost AS x FROM sales.orderlines "
                "WHERE status = 'shipped'",
            )
        ],
    )

    assert _run() == ["Shipped Margin"]


def test_llm_failure_returns_nothing(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(extractor, "get_llm_client", lambda **_k: object())
    monkeypatch.setattr(extractor, "invoke_with_structured_output", lambda *_a: None)

    assert _run() == []
