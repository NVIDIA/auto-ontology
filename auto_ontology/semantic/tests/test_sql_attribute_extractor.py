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


def test_differently_formatted_duplicates_are_dropped(
    monkeypatch: MonkeyPatch,
) -> None:
    _stub_llm(
        monkeypatch,
        [
            _proposal("one", _MARGIN + "status = 'Open'"),
            _proposal(
                "two",
                "select  unitprice-unitcost as X\nfrom sales.orderlines "
                "where STATUS = 'Open'",
            ),
        ],
    )

    assert _run() == ["one"]


def test_expressions_differing_only_in_a_literal_are_kept(
    monkeypatch: MonkeyPatch,
) -> None:
    columns = [
        *COLUMNS[:2],
        {"name": "status", "data_type": "text", "sample_values": ["Open", "OPEN"]},
    ]
    _stub_llm(
        monkeypatch,
        [
            _proposal("one", _MARGIN + "status = 'Open'"),
            _proposal("two", _MARGIN + "status = 'OPEN'"),
        ],
    )

    assert _run(columns) == ["one", "two"]


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


_MARGIN = "SELECT unitprice - unitcost AS x FROM sales.orderlines WHERE "


@pytest.mark.parametrize(
    "where",
    [
        "sales.orderlines.status = 'Shipped'",
        "'Shipped' = status",
        "status IN ('Open', 'Shipped') AND unitprice <> 12.5",
        "unitprice = 10",
        "unitcost > 0",
    ],
    ids=["qualified", "literal-first", "in-and-numeric", "numeric-eq", "range"],
)
def test_where_literal_backed_by_its_columns_samples_survives(
    monkeypatch: MonkeyPatch, where: str
) -> None:
    _stub_llm(monkeypatch, [_proposal("Margin", _MARGIN + where)])

    assert _run() == ["Margin"]


@pytest.mark.parametrize(
    "where",
    [
        "status = 'shipped'",
        "status = '10'",
        "unitprice = '10'",
        "unitcost = 5",
        "status IN ('Open', 'Cancelled')",
        "LOWER(status) = 'Shipped'",
        "status LIKE 'Ship%'",
    ],
    ids=[
        "case-mismatch",
        "another-columns-sample",
        "string-vs-numeric-sample",
        "unsampled-column",
        "one-in-value-unsampled",
        "wrapped-column",
        "untied-string-literal",
    ],
)
def test_where_literal_not_in_its_columns_samples_is_dropped(
    monkeypatch: MonkeyPatch, where: str
) -> None:
    _stub_llm(monkeypatch, [_proposal("Margin", _MARGIN + where)])

    assert _run() == []


def test_llm_failure_raises(monkeypatch: MonkeyPatch) -> None:
    """A failed call is not reported as an empty proposal list."""
    monkeypatch.setattr(extractor, "get_llm_client", lambda **_k: object())
    monkeypatch.setattr(extractor, "invoke_with_structured_output", lambda *_a: None)

    with pytest.raises(extractor.SqlAttributeExtractionFailed):
        _run()


def test_zero_proposals_is_not_a_failure(monkeypatch: MonkeyPatch) -> None:
    _stub_llm(monkeypatch, [])

    assert _run() == []
