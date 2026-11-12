# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AGE helpers that need no database: parsing, validation, statement text."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from auto_ontology.graph import age, projection
from auto_ontology.graph.mapping import EDGES


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('"a"', "a"),
        ("2", 2),
        ("true", True),
        ("null", None),
        (
            '{"id": 1, "label": "Term", "properties": {"id": "t"}}::vertex',
            {"id": 1, "label": "Term", "properties": {"id": "t"}},
        ),
        ('[{"id": 1}::vertex, {"id": 2}::edge]::path', [{"id": 1}, {"id": 2}]),
        ("1.5::numeric", 1.5),
        # The annotation text inside a string is data, not an annotation.
        ('{"description": "ends in }::vertex"}', {"description": "ends in }::vertex"}),
        ('"quote \\" then ::edge"', 'quote " then ::edge'),
    ],
)
def test_parse_agtype(text: str, expected: object) -> None:
    assert age.parse_agtype(text) == expected


def test_parse_agtype_none() -> None:
    assert age.parse_agtype(None) is None


@pytest.mark.parametrize("name", ["auto_ontology", "g", "_x1"])
def test_graph_name_accepts(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    monkeypatch.setenv(age.GRAPH_NAME_ENV, name)
    assert age.graph_name() == name


@pytest.mark.parametrize("name", ["", "Graph", "a-b", "a'b", "1abc", "x" * 64])
def test_graph_name_rejects(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    monkeypatch.setenv(age.GRAPH_NAME_ENV, name)
    with pytest.raises(ValueError):
        age.graph_name()


@pytest.mark.parametrize(
    ("value", "expected"), [("1", True), ("true", True), ("", False), ("no", False)]
)
def test_enabled(monkeypatch: pytest.MonkeyPatch, value: str, expected: bool) -> None:
    monkeypatch.setenv(age.ENABLED_ENV, value)
    assert age.enabled() is expected


def test_cypher_refuses_the_dollar_quote() -> None:
    with pytest.raises(ValueError):
        age.cypher(MagicMock(), "g", "RETURN '$cypher$'")


def test_cypher_binds_values_instead_of_splicing_them() -> None:
    conn = MagicMock()
    conn.exec_driver_sql.return_value = [("1",)]
    rows = age.cypher(
        conn, "g", "MATCH (n {name: $name}) RETURN 1 % 2", {"name": "x'); DROP"}
    )

    sql, (payload,) = conn.exec_driver_sql.call_args.args
    assert "DROP" not in sql
    assert "DROP" in payload
    assert "1 %% 2" in sql
    assert rows == [(1,)]


def test_to_agtype_value() -> None:
    when = datetime(2026, 1, 2, tzinfo=timezone.utc)
    assert age.to_agtype_value({"a": [when, Decimal("1.5")]}) == {
        "a": ["2026-01-02T00:00:00+00:00", 1.5]
    }


def test_identifier_rejects_injection() -> None:
    with pytest.raises(ValueError):
        age.identifier("Term) DETACH DELETE (n")


def test_edge_statements_name_both_endpoint_labels() -> None:
    for spec in EDGES:
        statement = projection.create_edges_statement(spec)
        assert f"(a:{spec.source_label} {{id: r.s}})" in statement
        assert f"(b:{spec.target_label} {{id: r.t}})" in statement
        assert f"-[e:{spec.label}]->" in statement


def test_batches_cover_every_row() -> None:
    rows = [{"i": i} for i in range(2501)]
    batches = list(projection._batches(rows, 1000))
    assert [len(b) for b in batches] == [1000, 1000, 501]


def test_sync_if_enabled_is_a_noop_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(age.ENABLED_ENV, raising=False)
    called = MagicMock()
    monkeypatch.setattr(projection, "sync_graph", called)
    assert projection.sync_if_enabled() is None
    called.assert_not_called()


def test_sync_if_enabled_swallows_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(age.ENABLED_ENV, "true")
    monkeypatch.setattr(
        projection, "sync_graph", MagicMock(side_effect=age.AgeUnavailable("x"))
    )
    assert projection.sync_if_enabled() is None
