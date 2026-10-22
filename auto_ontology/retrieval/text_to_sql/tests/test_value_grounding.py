# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from auto_ontology.retrieval.text_to_sql.agents import value_grounding
from auto_ontology.retrieval.text_to_sql.agents.value_grounding import (
    EvidenceColumnRef,
    GroundingColumnRef,
    ValueGroundingAgent,
    ValueGroundingCandidate,
    ValueGroundingExtraction,
)
from auto_ontology.retrieval.text_to_sql.db_probe.value_lookup import (
    lookup_text_value,
)


def _rows() -> list[dict]:
    return [
        {
            "database_name": "card_games",
            "table_schema": "main",
            "table_name": "cards",
            "table_id": "cards-table",
            "column_name": "name",
            "column_id": "cards-name",
            "data_type": "TEXT",
        },
        {
            "database_name": "card_games",
            "table_schema": "main",
            "table_name": "cards",
            "table_id": "cards-table",
            "column_name": "type",
            "column_id": "cards-type",
            "data_type": "TEXT",
        },
        {
            "database_name": "card_games",
            "table_schema": "main",
            "table_name": "cards",
            "table_id": "cards-table",
            "column_name": "release_year",
            "column_id": "cards-year",
            "data_type": "INTEGER",
        },
    ]


class _FakeProbeExecutor:
    def __init__(self, _connector) -> None:
        self.budget_left = 40

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None


def test_agent_uses_original_inputs_names_only_and_first_matching_suspect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, str] = {}
    probes: list[str] = []
    extraction = ValueGroundingExtraction(
        evidence_columns=[
            EvidenceColumnRef(
                schema_name="main",
                table_name="cards",
                column_name="type",
                evidence_excerpt="cards.type identifies the card type",
            )
        ],
        values=[
            ValueGroundingCandidate(
                value="Angel of Mercy",
                source_excerpt="card name Angel of Mercy",
                suspected_columns=[
                    GroundingColumnRef(
                        schema_name="main",
                        table_name="cards",
                        column_name="type",
                    ),
                    GroundingColumnRef(
                        schema_name="main",
                        table_name="cards",
                        column_name="name",
                    ),
                ],
            ),
            ValueGroundingCandidate(
                value="2001",
                source_excerpt="released in 2001",
                suspected_columns=[
                    GroundingColumnRef(
                        schema_name="main",
                        table_name="cards",
                        column_name="release_year",
                    )
                ],
            ),
            ValueGroundingCandidate(
                value="name",
                source_excerpt="card name",
                suspected_columns=[
                    GroundingColumnRef(
                        schema_name="main",
                        table_name="cards",
                        column_name="name",
                    )
                ],
            ),
        ],
    )

    def fake_invoke(_llm, messages, _schema):
        captured["prompt"] = messages[0].content
        return extraction

    def fake_lookup(_executor, _connector, **kwargs):
        probes.append(kwargs["column_name"])
        if kwargs["column_name"] == "name":
            return "Angel of Mercy", "value"
        return None

    monkeypatch.setattr(
        value_grounding, "fetch_schema_ids_for_database", lambda _db: ["s"]
    )
    monkeypatch.setattr(value_grounding, "fetch_schemas_by_ids", lambda _ids: _rows())
    monkeypatch.setattr(value_grounding, "invoke_with_structured_output", fake_invoke)
    monkeypatch.setattr(value_grounding, "ProbeExecutor", _FakeProbeExecutor)
    monkeypatch.setattr(value_grounding, "lookup_text_value", fake_lookup)

    connector = SimpleNamespace(database_name="card_games", dialect="sqlite")
    result = ValueGroundingAgent().execute(
        {
            "llm": object(),
            "initial_question": ("Find the card name Angel of Mercy released in 2001"),
            "original_evidence": "cards.type identifies the card type",
            "evidence": "rewritten evidence that must not be used",
            "connectors": [connector],
            "path_state": {
                "target_db": "card_games",
                "normalized_question": "rewritten question",
            },
        }
    )

    assert "Find the card name Angel of Mercy released in 2001" in captured["prompt"]
    assert "cards.type identifies the card type" in captured["prompt"]
    assert "rewritten question" not in captured["prompt"]
    assert "rewritten evidence" not in captured["prompt"]
    assert (
        '"schema":"main","table":"cards","columns":["name","release_year","type"]'
        in captured["prompt"]
    )
    assert "data_type" not in captured["prompt"]
    assert probes == ["type", "name"]
    assert result["value_anchors"] == [
        {
            "phrase": "Angel of Mercy",
            "kind": "value",
            "tbl": "cards",
            "col": "name",
            "stored_value": "Angel of Mercy",
            "database_name": "card_games",
            "schema_name": "main",
            "table_id": "cards-table",
            "column_id": "cards-name",
        }
    ]
    assert (
        result["path_state"]["value_grounding_evidence_columns"][0]["column_id"]
        == "cards-type"
    )


def test_agent_does_not_expand_empty_schema_ids_to_every_database(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fetch = MagicMock()
    monkeypatch.setattr(
        value_grounding, "fetch_schema_ids_for_database", lambda _db: []
    )
    monkeypatch.setattr(value_grounding, "fetch_schemas_by_ids", fetch)

    result = ValueGroundingAgent().execute(
        {
            "llm": object(),
            "initial_question": "question",
            "connectors": [],
            "path_state": {"target_db": "missing"},
        }
    )

    fetch.assert_not_called()
    assert result["value_anchors"] == []


class _QueuedExecutor:
    def __init__(self, results: list[dict]) -> None:
        self.results = iter(results)
        self.calls: list[tuple[str, str]] = []
        self.budget_left = 20

    def run(self, sql: str, purpose: str = "") -> dict:
        self.calls.append((purpose, sql))
        self.budget_left -= 1
        return next(self.results)


def _result(*values: str) -> dict:
    return {
        "ok": True,
        "rows": [{"value": value} for value in values],
    }


def test_live_lookup_prefers_exact_then_normalized_then_contains() -> None:
    connector = SimpleNamespace(
        dialect="sqlite",
        qualify=lambda schema, table: f'"{schema}"."{table}"',
    )
    exact = _QueuedExecutor([_result("Active")])
    assert lookup_text_value(
        exact,
        connector,
        schema_name="main",
        table_name="accounts",
        column_name="status",
        value="Active",
    ) == ("Active", "value")
    assert [purpose for purpose, _ in exact.calls] == ["value_anchor_exact"]

    normalized = _QueuedExecutor([_result(), _result("ACTIVE")])
    assert lookup_text_value(
        normalized,
        connector,
        schema_name="main",
        table_name="accounts",
        column_name="status",
        value="active",
    ) == ("ACTIVE", "value")
    assert [purpose for purpose, _ in normalized.calls] == [
        "value_anchor_exact",
        "value_anchor_normalized",
    ]

    contains = _QueuedExecutor([_result(), _result(), _result("North_100% Region")])
    assert lookup_text_value(
        contains,
        connector,
        schema_name="main",
        table_name="accounts",
        column_name="region",
        value="North_100%",
    ) == ("North_100% Region", "contains")
    assert [purpose for purpose, _ in contains.calls] == [
        "value_anchor_exact",
        "value_anchor_normalized",
        "value_anchor_contains",
    ]
    assert "\\_" in contains.calls[-1][1]
    assert "\\%" in contains.calls[-1][1]
