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
            "description": "Printed card name",
            "sample_values": ["Angel of Mercy", "Counterspell"],
        },
        {
            "database_name": "card_games",
            "table_schema": "main",
            "table_name": "cards",
            "table_id": "cards-table",
            "column_name": "type",
            "column_id": "cards-type",
            "data_type": "TEXT",
            "description": "Card type",
            "sample_values": ["Creature"],
        },
        {
            "database_name": "card_games",
            "table_schema": "main",
            "table_name": "cards",
            "table_id": "cards-table",
            "column_name": "release_year",
            "column_id": "cards-year",
            "data_type": "INTEGER",
            "description": "Release year",
            "sample_values": "[2001]",
        },
        {
            "database_name": "card_games",
            "table_schema": "main",
            "table_name": "cards",
            "table_id": "cards-table",
            "column_name": "released_at",
            "column_id": "cards-released-at",
            "data_type": "DATE",
            "description": "Release date",
            "sample_values": ["2001-01-01"],
        },
        {
            "database_name": "card_games",
            "table_schema": "main",
            "table_name": "cards",
            "table_id": "cards-table",
            "column_name": "is_foil",
            "column_id": "cards-is-foil",
            "data_type": "BOOLEAN",
            "description": "Whether the card is foil",
            "sample_values": [True, False],
        },
    ]


class _FakeProbeExecutor:
    def __init__(self, _connector) -> None:
        self.budget_left = 40

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None


def test_agent_uses_original_inputs_schema_hints_and_first_matching_suspect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, str] = {}
    probes: list[tuple[str, str]] = []
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
        probes.append((kwargs["match_kind"], kwargs["column_name"]))
        if kwargs["match_kind"] == "contains" and kwargs["column_name"] == "name":
            return "Angel of Mercy card"
        if kwargs["match_kind"] == "equal" and kwargs["column_name"] == "release_year":
            return "2001"
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
        '"name":"name","data_type":"TEXT","description":"Printed card name"'
        in captured["prompt"]
    )
    assert (
        '"name":"release_year","data_type":"INTEGER",'
        '"description":"Release year"' in captured["prompt"]
    )
    assert '"name":"released_at","data_type":"DATE"' in captured["prompt"]
    assert '"name":"is_foil","data_type":"BOOLEAN"' in captured["prompt"]
    assert '"samples"' not in captured["prompt"]
    assert probes == [
        ("equal", "type"),
        ("equal", "name"),
        ("contains", "type"),
        ("contains", "name"),
        ("equal", "release_year"),
    ]
    assert result["value_anchors"] == [
        {
            "phrase": "Angel of Mercy",
            "kind": "contains",
            "tbl": "cards",
            "col": "name",
            "stored_value": "Angel of Mercy card",
            "database_name": "card_games",
            "schema_name": "main",
            "table_id": "cards-table",
            "column_id": "cards-name",
        },
        {
            "phrase": "2001",
            "kind": "value",
            "tbl": "cards",
            "col": "release_year",
            "stored_value": "2001",
            "database_name": "card_games",
            "schema_name": "main",
            "table_id": "cards-table",
            "column_id": "cards-year",
        },
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


def test_live_lookup_supports_case_insensitive_equal_and_contains_probes() -> None:
    connector = SimpleNamespace(
        dialect="sqlite",
        qualify=lambda schema, table: f'"{schema}"."{table}"',
    )
    exact = _QueuedExecutor([_result("Active")])
    assert (
        lookup_text_value(
            exact,
            connector,
            schema_name="main",
            table_name="accounts",
            column_name="status",
            value="Active",
            match_kind="equal",
        )
        == "Active"
    )
    assert [purpose for purpose, _ in exact.calls] == ["value_anchor_equal"]
    assert "LOWER(TRIM(" in exact.calls[0][1]
    assert " = LOWER(TRIM(" in exact.calls[0][1]
    assert 'CAST("status" AS TEXT)' in exact.calls[0][1]

    normalized = _QueuedExecutor([_result(" ACTIVE ")])
    assert (
        lookup_text_value(
            normalized,
            connector,
            schema_name="main",
            table_name="accounts",
            column_name="status",
            value="active",
            match_kind="equal",
        )
        == " ACTIVE "
    )
    assert [purpose for purpose, _ in normalized.calls] == ["value_anchor_equal"]

    contains = _QueuedExecutor([_result("North_100% Region")])
    assert (
        lookup_text_value(
            contains,
            connector,
            schema_name="main",
            table_name="accounts",
            column_name="region",
            value="North_100%",
            match_kind="contains",
        )
        == "North_100% Region"
    )
    assert [purpose for purpose, _ in contains.calls] == ["value_anchor_contains"]
    assert "!_" in contains.calls[-1][1]
    assert "!%" in contains.calls[-1][1]
    assert "ESCAPE '!'" in contains.calls[-1][1]
