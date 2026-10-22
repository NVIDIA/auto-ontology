# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structured-loop and downstream-contract tests for semantic preparation."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from auto_ontology.retrieval.text_to_sql.agents import semantic_preparation_agent
from auto_ontology.retrieval.text_to_sql.agents.semantic_preparation_agent import (
    SemanticPreparationAction,
    SemanticPreparationActionType,
    SemanticPreparationAgent,
    SemanticPreparationResult,
    SemanticPreparationToolArguments,
)
from auto_ontology.retrieval.text_to_sql.agents.semantic_preparation_tools import (
    SemanticPreparationToolName,
)
from auto_ontology.retrieval.text_to_sql.agents.sql_from_semantic import (
    SQLFromCandidatesAgent,
)
from auto_ontology.semantic.constants import LABEL_COLUMN_ATTRIBUTE


def _state() -> dict[str, Any]:
    connector = MagicMock()
    connector.database_name = "cards"
    connector.dialect = "sqlite"
    return {
        "llm": MagicMock(),
        "initial_question": "Who illustrated Angel of Mercy?",
        "evidence": "",
        "messages": [],
        "connectors": [connector],
        "semantic_retriever": MagicMock(),
        "data_retriever": MagicMock(),
        "glossary": [{"name": "Illustrator", "description": "Card artist"}],
        "path_state": {
            "question_type": "descriptive",
            "target_db": "cards",
        },
    }


def _tool_action(
    tool: SemanticPreparationToolName,
    **arguments: Any,
) -> SemanticPreparationAction:
    return SemanticPreparationAction(
        action=SemanticPreparationActionType.TOOL,
        tool=tool,
        arguments=SemanticPreparationToolArguments(**arguments),
    )


def _complete_action() -> SemanticPreparationAction:
    return SemanticPreparationAction(
        action=SemanticPreparationActionType.COMPLETE,
        result=SemanticPreparationResult(
            normalized_question="Who illustrated Angel of Mercy?",
            entities=["card artist", "card name"],
            subject="card",
            target_db="cards",
            used_glossary_names=["Illustrator"],
            column_attribute_ids=["attr-artist", "attr-name", "invented"],
            table_ids=["table-cards"],
            primary_attribute_id="attr-name",
            reasoning="The verified card name anchors the requested artist.",
        ),
    )


def test_agent_grounded_value_becomes_attribute_and_anchor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actions = iter(
        [
            _tool_action(
                SemanticPreparationToolName.SEARCH_SEMANTIC_LAYER,
                query="card artist and card name",
                database_name="cards",
            ),
            _tool_action(
                SemanticPreparationToolName.PROBE_VALUE,
                value="Angel of Mercy",
                attribute_ids=["attr-name"],
                database_name="cards",
            ),
            _complete_action(),
        ]
    )
    monkeypatch.setattr(
        semantic_preparation_agent,
        "invoke_with_structured_output",
        lambda *_args, **_kwargs: next(actions),
    )

    def run_tool(
        tool: SemanticPreparationToolName,
        _arguments: dict[str, Any],
        **_: Any,
    ) -> dict[str, Any]:
        if tool == SemanticPreparationToolName.SEARCH_SEMANTIC_LAYER:
            return {
                "ok": True,
                "data": {
                    "matches": [
                        {
                            "id": "attr-artist",
                            "label": LABEL_COLUMN_ATTRIBUTE,
                            "text": "Card artist",
                        },
                        {
                            "id": "attr-name",
                            "label": LABEL_COLUMN_ATTRIBUTE,
                            "text": "Card name",
                        },
                        {
                            "id": "table-cards",
                            "label": "Table",
                            "name": "cards",
                        },
                    ]
                },
            }
        return {
            "ok": True,
            "data": {
                "input_value": "Angel of Mercy",
                "stored_value": "Angel of Mercy",
                "match_type": "equal",
                "column_attribute_id": "attr-name",
                "column_id": "col-name",
                "column_name": "name",
                "table_id": "table-cards",
                "table_name": "cards",
                "schema_name": "main",
                "database_name": "cards",
            },
        }

    monkeypatch.setattr(
        semantic_preparation_agent,
        "run_semantic_preparation_tool",
        run_tool,
    )
    prepared_inputs: list[dict[str, Any]] = []

    def prepare(_self: object, state: dict[str, Any]) -> dict[str, Any]:
        prepared_inputs.append(state)
        return {
            "path_state": {
                **state["path_state"],
                "primary_attribute": {"id": "attr-name"},
                "attribute_join_paths": [],
                "relevant_tables": [{"id": "table-cards", "name": "cards"}],
                "relevant_queries": [],
                "custom_analyses": [],
                "sql_attributes": [],
                "term_synonyms": {},
            },
            "value_anchors": state["value_anchors"],
        }

    monkeypatch.setattr(
        semantic_preparation_agent.CandidatePreparationAgent,
        "execute",
        prepare,
    )

    result = SemanticPreparationAgent().execute(_state())

    prepared_state = prepared_inputs[0]
    path_state = prepared_state["path_state"]
    assert [hit["id"] for hit in path_state["retrieved_column_attributes"]] == [
        "attr-artist",
        "attr-name",
    ]
    assert "invented" not in {
        hit["id"] for hit in path_state["retrieved_column_attributes"]
    }
    assert path_state["semantic_preparation_primary_attribute_id"] == "attr-name"
    assert prepared_state["value_anchors"] == [
        {
            "phrase": "Angel of Mercy",
            "kind": "value",
            "tbl": "cards",
            "col": "name",
            "stored_value": "Angel of Mercy",
            "n_rows": "1+",
            "column_attribute_id": "attr-name",
            "match_type": "equal",
        }
    ]
    assert result["path_state"]["primary_attribute"]["id"] == "attr-name"
    assert result["glossary"][0]["name"] == "Illustrator"
    assert SQLFromCandidatesAgent().validate_input(result) is True


def test_duplicate_tool_call_runs_once_and_failure_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    action = _tool_action(
        SemanticPreparationToolName.SEARCH_TABLES,
        query="cards",
        database_name="cards",
    )
    actions = iter([action, action, None])
    monkeypatch.setattr(
        semantic_preparation_agent,
        "invoke_with_structured_output",
        lambda *_args, **_kwargs: next(actions),
    )
    calls: list[object] = []
    monkeypatch.setattr(
        semantic_preparation_agent,
        "run_semantic_preparation_tool",
        lambda *_args, **_kwargs: (
            calls.append(object())
            or {
                "ok": True,
                "data": {
                    "tables": [
                        {
                            "id": "table-cards",
                            "label": "Table",
                            "name": "cards",
                        }
                    ]
                },
            }
        ),
    )
    monkeypatch.setattr(
        semantic_preparation_agent.CandidatePreparationAgent,
        "execute",
        lambda _self, state: {
            "path_state": {
                **state["path_state"],
                "primary_attribute": None,
                "attribute_join_paths": [],
                "relevant_tables": [],
            }
        },
    )

    result = SemanticPreparationAgent().execute(_state())

    assert len(calls) == 1
    assert result["path_state"]["normalized_question"]
    assert len(result["path_state"]["semantic_preparation_tool_calls"]) == 1
