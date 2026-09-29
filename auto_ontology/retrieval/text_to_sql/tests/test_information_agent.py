# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structured information-agent loop and terminal response contract."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from auto_ontology.retrieval.text_to_sql.agents import information_agent
from auto_ontology.retrieval.text_to_sql.agents.information_agent import (
    InformationActionType,
    InformationAgent,
    InformationAgentAction,
    InformationToolArguments,
)
from auto_ontology.retrieval.text_to_sql.agents.information_tools import (
    InformationToolName,
)


def _state() -> dict[str, Any]:
    return {
        "llm": MagicMock(),
        "initial_question": "How is revenue calculated?",
        "messages": [],
        "path_state": {
            "question_type": "information",
            "normalized_question": "How is revenue calculated?",
            "entities": ["revenue"],
            "target_db": "sales",
            "retrieved_column_attributes": [
                {
                    "id": "attr-1",
                    "name": "Revenue",
                    "table_id": "table-1",
                    "table_name": "orders",
                    "source_column": "revenue",
                }
            ],
            "retrieved_sql_attributes": [],
            "retrieved_custom_analyses": [],
        },
    }


def _tool_action() -> InformationAgentAction:
    return InformationAgentAction(
        action=InformationActionType.TOOL,
        tool=InformationToolName.GET_COLUMN,
        arguments=InformationToolArguments(
            column_name="revenue",
            table_name="orders",
            database_name="sales",
        ),
    )


def _answer_action() -> InformationAgentAction:
    return InformationAgentAction(
        action=InformationActionType.ANSWER,
        answer=(
            "Revenue is gross sales minus refunds in `orders.revenue`. "
            "The orders table connects to customers."
        ),
        source_ids=["attr-1", "column-1", "table-1"],
    )


def test_agent_uses_multiple_tools_and_emits_terminal_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actions = iter(
        [
            _tool_action(),
            InformationAgentAction(
                action=InformationActionType.TOOL,
                tool=InformationToolName.GET_TABLE_SEMANTIC_FKS,
                arguments=InformationToolArguments(table_id="table-1"),
            ),
            _answer_action(),
        ]
    )
    monkeypatch.setattr(
        information_agent,
        "invoke_with_structured_output",
        lambda *_args, **_kwargs: next(actions),
    )

    def run_tool(
        tool: InformationToolName, _arguments: dict, **_kwargs: object
    ) -> dict:
        if tool == InformationToolName.GET_COLUMN:
            return {
                "ok": True,
                "data": {
                    "column": {"id": "column-1"},
                    "table": {"id": "table-1"},
                },
            }
        return {
            "ok": True,
            "data": {
                "table": {"id": "table-1"},
                "semantic_foreign_keys": [],
            },
        }

    monkeypatch.setattr(information_agent, "run_information_tool", run_tool)

    result = InformationAgent().execute(_state())

    assert len(result["path_state"]["information_tool_calls"]) == 2
    assert result["path_state"]["information_sources"] == [
        "attr-1",
        "column-1",
        "table-1",
    ]
    final_response = result["path_state"]["final_response"]
    assert "gross sales minus refunds" in final_response["response"]
    assert final_response["sql_code"] == ""
    assert final_response["sql_columns"] == ["column-1"]
    assert final_response["sql_response_from_db"] is None
    assert result["messages"][-1].content == final_response["response"]


def test_duplicate_tool_call_is_not_executed_twice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actions = iter([_tool_action(), _tool_action(), _answer_action()])
    monkeypatch.setattr(
        information_agent,
        "invoke_with_structured_output",
        lambda *_args, **_kwargs: next(actions),
    )
    tool_calls: list[object] = []
    monkeypatch.setattr(
        information_agent,
        "run_information_tool",
        lambda *_args, **_kwargs: (
            tool_calls.append(object()) or {"ok": True, "data": {"id": "column-1"}}
        ),
    )

    result = InformationAgent().execute(_state())

    assert len(tool_calls) == 1
    assert len(result["path_state"]["information_tool_calls"]) == 1


def test_llm_failure_returns_safe_non_sql_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        information_agent,
        "invoke_with_structured_output",
        lambda *_args, **_kwargs: None,
    )

    result = InformationAgent().execute(_state())

    response = result["path_state"]["final_response"]
    assert "couldn't complete the metadata lookup safely" in response["response"]
    assert "Revenue" in response["response"]
    assert response["sql_code"] == ""
    assert result["path_state"]["information_tool_calls"] == []


def test_tool_failure_is_observed_and_answered_safely(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actions = iter(
        [
            _tool_action(),
            InformationAgentAction(
                action=InformationActionType.ANSWER,
                answer="Revenue metadata could not be retrieved.",
                source_ids=[],
            ),
        ]
    )
    monkeypatch.setattr(
        information_agent,
        "invoke_with_structured_output",
        lambda *_args, **_kwargs: next(actions),
    )
    monkeypatch.setattr(
        information_agent,
        "run_information_tool",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("offline")),
    )

    result = InformationAgent().execute(_state())

    assert result["path_state"]["information_tool_calls"] == [
        {
            "tool": "get_column",
            "arguments": {
                "table_name": "orders",
                "column_name": "revenue",
                "database_name": "sales",
            },
            "ok": False,
        }
    ]
    assert result["path_state"]["final_response"]["sql_code"] == ""


def test_first_prompt_is_grounded_with_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompts: list[str] = []

    def answer(_llm: object, messages: list[Any], _model: object) -> object:
        prompts.append(str(messages[0].content))
        return _answer_action()

    monkeypatch.setattr(
        information_agent,
        "invoke_with_structured_output",
        answer,
    )

    InformationAgent().execute(_state())

    assert '"question": "How is revenue calculated?"' in prompts[0]
    assert '"entities": ["revenue"]' in prompts[0]
    assert '"dataset": "sales"' in prompts[0]
    assert '"id": "attr-1"' in prompts[0]
    assert '"table_id": "table-1"' in prompts[0]
    assert "get_term" in prompts[0]
    assert "get_column_attribute" in prompts[0]
    assert "get_sql_attribute" in prompts[0]
    assert "list_terms" in prompts[0]
    assert "search_semantic_layer" in prompts[0]


def test_semantic_search_tool_receives_state_retriever(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retriever = object()
    actions = iter(
        [
            InformationAgentAction(
                action=InformationActionType.TOOL,
                tool=InformationToolName.SEARCH_SEMANTIC_LAYER,
                arguments=InformationToolArguments(
                    query="revenue",
                    labels=["Term", "ColumnAttribute"],
                    limit=5,
                ),
            ),
            _answer_action(),
        ]
    )
    monkeypatch.setattr(
        information_agent,
        "invoke_with_structured_output",
        lambda *_args, **_kwargs: next(actions),
    )
    captured: dict[str, object] = {}

    def run_tool(
        _tool: InformationToolName,
        _arguments: dict,
        **kwargs: object,
    ) -> dict:
        captured.update(kwargs)
        return {"ok": True, "data": {"matches": []}}

    monkeypatch.setattr(information_agent, "run_information_tool", run_tool)
    state = _state()
    state["semantic_retriever"] = retriever

    InformationAgent().execute(state)

    assert captured["semantic_retriever"] is retriever


def test_reasoning_limit_stops_repeated_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        information_agent,
        "invoke_with_structured_output",
        lambda *_args, **_kwargs: _tool_action(),
    )
    tool_calls: list[object] = []
    monkeypatch.setattr(
        information_agent,
        "run_information_tool",
        lambda *_args, **_kwargs: (
            tool_calls.append(object()) or {"ok": True, "data": {}}
        ),
    )

    result = InformationAgent().execute(_state())

    assert len(tool_calls) == 1
    assert "reasoning limit" in result["path_state"]["final_response"]["response"]
