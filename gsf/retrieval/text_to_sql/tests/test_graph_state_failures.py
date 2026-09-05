# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest

from gsf.retrieval.text_to_sql.agents.sql_execution import SQLExecutionAgent
from gsf.retrieval.text_to_sql.base import AgentExecutionError, BaseAgent, agent_wrapper
from gsf.retrieval.text_to_sql.state import AgentState
from gsf.retrieval.text_to_sql.text_to_sql_graph import route_decision


class _InvalidAgent(BaseAgent):
    def __init__(self, failure_decision: str | None = None) -> None:
        super().__init__("invalid_fixture", failure_decision=failure_decision)

    def validate_input(self, state: AgentState) -> bool:
        return False

    def execute(self, state: AgentState) -> dict[str, Any]:
        raise AssertionError("invalid input must not execute")


class _MalformedAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__("malformed_fixture")

    def execute(self, state: AgentState) -> dict[str, Any]:
        return None  # type: ignore[return-value]


def _state(decision: str = "intent_valid") -> dict:
    return {
        "decision": decision,
        "path_state": {},
        "connectors": [],
        "messages": [],
        "domain_rules": [],
    }


def test_validation_failure_never_returns_an_empty_state_update() -> None:
    try:
        agent_wrapper(_InvalidAgent())(_state())
    except AgentExecutionError as exc:
        assert "invalid_fixture" in str(exc)
    else:
        raise AssertionError("missing failure decision must fail explicitly")


def test_execution_validation_failure_replaces_stale_intent_decision() -> None:
    state = _state()
    update = agent_wrapper(SQLExecutionAgent())(state)
    merged = {**state, **update}

    assert update["decision"] == "invalid_sql"
    assert route_decision(merged) == "invalid_sql"
    assert update["path_state"]["error"]["agent"] == "sql_execution"


def test_execution_exception_replaces_stale_intent_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = SQLExecutionAgent()
    monkeypatch.setattr(agent, "validate_input", lambda state: True)
    monkeypatch.setattr(
        agent,
        "execute",
        lambda state: (_ for _ in ()).throw(RuntimeError("fixture failure")),
    )

    state = _state()
    update = agent_wrapper(agent)(state)
    merged = {**state, **update}

    assert update["decision"] == "invalid_sql"
    assert route_decision(merged) == "invalid_sql"
    assert update["path_state"]["error"] == {
        "type": "RuntimeError",
        "message": "fixture failure",
        "agent": "sql_execution",
    }


def test_malformed_result_without_failure_transition_raises() -> None:
    with pytest.raises(AgentExecutionError, match="returned non-dict"):
        agent_wrapper(_MalformedAgent())(_state())
