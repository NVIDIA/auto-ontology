# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from gsf.retrieval.text_to_sql.agents.sql_execution import SQLExecutionAgent
from gsf.retrieval.text_to_sql.base import BaseAgent, AgentExecutionError, agent_wrapper
from gsf.retrieval.text_to_sql.text_to_sql_graph import route_decision


class _InvalidAgent(BaseAgent):
    def __init__(self, failure_decision: str | None = None):
        super().__init__("invalid_fixture", failure_decision=failure_decision)

    def validate_input(self, state):
        return False

    def execute(self, state):
        raise AssertionError("invalid input must not execute")


def _state(decision: str = "intent_valid") -> dict:
    return {
        "decision": decision,
        "path_state": {},
        "connectors": [],
        "messages": [],
        "domain_rules": [],
    }


def test_validation_failure_never_returns_an_empty_state_update():
    try:
        agent_wrapper(_InvalidAgent())(_state())
    except AgentExecutionError as exc:
        assert "invalid_fixture" in str(exc)
    else:
        raise AssertionError("missing failure decision must fail explicitly")


def test_execution_validation_failure_replaces_stale_intent_decision():
    state = _state()
    update = agent_wrapper(SQLExecutionAgent())(state)
    merged = {**state, **update}

    assert update["decision"] == "invalid_sql"
    assert route_decision(merged) == "invalid_sql"
    assert update["path_state"]["error"]["agent"] == "sql_execution"


def test_execution_exception_replaces_stale_intent_decision(monkeypatch):
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
