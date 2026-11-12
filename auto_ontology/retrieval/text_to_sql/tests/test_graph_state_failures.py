# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest

from auto_ontology.retrieval.text_to_sql.base import (
    AgentExecutionError,
    BaseAgent,
    agent_wrapper,
)
from auto_ontology.retrieval.text_to_sql.state import AgentState


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


def _state(decision: str = "valid_sql") -> dict:
    return {
        "decision": decision,
        "path_state": {},
        "connectors": [],
        "messages": [],
    }


def test_validation_failure_never_returns_an_empty_state_update() -> None:
    try:
        agent_wrapper(_InvalidAgent())(_state())
    except AgentExecutionError as exc:
        assert "invalid_fixture" in str(exc)
    else:
        raise AssertionError("missing failure decision must fail explicitly")


def test_validation_failure_replaces_stale_decision() -> None:
    update = agent_wrapper(_InvalidAgent(failure_decision="unconstructable"))(_state())

    assert update["decision"] == "unconstructable"
    assert update["path_state"]["error"]["agent"] == "invalid_fixture"


def test_malformed_result_without_failure_transition_raises() -> None:
    with pytest.raises(AgentExecutionError, match="returned non-dict"):
        agent_wrapper(_MalformedAgent())(_state())
