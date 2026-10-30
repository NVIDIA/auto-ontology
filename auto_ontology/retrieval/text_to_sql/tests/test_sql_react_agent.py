# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ReAct agent's parity with the deep agent.

The ReAct arm is only comparable to the deep arm while the model sees the same
thing in both, so these pin what it sends: the final tool, the reply to it, and
the budgets it runs under.
"""

from __future__ import annotations

from langchain.agents.structured_output import OutputToolBinding, _SchemaSpec

from auto_ontology.retrieval.text_to_sql.models import SQLGenerationModel
from auto_ontology.retrieval.text_to_sql.sql_agent import config, react


def test_final_tool_is_the_deep_agents_structured_answer() -> None:
    (schema,) = react._TERMINAL_SCHEMAS
    deep_tool = OutputToolBinding.from_schema_spec(_SchemaSpec(SQLGenerationModel)).tool

    assert schema["function"]["name"] == deep_tool.name == "SQLGenerationModel"
    assert schema["function"]["description"] == deep_tool.description
    assert set(schema["function"]["parameters"]["properties"]) == set(
        SQLGenerationModel.model_fields
    )


def test_answer_reply_matches_langchains() -> None:
    args = {"thought": "t", "sql_code": "SELECT 1", "response": "r"}

    assert react._answer_reply(args) == (
        f"Returning structured response: {SQLGenerationModel(**args)}"
    )


def test_answer_reply_falls_back_on_an_invalid_answer() -> None:
    assert react._answer_reply({"sql_code": "SELECT 1"}) == "Recorded."


def test_budgets_default_to_the_full_dev_run(monkeypatch) -> None:
    monkeypatch.delenv("SQL_AGENT_MAX_TURNS", raising=False)
    monkeypatch.delenv("SQL_AGENT_DEADLINE_S", raising=False)

    assert config.react_max_turns() == 30
    assert config.react_deadline_s() == 600.0
    # The loop's own budgets are untouched.
    assert config.max_turns() == 12
    assert config.deadline_s() == 120.0


def test_budgets_follow_the_shared_overrides(monkeypatch) -> None:
    monkeypatch.setenv("SQL_AGENT_MAX_TURNS", "8")
    monkeypatch.setenv("SQL_AGENT_DEADLINE_S", "45")

    assert config.react_max_turns() == 8
    assert config.react_deadline_s() == 45.0
