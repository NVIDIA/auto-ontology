# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from auto_ontology.retrieval.text_to_sql import pipeline
from auto_ontology.retrieval.text_to_sql.agents.sql_execution import QueryResponse
from auto_ontology.retrieval.text_to_sql.react import agent as agent_module
from auto_ontology.retrieval.text_to_sql.react.loop import EpisodeResult, Submission


def _state() -> dict:
    return {
        "initial_question": "How much revenue came from active customers?",
        "evidence": "",
        "llm": SimpleNamespace(),
        "connectors": [SimpleNamespace(database_name="shop", dialect="sqlite")],
        "semantic_retriever": object(),
        "path_state": {"target_db": "shop"},
    }


def _episode(submission: Submission | None, reason: str) -> EpisodeResult:
    trajectory = [
        {
            "role": "ai",
            "tool_calls": [{"name": "retrieve", "args": {"query": "revenue"}}],
        }
    ]
    return EpisodeResult(submission, reason, 1, trajectory)


def test_an_answer_is_handed_to_formatting(monkeypatch: pytest.MonkeyPatch) -> None:
    db_response = QueryResponse(result=['[{"total": 160.0}]'], sliced=False)
    submission = Submission("SELECT 1", "Revenue of active customers.", "", db_response)
    monkeypatch.setattr(
        agent_module, "run_episode", lambda *a, **k: _episode(submission, "submitted")
    )

    result = agent_module.ReactSQLAgent().execute(_state())

    path_state = result["path_state"]
    assert result["decision"] == "valid_sql"
    assert path_state["sql_generation_result"].sql_code == "SELECT 1"
    assert (
        path_state["sql_generation_result"].response == "Revenue of active customers."
    )
    assert path_state["sql_response_from_db"] == ['[{"total": 160.0}]']
    assert path_state["thoughts_log"] == [
        {"node": "react_sql_agent", "text": "Searching for: revenue"}
    ]
    assert pipeline.route_react(result) == "format_and_respond"


def test_no_answer_is_reported_as_unconstructable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        agent_module, "run_episode", lambda *a, **k: _episode(None, "budget_exhausted")
    )

    result = agent_module.ReactSQLAgent().execute(_state())

    assert result["decision"] == "unconstructable"
    assert result["path_state"]["react_end_reason"] == "budget_exhausted"
    assert result["path_state"]["unconstructable_explanation"]
    assert pipeline.route_react(result) == "unconstructable_sql_response"


def test_the_pipeline_has_only_the_agent_and_its_two_outcomes() -> None:
    assert set(pipeline.create_steps()) == {
        "react_sql_agent",
        "format_and_respond",
        "unconstructable_sql_response",
    }
