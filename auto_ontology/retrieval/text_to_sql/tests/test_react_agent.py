# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from auto_ontology.retrieval.text_to_sql import text_to_sql_graph
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
    assert text_to_sql_graph.route_react(result) == "format_and_respond"


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
    assert text_to_sql_graph.route_react(result) == "unconstructable_sql_response"


def test_flag_routes_fresh_questions_to_the_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEXT_TO_SQL_REACT", "true")
    graph = text_to_sql_graph.create_graph()

    assert "react_sql_agent" in graph.nodes
    assert (
        text_to_sql_graph._react_entry_router_fn({"path_state": {}})
        == "react_sql_agent"
    )
    assert (
        text_to_sql_graph._react_entry_router_fn(
            {"path_state": {"_resume_from": "reconstruct_sql"}}
        )
        == "reconstruct_sql"
    )


def test_flag_off_keeps_the_fixed_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TEXT_TO_SQL_REACT", raising=False)

    assert "react_sql_agent" not in text_to_sql_graph.create_graph().nodes
