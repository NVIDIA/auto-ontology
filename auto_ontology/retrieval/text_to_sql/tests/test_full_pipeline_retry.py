# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Full-pipeline retries restart from the immutable request boundary."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, cast

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from auto_ontology.retrieval.text_to_sql import main
from auto_ontology.retrieval.text_to_sql.state import AgentState, TextToSQLPayload


def _initial_state() -> AgentState:
    base_messages = [
        SystemMessage(content="base system"),
        HumanMessage(content="original question"),
    ]
    baseline_path = {
        "processing_question": "original question",
        "target_db": "test_db",
        "caller_context": "keep",
    }
    return cast(
        AgentState,
        {
            "llm": object(),
            "reasoning_llm": object(),
            "initial_question": "original question",
            "evidence": "original evidence",
            "sql_examples": [{"question": "original", "sql": "SELECT 1"}],
            "value_anchors": [{"phrase": "active"}],
            "calculation_only": False,
            "shorten_answer": False,
            "validate_sql_values": True,
            "sql_value_validation_cache": {
                "queries": {"SELECT status FROM accounts": {"ok": True}}
            },
            "messages": list(base_messages),
            "decision": "",
            "connectors": [],
            "path_state": dict(baseline_path),
            "data_retriever": object(),
            "semantic_retriever": object(),
            "domain_rules": [],
            "glossary": [{"name": "ARR"}],
            "restore_from": {
                "initial_question": "original question",
                "evidence": "original evidence",
                "messages": list(base_messages),
                "sql_examples": [{"question": "original", "sql": "SELECT 1"}],
                "value_anchors": [{"phrase": "active"}],
                "glossary": [{"name": "ARR"}],
                "path_state": dict(baseline_path),
            },
            "full_pipeline_attempt": 1,
        },
    )


def _unconstructable_update(fallback_sql: str = "") -> dict[str, Any]:
    path_state: dict[str, Any] = {
        "normalized_question": "mutated question",
        "failed_attempts": [{"sql": "SELECT bad"}],
        "sql_generation_result": object(),
        "final_response": {
            "response": "SQL cannot be constructed.",
            "sql_code": "",
        },
    }
    if fallback_sql:
        path_state["last_intent_rejected_sql"] = fallback_sql
    return {
        "unconstructable_sql_response": {
            "evidence": "mutated evidence",
            "sql_examples": [],
            "glossary": [],
            "messages": [AIMessage(content="failed")],
            "decision": "unconstructable",
            "path_state": path_state,
        }
    }


def _success_update() -> dict[str, Any]:
    return {
        "format_and_respond": {
            "path_state": {
                "sql_code": "SELECT 42",
                "final_response": {
                    "response": "Success.",
                    "sql_code": "SELECT 42",
                },
            }
        }
    }


class _AttemptApp:
    def __init__(
        self,
        outcomes: list[str],
    ) -> None:
        self.outcomes = outcomes
        self.states: list[AgentState] = []

    def stream(
        self,
        state: AgentState,
        stream_mode: list[str] | None = None,
        config: dict[str, Any] | None = None,
    ) -> Iterator[Any]:
        state_at_entry = dict(state)
        state_at_entry["messages"] = list(state["messages"])
        state_at_entry["path_state"] = dict(state["path_state"])
        self.states.append(cast(AgentState, state_at_entry))
        outcome = self.outcomes[len(self.states) - 1]
        if outcome == "success":
            update = _success_update()
        elif outcome == "failure_with_fallback":
            update = _unconstructable_update("SELECT fallback")
        elif outcome == "information":
            update = {
                "information_agent": {
                    "path_state": {
                        "final_response": {
                            "response": "Metadata answer.",
                            "sql_code": "",
                        }
                    }
                }
            }
        else:
            update = _unconstructable_update()

        if stream_mode:
            node_name = next(iter(update))
            yield ("custom", {"type": main.NODE_START_EVENT, "node": node_name})
            yield ("updates", update)
        else:
            yield update


def _payload() -> TextToSQLPayload:
    return cast(TextToSQLPayload, {"question": "original question"})


def test_restore_discards_failed_attempt_mutations_and_messages() -> None:
    state = _initial_state()
    state.update(_unconstructable_update()["unconstructable_sql_response"])
    state["path_state"].update(
        {
            "_resume_from": "reconstruct_sql",
            "error": "bad SQL",
            "thoughts_log": [{"node": "reconstruct_sql", "text": "failed"}],
        }
    )

    restored = main._restore_for_full_pipeline_retry(state)

    assert restored["full_pipeline_attempt"] == 2
    assert restored["initial_question"] == "original question"
    assert restored["evidence"] == "original evidence"
    assert [message.content for message in restored["messages"]] == [
        "base system",
        "original question",
    ]
    assert restored["sql_examples"] == [{"question": "original", "sql": "SELECT 1"}]
    assert restored["glossary"] == [{"name": "ARR"}]
    assert restored["decision"] == ""
    assert restored["validate_sql_values"] is True
    assert restored["sql_value_validation_cache"] == {
        "queries": {"SELECT status FROM accounts": {"ok": True}}
    }
    assert restored["path_state"] == {
        "processing_question": "original question",
        "target_db": "test_db",
        "caller_context": "keep",
    }


def test_stream_retries_once_then_returns_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _AttemptApp(["failure", "success"])
    monkeypatch.setattr(main, "_build_state", lambda _payload: _initial_state())
    monkeypatch.setattr(main, "app", app)

    events = list(main.stream_agent_response(_payload()))

    results = [event for event in events if event["type"] == "result"]
    assert results == [
        {
            "type": "result",
            "answer": {"response": "Success.", "sql_code": "SELECT 42"},
        }
    ]
    assert [state["full_pipeline_attempt"] for state in app.states] == [1, 2]
    assert app.states[1]["initial_question"] == "original question"
    assert app.states[1]["evidence"] == "original evidence"
    assert [message.content for message in app.states[1]["messages"]] == [
        "base system",
        "original question",
    ]
    assert "failed_attempts" not in app.states[1]["path_state"]


def test_stream_stops_after_two_unconstructable_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _AttemptApp(["failure", "failure"])
    monkeypatch.setattr(main, "_build_state", lambda _payload: _initial_state())
    monkeypatch.setattr(main, "app", app)

    events = list(main.stream_agent_response(_payload()))

    assert len(app.states) == 2
    assert events[-1]["type"] == "result"
    assert events[-1]["answer"]["sql_code"] == ""
    assert events[-1]["answer"]["response"] == "SQL cannot be constructed."


@pytest.mark.parametrize("outcome", ["success", "information"])
def test_stream_does_not_retry_success_or_legitimate_no_sql(
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
) -> None:
    app = _AttemptApp([outcome])
    monkeypatch.setattr(main, "_build_state", lambda _payload: _initial_state())
    monkeypatch.setattr(main, "app", app)

    events = list(main.stream_agent_response(_payload()))

    assert len(app.states) == 1
    assert events[-1]["type"] == "result"


def test_response_with_state_uses_the_same_full_retry_cycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _AttemptApp(["failure", "success"])
    monkeypatch.setattr(main, "_build_state", lambda _payload: _initial_state())
    monkeypatch.setattr(main, "app", app)

    result = main.get_agent_response_with_state(_payload())

    assert len(app.states) == 2
    assert result["sql_code"] == "SELECT 42"
    assert result["path_state"]["sql_code"] == "SELECT 42"
    assert result["path_state"]["caller_context"] == "keep"


def test_final_attempt_returns_last_sql_rejected_only_by_intent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _AttemptApp(["failure", "failure_with_fallback"])
    monkeypatch.setattr(main, "_build_state", lambda _payload: _initial_state())
    monkeypatch.setattr(main, "app", app)

    events = list(main.stream_agent_response(_payload()))

    assert len(app.states) == 2
    assert events[-1]["answer"]["sql_code"] == "SELECT fallback"
