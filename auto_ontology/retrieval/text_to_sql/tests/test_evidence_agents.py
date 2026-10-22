# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from auto_ontology.retrieval.text_to_sql.agents import sql_from_tables
from auto_ontology.retrieval.text_to_sql.agents.sql_from_tables import (
    SQLFromTablesAgent,
)


def _run_sql_from_tables(
    monkeypatch: pytest.MonkeyPatch,
    *,
    question: str,
    evidence: str,
    sql_examples: list[dict] | None = None,
    shorten_answer: bool | None = None,
    relevant_tables: list[dict] | None = None,
) -> list:
    captured: list = []
    monkeypatch.setattr(sql_from_tables, "format_tables_for_prompt", lambda *a, **k: "")
    monkeypatch.setattr(
        sql_from_tables,
        "resolve_connector_from_tables",
        lambda *a, **k: SimpleNamespace(dialect="sqlite"),
    )

    def fake_invoke(_llm: object, messages: list, _model: object) -> None:
        captured.extend(messages)
        return None

    monkeypatch.setattr(sql_from_tables, "invoke_with_structured_output", fake_invoke)
    state = {
        "llm": MagicMock(),
        "initial_question": question,
        "evidence": evidence,
        "messages": [],
        "connectors": [object()],
        "data_retriever": object(),
        "sql_examples": sql_examples or [],
        "path_state": {
            "relevant_tables": relevant_tables if relevant_tables is not None else [{}]
        },
    }
    if shorten_answer is not None:
        state["shorten_answer"] = shorten_answer
    SQLFromTablesAgent().execute(state)
    return captured


def test_tables_agent_uses_state_evidence_as_authoritative_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages = _run_sql_from_tables(
        monkeypatch,
        question="What happened at 0:01:54?",
        evidence="The time refers to events.duration LIKE 'M:SS%'.",
    )

    assert len(messages) == 2
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)
    assert "## Authoritative Evidence" in messages[1].content
    assert "MUST follow every instruction" in messages[1].content
    assert (
        messages[1].content.count("The time refers to events.duration LIKE 'M:SS%'.")
        == 1
    )
    assert "events.duration LIKE '1:54%'" not in messages[1].content


def test_tables_agent_uses_internally_grounded_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list = []
    monkeypatch.setattr(sql_from_tables, "format_tables_for_prompt", lambda *a, **k: "")
    monkeypatch.setattr(
        sql_from_tables,
        "resolve_connector_from_tables",
        lambda *a, **k: SimpleNamespace(dialect="sqlite"),
    )
    monkeypatch.setattr(
        sql_from_tables,
        "invoke_with_structured_output",
        lambda _llm, messages, _model: captured.extend(messages),
    )

    SQLFromTablesAgent().execute(
        {
            "llm": MagicMock(),
            "initial_question": "Show active accounts",
            "evidence": "",
            "messages": [],
            "connectors": [object()],
            "data_retriever": object(),
            "value_anchors": [
                {
                    "phrase": "active",
                    "kind": "value",
                    "tbl": "accounts",
                    "col": "status",
                    "stored_value": "ACTIVE",
                }
            ],
            "path_state": {"relevant_tables": [{}]},
        }
    )

    assert "## Verified Database Values" in captured[-1].content
    assert 'accounts."status"' in captured[-1].content
    assert "'ACTIVE'" in captured[-1].content


def test_tables_agent_does_not_scan_question_for_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages = _run_sql_from_tables(
        monkeypatch,
        question=(
            "What happened at 0:01:54?\n\n"
            "Evidence: The time refers to events.duration LIKE 'M:SS%'."
        ),
        evidence="",
    )

    assert "## Authoritative Evidence" not in messages[1].content
    assert "MUST follow every instruction" not in messages[1].content


def test_tables_agent_enables_reference_query_guidance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages = _run_sql_from_tables(
        monkeypatch,
        question="How many customers placed orders?",
        evidence="Count distinct customers.",
        sql_examples=[{"question": "How many users purchased?", "sql": "SELECT 1"}],
    )

    assert "## How To Use Reference Query Patterns" in messages[0].content


def test_tables_agent_places_calendar_rule_before_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages = _run_sql_from_tables(
        monkeypatch,
        question="How many customers ordered last month?",
        evidence="",
    )

    assert len(messages) == 2
    assert isinstance(messages[-1], HumanMessage)
    assert "Apply only calendar time windows" in messages[-1].content
    assert messages[-1].content.index("Apply only calendar time windows") < (
        messages[-1].content.index("## Final Checklist")
    )


def test_tables_agent_uses_request_projection_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages = _run_sql_from_tables(
        monkeypatch,
        question="Which account has the highest balance?",
        evidence="Use accounts.balance.",
        shorten_answer=True,
    )

    assert "Return exactly the requested output fields and NO others" in (
        messages[-1].content
    )


def test_tables_agent_passes_relevant_tables_to_projection_rules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages = _run_sql_from_tables(
        monkeypatch,
        question="Which people participate in the relationship?",
        evidence="",
        relevant_tables=[
            {
                "name": "relationship",
                "columns": [
                    {"name": "source_id", "semantic_fk": "main.people.id"},
                    {"name": "target_id", "semantic_fk": "people.id"},
                ],
            }
        ],
    )

    assert "multiple foreign-key columns referencing the same" in messages[-1].content
