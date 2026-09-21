# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gsf.retrieval.text_to_sql.agents import sql_from_tables
from gsf.retrieval.text_to_sql.agents.sql_from_tables import SQLFromTablesAgent


def _run_sql_from_tables(
    monkeypatch: pytest.MonkeyPatch,
    *,
    question: str,
    evidence: str,
    sql_examples: list[dict] | None = None,
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
    SQLFromTablesAgent().execute(
        {
            "llm": MagicMock(),
            "initial_question": question,
            "evidence": evidence,
            "messages": [],
            "connectors": [object()],
            "data_retriever": object(),
            "sql_examples": sql_examples or [],
            "path_state": {"relevant_tables": [{}]},
        }
    )
    return captured


def test_tables_agent_uses_state_evidence_as_authoritative_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages = _run_sql_from_tables(
        monkeypatch,
        question="What happened at 0:01:54?",
        evidence="The time refers to events.duration LIKE 'M:SS%'.",
    )

    assert "## Authoritative Evidence" in messages[1].content
    assert "MUST follow every instruction" in messages[1].content
    assert messages[1].content.endswith(
        "The time refers to events.duration LIKE 'M:SS%'."
    )
    assert "events.duration LIKE '1:54%'" not in messages[2].content
    assert "The time refers to" not in messages[2].content


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
