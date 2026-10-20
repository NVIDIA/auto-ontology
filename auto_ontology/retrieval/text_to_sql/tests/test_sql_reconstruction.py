# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import cast
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from auto_ontology.retrieval.text_to_sql.agents import sql_reconstruction
from auto_ontology.retrieval.text_to_sql.agents.sql_reconstruction import (
    SQLReconstructionAgent,
)
from auto_ontology.retrieval.text_to_sql.state import AgentState


def test_table_discovery_uses_target_database(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_names: list[str | None] = []

    def fake_get_relevant_tables(
        _retriever: object,
        _query_text: str,
        *,
        k: int,
        database_name: str | None,
    ) -> list[dict]:
        assert k == 3
        database_names.append(database_name)
        return []

    monkeypatch.setattr(
        sql_reconstruction,
        "get_relevant_tables",
        fake_get_relevant_tables,
    )
    state = cast(
        AgentState,
        {
            "data_retriever": object(),
            "path_state": {"target_db": "target-db"},
        },
    )

    result = SQLReconstructionAgent()._discover_tables(
        state,
        ["orders"],
        [],
    )

    assert result == []
    assert database_names == ["target-db"]


def test_reconstruction_uses_evidence_from_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_messages: list = []

    def fake_invoke(_llm: object, messages: list, _model: object) -> None:
        captured_messages.extend(messages)
        return None

    monkeypatch.setattr(
        sql_reconstruction,
        "invoke_with_structured_output",
        fake_invoke,
    )
    state = cast(
        AgentState,
        {
            "llm": MagicMock(),
            "initial_question": "What happened at 0:01:54?",
            "evidence": "The time refers to events.duration LIKE 'M:SS%'.",
            "messages": [],
            "path_state": {
                "error": "invalid time filter",
                "error_analysis_done": True,
                "sql_generation_result": SimpleNamespace(
                    sql_code="SELECT 1",
                    thought="",
                ),
            },
        },
    )

    SQLReconstructionAgent().execute(state)

    prompt = captured_messages[-1].content
    evidence_message = captured_messages[-2].content
    assert "## Authoritative Evidence" in evidence_message
    assert "MUST follow every instruction" in evidence_message
    assert evidence_message.endswith("The time refers to events.duration LIKE 'M:SS%'.")
    assert "The time refers to" not in prompt
    assert "events.duration LIKE '1:54%'" not in prompt


def test_reconstruction_drops_prior_repair_turns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_messages: list = []

    def fake_invoke(_llm: object, messages: list, _model: object) -> SimpleNamespace:
        captured_messages.extend(messages)
        return SimpleNamespace(sql_code="SELECT 2", thought="fixed", response="ok")

    monkeypatch.setattr(
        sql_reconstruction,
        "invoke_with_structured_output",
        fake_invoke,
    )
    constructor_system = SystemMessage(content="SQL construction rules")
    constructor_human = HumanMessage(content="Available tables: accounts")
    prior_repair = HumanMessage(
        content=(
            "The following SQL contains an ERROR:\n\n"
            "```sql\nSELECT bad\n```\n\n"
            "Validation failed with the following message:\ncolumn missing\n\n"
        )
    )
    state = cast(
        AgentState,
        {
            "llm": MagicMock(),
            "initial_question": "How many accounts?",
            "evidence": "",
            "messages": [constructor_system, constructor_human, prior_repair],
            "path_state": {
                "error": "still wrong",
                "error_analysis_done": True,
                "failed_attempts": [{"sql": "SELECT bad", "error": "column missing"}],
                "sql_generation_result": SimpleNamespace(
                    sql_code="SELECT bad2",
                    thought="",
                ),
            },
        },
    )

    result = SQLReconstructionAgent().execute(state)

    assert "messages" not in result
    assert captured_messages[:2] == [constructor_system, constructor_human]
    repair_turns = [
        message
        for message in captured_messages
        if isinstance(message, HumanMessage)
        and message.content.startswith("The following SQL contains an ERROR:")
    ]
    assert repair_turns == [captured_messages[-1]]
    assert prior_repair not in captured_messages
    assert "SELECT bad" in captured_messages[-1].content
    assert "column missing" in captured_messages[-1].content
