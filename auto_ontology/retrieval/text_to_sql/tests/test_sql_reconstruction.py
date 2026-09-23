# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import cast
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from auto_ontology.retrieval.text_to_sql.agents import sql_reconstruction
from auto_ontology.retrieval.text_to_sql.agents.sql_reconstruction import (
    SQLReconstructionAgent,
)
from auto_ontology.retrieval.text_to_sql.state import AgentState


@pytest.mark.parametrize(
    ("path_state", "expected_database"),
    [
        ({"retrieval_database": "retrieved-db"}, "retrieved-db"),
        (
            {
                "target_db": "target-db",
                "retrieval_database": "retrieved-db",
            },
            "target-db",
        ),
    ],
)
def test_table_discovery_uses_retrieval_database(
    monkeypatch: pytest.MonkeyPatch,
    path_state: dict,
    expected_database: str,
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
            "path_state": path_state,
        },
    )

    result = SQLReconstructionAgent()._discover_tables(
        state,
        ["orders"],
        [],
    )

    assert result == []
    assert database_names == [expected_database]


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
