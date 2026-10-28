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


def _repair_state(candidates: list | None) -> AgentState:
    """A state parked at the repair loop, with *candidates* as the pool."""
    path_state: dict = {
        "error": "no such column: totl",
        "error_analysis_done": True,
        "sql_generation_result": SimpleNamespace(sql_code="SELECT totl", thought=""),
    }
    if candidates is not None:
        path_state["sql_candidates"] = candidates
    return cast(
        AgentState,
        {
            "llm": MagicMock(),
            "initial_question": "How many orders?",
            "evidence": "",
            "messages": [],
            "path_state": path_state,
        },
    )


def _reconstructs_as(
    monkeypatch: pytest.MonkeyPatch, sql: str
) -> SimpleNamespace:
    """Make the repair LLM return *sql*, and hand back what it returned."""
    rebuilt = SimpleNamespace(sql_code=sql, thought="")
    monkeypatch.setattr(
        sql_reconstruction,
        "invoke_with_structured_output",
        lambda _llm, _messages, _model: rebuilt,
    )
    return rebuilt


def test_repaired_sql_joins_the_candidate_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = [
        SimpleNamespace(sql_code="SELECT totl", thought=""),
        SimpleNamespace(sql_code="SELECT other", thought=""),
    ]
    rebuilt = _reconstructs_as(monkeypatch, "SELECT total")

    result = SQLReconstructionAgent().execute(_repair_state(pool))

    candidates = result["path_state"]["sql_candidates"]
    assert [c.sql_code for c in candidates] == [
        "SELECT totl",
        "SELECT other",
        "SELECT total",
    ]
    assert candidates[-1] is rebuilt
    assert result["path_state"]["sql_generation_result"] is rebuilt


def test_appending_leaves_the_generated_slots_where_they_were(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = [SimpleNamespace(sql_code=f"SELECT {i}", thought="") for i in range(6)]
    _reconstructs_as(monkeypatch, "SELECT repaired")

    result = SQLReconstructionAgent().execute(_repair_state(pool))

    candidates = result["path_state"]["sql_candidates"]
    assert [c.sql_code for c in candidates[:6]] == [f"SELECT {i}" for i in range(6)]
    assert candidates[6].sql_code == "SELECT repaired"


def test_a_repair_that_converges_adds_no_duplicate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = [
        SimpleNamespace(sql_code="SELECT totl", thought=""),
        SimpleNamespace(sql_code="  SELECT total  ", thought=""),
    ]
    _reconstructs_as(monkeypatch, "SELECT total")

    result = SQLReconstructionAgent().execute(_repair_state(pool))

    assert len(result["path_state"]["sql_candidates"]) == 2


def test_the_pool_does_not_mutate_in_place(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = [SimpleNamespace(sql_code="SELECT totl", thought="")]
    _reconstructs_as(monkeypatch, "SELECT total")

    SQLReconstructionAgent().execute(_repair_state(pool))

    assert [c.sql_code for c in pool] == ["SELECT totl"]


def test_repair_starts_a_pool_when_there_was_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reconstructs_as(monkeypatch, "SELECT total")

    result = SQLReconstructionAgent().execute(_repair_state(None))

    candidates = result["path_state"]["sql_candidates"]
    assert [c.sql_code for c in candidates] == ["SELECT total"]


def test_an_unconstructable_repair_leaves_the_pool_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = [SimpleNamespace(sql_code="SELECT totl", thought="")]
    monkeypatch.setattr(
        sql_reconstruction,
        "invoke_with_structured_output",
        lambda _llm, _messages, _model: None,
    )

    result = SQLReconstructionAgent().execute(_repair_state(pool))

    assert result["decision"] == "unconstructable"
    assert result["path_state"]["sql_candidates"] is pool
