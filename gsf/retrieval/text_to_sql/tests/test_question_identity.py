# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from gsf.retrieval.entity_coverage.agents import question_extraction
from gsf.retrieval.entity_coverage.agents.question_extraction import (
    QuestionExtractionAgent,
)
from gsf.retrieval.text_to_sql.state import (
    get_original_question,
    get_question_for_processing,
)


def _state() -> dict:
    return {
        "initial_question": "What about those in 2025?",
        "path_state": {
            "processing_question": "List active suppliers in 2025",
        },
        "llm": object(),
        "glossary": [],
    }


def test_submitted_and_processing_questions_remain_distinct():
    state = _state()

    assert get_original_question(state) == "What about those in 2025?"
    assert get_question_for_processing(state) == "List active suppliers in 2025"

    state["path_state"]["normalized_question"] = "active suppliers during 2025"
    assert get_original_question(state) == "What about those in 2025?"
    assert get_question_for_processing(state) == "active suppliers during 2025"


def test_extraction_uses_standalone_processing_question(monkeypatch):
    seen: list[str] = []

    def fake_invoke(llm, messages, schema):
        seen.append(messages[0].content)
        return SimpleNamespace(
            sanitized_question="active suppliers during 2025",
            required_entity_name=["active suppliers", "2025"],
            subject="suppliers",
            used_glossary_names=[],
        )

    monkeypatch.setattr(question_extraction, "invoke_with_structured_output", fake_invoke)

    result = QuestionExtractionAgent().execute(_state())

    assert "List active suppliers in 2025" in seen[0]
    assert "What about those in 2025?" not in seen[0]
    assert result["path_state"]["normalized_question"] == "active suppliers during 2025"
