# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest

from auto_ontology.retrieval.entity_coverage.agents import question_extraction
from auto_ontology.retrieval.entity_coverage.agents.question_extraction import (
    QuestionExtractionAgent,
)
from auto_ontology.retrieval.entity_coverage.models import QuestionExtractionLiteModel
from auto_ontology.retrieval.entity_coverage.prompts import (
    create_question_extraction_prompt,
)


def test_prompt_limits_evidence_to_sanitized_question_disambiguation() -> None:
    prompt = create_question_extraction_prompt(
        "Show points earned by customers",
        evidence="Points earned refers to loyalty_points, not reward_balance.",
        include_subject=False,
    )

    assert "## Evidence for sanitization" in prompt
    assert "at most one concise disambiguating qualifier" in prompt
    assert "Do not append or summarize the evidence" in prompt
    assert "derive entities only from the completed sanitized question" in prompt
    assert "Points earned refers to loyalty_points" in prompt
    assert "AVAILABLE TABLES" not in prompt


@pytest.mark.parametrize("evidence", [None, ""])
def test_prompt_omits_evidence_section_when_not_provided(
    evidence: str | None,
) -> None:
    prompt = create_question_extraction_prompt(
        "Show points earned by customers",
        evidence=evidence,
        include_subject=False,
    )

    assert "## Evidence for sanitization" not in prompt


def test_agent_forwards_state_evidence_without_changing_original_question(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_messages: list = []

    def fake_invoke(
        _llm: object,
        messages: list,
        _schema: object,
    ) -> QuestionExtractionLiteModel:
        captured_messages.extend(messages)
        return QuestionExtractionLiteModel(
            sanitized_question="Show loyalty points earned by customers",
            required_entity_name=["loyalty points", "customer"],
        )

    monkeypatch.setattr(
        question_extraction,
        "invoke_with_structured_output",
        fake_invoke,
    )
    state = {
        "llm": MagicMock(),
        "initial_question": "Show points earned by customers",
        "evidence": "Points earned refers to loyalty_points.",
        "path_state": {},
        "glossary": [],
    }

    result = QuestionExtractionAgent(include_subject=False).execute(state)

    assert "Points earned refers to loyalty_points." in captured_messages[0].content
    assert state["initial_question"] == "Show points earned by customers"
    assert result["path_state"]["normalized_question"] == (
        "Show loyalty points earned by customers"
    )
    assert result["path_state"]["entities"] == ["loyalty points", "customer"]
