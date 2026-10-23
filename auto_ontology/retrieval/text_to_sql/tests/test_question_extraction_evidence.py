# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from auto_ontology.retrieval.entity_coverage.agents import question_extraction
from auto_ontology.retrieval.entity_coverage.agents.question_extraction import (
    QuestionExtractionAgent,
)
from auto_ontology.retrieval.entity_coverage.models import QuestionExtractionLiteModel
from auto_ontology.retrieval.entity_coverage.models import ExtractedEntity
from auto_ontology.retrieval.entity_coverage.prompts import (
    create_question_extraction_prompt,
)
from auto_ontology.retrieval.text_to_sql.agents import question_intent
from auto_ontology.retrieval.text_to_sql.agents.question_intent import (
    CalculationSubtype,
    QuestionIntentAgent,
    QuestionIntentModel,
    QuestionType,
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
    assert "derive entity spans only from the completed sanitized question" in prompt
    assert "use that hint in the matching entity's `meaning`" in prompt
    assert "must never create an entity" in prompt
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


def test_typed_entity_prompt_preserves_existing_field_rules() -> None:
    prompt = create_question_extraction_prompt(
        "Find a waterproof hiking tent for family camping.",
        include_subject=False,
    )

    assert "Aggregation-qualified metric rule" in prompt
    assert "Filter-item rule" in prompt
    assert "For interrogative words" in prompt
    assert "Date-field rule" in prompt
    assert "Glossary rule" in prompt
    assert "Do not emit bare action verbs" in prompt
    assert "extension of the field" in prompt
    assert "not a replacement for them" in prompt
    assert "`exact_value`" in prompt
    assert "The Lion King" not in prompt
    assert "Angel of Mercy" not in prompt
    assert "total enrollment of over 500" not in prompt


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
            required_entity_name=[
                ExtractedEntity(
                    type="field",
                    original_text="loyalty points",
                    meaning="loyalty points",
                ),
                ExtractedEntity(
                    type="field",
                    original_text="customers",
                    meaning="customer",
                ),
            ],
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
    assert result["path_state"]["entities"] == [
        {
            "type": "field",
            "original_text": "loyalty points",
            "meaning": "loyalty points",
        },
        {
            "type": "field",
            "original_text": "customers",
            "meaning": "customer",
        },
    ]


def test_intent_rewrite_and_evidence_feed_question_extraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        question_intent,
        "invoke_with_structured_output",
        lambda *_args, **_kwargs: QuestionIntentModel(
            question_type=QuestionType.CALCULATION,
            calculation_subtype=CalculationSubtype.AGGREGATION,
            rewritten_question="find revenue",
            extracted_evidence="",
            target_db="test_dataset",
        ),
    )
    state = {
        "llm": MagicMock(),
        "initial_question": "find revenue in test_dataset dataset",
        "evidence": "Revenue means gross_sales minus refunds.",
        "path_state": {},
        "glossary": [],
        "connectors": [SimpleNamespace(database_name="test_dataset")],
    }
    state.update(QuestionIntentAgent().execute(state))
    captured_messages: list = []

    def fake_extract(
        _llm: object,
        messages: list,
        _schema: object,
    ) -> QuestionExtractionLiteModel:
        captured_messages.extend(messages)
        return QuestionExtractionLiteModel(
            sanitized_question="find revenue",
            required_entity_name=[
                ExtractedEntity(
                    type="field",
                    original_text="revenue",
                    meaning="revenue",
                )
            ],
        )

    monkeypatch.setattr(
        question_extraction,
        "invoke_with_structured_output",
        fake_extract,
    )

    result = QuestionExtractionAgent(include_subject=False).execute(state)

    prompt = captured_messages[0].content
    assert "## Input\n\nfind revenue" in prompt
    assert "Revenue means gross_sales minus refunds." in prompt
    assert "Use the test_dataset dataset." not in prompt
    assert state["path_state"]["target_db"] == "test_dataset"
    assert state["initial_question"] == "find revenue in test_dataset dataset"
    assert result["path_state"]["normalized_question"] == "find revenue"


def test_value_words_are_not_reused_by_field_entities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        question_extraction,
        "invoke_with_structured_output",
        lambda *_args, **_kwargs: QuestionExtractionLiteModel(
            sanitized_question="Did class 2 see The Lion King?",
            required_entity_name=[
                ExtractedEntity(
                    type="exact_value",
                    original_text="class 2",
                    meaning="class",
                ),
                ExtractedEntity(
                    type="field",
                    original_text="class",
                    meaning="class",
                ),
                ExtractedEntity(
                    type="exact_value",
                    original_text="The Lion King",
                    meaning="movie",
                ),
            ],
        ),
    )
    state = {
        "llm": MagicMock(),
        "initial_question": "Did class 2 see The Lion King?",
        "path_state": {},
        "glossary": [],
    }

    result = QuestionExtractionAgent(include_subject=False).execute(state)

    assert result["path_state"]["entities"] == [
        {"type": "exact_value", "original_text": "class 2", "meaning": "class"},
        {
            "type": "exact_value",
            "original_text": "The Lion King",
            "meaning": "movie",
        },
    ]


def test_ordered_comparison_threshold_is_not_an_exact_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        question_extraction,
        "invoke_with_structured_output",
        lambda *_args, **_kwargs: QuestionExtractionLiteModel(
            sanitized_question=(
                "List the codes of schools with a total enrollment of over 500."
            ),
            required_entity_name=[
                ExtractedEntity(
                    type="field",
                    original_text="schools",
                    meaning="school",
                ),
                ExtractedEntity(
                    type="field",
                    original_text="codes",
                    meaning="school code",
                ),
                ExtractedEntity(
                    type="field",
                    original_text="total enrollment",
                    meaning="total enrollment",
                ),
                ExtractedEntity(
                    type="exact_value",
                    original_text="500",
                    meaning="total enrollment",
                ),
            ],
        ),
    )
    state = {
        "llm": MagicMock(),
        "initial_question": (
            "Please list the codes of the schools with a total enrollment of over 500."
        ),
        "path_state": {},
        "glossary": [],
    }

    result = QuestionExtractionAgent(include_subject=False).execute(state)

    assert result["path_state"]["entities"] == [
        {"type": "field", "original_text": "codes", "meaning": "school code"},
        {
            "type": "field",
            "original_text": "total enrollment",
            "meaning": "total enrollment",
        },
    ]


def test_exact_value_replaces_duplicate_field_meaning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        question_extraction,
        "invoke_with_structured_output",
        lambda *_args, **_kwargs: QuestionExtractionLiteModel(
            sanitized_question="Find the school named Lincoln High.",
            required_entity_name=[
                ExtractedEntity(
                    type="field",
                    original_text="school",
                    meaning="school name",
                ),
                ExtractedEntity(
                    type="exact_value",
                    original_text="Lincoln High",
                    meaning="school name",
                ),
            ],
        ),
    )
    state = {
        "llm": MagicMock(),
        "initial_question": "Find the school named Lincoln High.",
        "path_state": {},
        "glossary": [],
    }

    result = QuestionExtractionAgent(include_subject=False).execute(state)

    assert result["path_state"]["entities"] == [
        {
            "type": "exact_value",
            "original_text": "Lincoln High",
            "meaning": "school name",
        }
    ]
