# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Question intent classification, rewriting, and evidence extraction."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from auto_ontology.retrieval.text_to_sql.agents import question_intent
from auto_ontology.retrieval.text_to_sql.agents.question_intent import (
    CALCULATION_SUBTYPE_DEFINITIONS,
    CalculationSubtype,
    CalculationOnlyQuestionIntentModel,
    QuestionIntentAgent,
    QuestionIntentModel,
    QuestionType,
    create_question_intent_prompt,
)
from auto_ontology.retrieval.text_to_sql.text_to_sql_graph import (
    _entry_router_fn,
    route_after_candidate_retrieval,
    route_question_type_or_evidence,
)


def _state(**updates: object) -> dict:
    state = {
        "llm": MagicMock(),
        "initial_question": "find revenue in test_dataset dataset",
        "path_state": {},
    }
    state.update(updates)
    return state


@pytest.mark.parametrize("subtype", list(CalculationSubtype))
def test_every_calculation_subtype_is_valid(subtype: CalculationSubtype) -> None:
    model = QuestionIntentModel(
        question_type=QuestionType.CALCULATION,
        calculation_subtype=subtype,
        rewritten_question="question",
        extracted_evidence="",
        target_db=None,
    )

    assert model.calculation_subtype == subtype


def test_json_defines_every_subtype_with_description_and_sql_template() -> None:
    assert set(CALCULATION_SUBTYPE_DEFINITIONS) == {
        subtype.value for subtype in CalculationSubtype
    }
    for definition in CALCULATION_SUBTYPE_DEFINITIONS.values():
        assert definition["description"]
        assert definition["sql_template"].startswith("SELECT")


@pytest.mark.parametrize(
    ("question_type", "subtype"),
    [
        (QuestionType.INFORMATION, CalculationSubtype.MATCH_BASED),
        (QuestionType.PREDICTION, CalculationSubtype.NUMERIC_COMPUTATION),
        (QuestionType.CALCULATION, None),
    ],
)
def test_model_rejects_mismatched_type_and_subtype(
    question_type: QuestionType,
    subtype: CalculationSubtype | None,
) -> None:
    with pytest.raises(ValidationError):
        QuestionIntentModel(
            question_type=question_type,
            calculation_subtype=subtype,
            rewritten_question="question",
            extracted_evidence="",
            target_db=None,
        )


@pytest.mark.parametrize(
    "question_type",
    [QuestionType.INFORMATION, QuestionType.PREDICTION],
)
def test_non_calculation_types_accept_null_subtype(
    question_type: QuestionType,
) -> None:
    model = QuestionIntentModel(
        question_type=question_type,
        calculation_subtype=None,
        rewritten_question="question",
        extracted_evidence="",
        target_db=None,
    )

    assert model.question_type == question_type


def test_agent_rewrites_question_and_merges_extracted_evidence(
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
            target_db="test_datset",
        ),
    )
    state = _state(
        evidence="Revenue means gross_sales minus refunds.",
        connectors=[SimpleNamespace(database_name="test_dataset")],
    )

    result = QuestionIntentAgent().execute(state)

    assert state["initial_question"] == "find revenue in test_dataset dataset"
    assert result["path_state"]["question_type"] == "calculation"
    assert result["path_state"]["calculation_subtype"] == "aggregation"
    assert result["path_state"]["extracted_evidence"] == ""
    assert result["path_state"]["normalized_question"] == "find revenue"
    assert result["path_state"]["target_db"] == "test_dataset"
    assert "AVG(T1.height_cm)" in result["path_state"]["sql_template"]
    assert result["evidence"] == "Revenue means gross_sales minus refunds."


def test_agent_ignores_question_database_when_no_configured_name_is_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        question_intent,
        "invoke_with_structured_output",
        lambda *_args, **_kwargs: QuestionIntentModel(
            question_type=QuestionType.CALCULATION,
            calculation_subtype=CalculationSubtype.MATCH_BASED,
            rewritten_question="find revenue",
            extracted_evidence="",
            target_db="unrelated_database",
        ),
    )
    state = _state(connectors=[SimpleNamespace(database_name="sales")])

    result = QuestionIntentAgent().execute(state)

    assert "target_db" not in result["path_state"]


def test_prompt_lists_configured_databases() -> None:
    prompt = create_question_intent_prompt(
        "Find revenue in finance",
        available_databases=["sales", "finance"],
    )

    assert "## Available databases" in prompt
    assert "- sales" in prompt
    assert "- finance" in prompt
    assert "Set target_db to an exact name from this list" in prompt
    assert "Correct an obvious typo or very close variant" in prompt
    assert "If no listed name is close, return null" in prompt


def test_agent_preserves_existing_target_when_question_does_not_select_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        question_intent,
        "invoke_with_structured_output",
        lambda *_args, **_kwargs: QuestionIntentModel(
            question_type=QuestionType.CALCULATION,
            calculation_subtype=CalculationSubtype.MATCH_BASED,
            rewritten_question="find revenue",
            extracted_evidence="",
            target_db=None,
        ),
    )
    state = _state(
        path_state={"target_db": "sales"},
        connectors=[SimpleNamespace(database_name="sales")],
    )

    result = QuestionIntentAgent().execute(state)

    assert result["path_state"]["target_db"] == "sales"


def test_agent_rejects_question_database_conflicting_with_existing_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        question_intent,
        "invoke_with_structured_output",
        lambda *_args, **_kwargs: QuestionIntentModel(
            question_type=QuestionType.CALCULATION,
            calculation_subtype=CalculationSubtype.MATCH_BASED,
            rewritten_question="find revenue",
            extracted_evidence="",
            target_db="finance",
        ),
    )
    state = _state(
        path_state={"target_db": "sales"},
        connectors=[
            SimpleNamespace(database_name="sales"),
            SimpleNamespace(database_name="finance"),
        ],
    )

    with pytest.raises(ValueError, match="conflicts with"):
        QuestionIntentAgent().execute(state)


def test_agent_does_not_duplicate_existing_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = "Use the test_dataset dataset."
    monkeypatch.setattr(
        question_intent,
        "invoke_with_structured_output",
        lambda *_args, **_kwargs: QuestionIntentModel(
            question_type=QuestionType.CALCULATION,
            calculation_subtype=CalculationSubtype.MATCH_BASED,
            rewritten_question="find revenue",
            extracted_evidence=evidence,
            target_db=None,
        ),
    )

    result = QuestionIntentAgent().execute(_state(evidence=evidence))

    assert result["evidence"] == evidence


def test_failed_classification_defaults_to_calculation_and_original_question(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        question_intent,
        "invoke_with_structured_output",
        lambda *_args, **_kwargs: None,
    )

    result = QuestionIntentAgent().execute(_state())

    assert result["path_state"]["question_type"] == "calculation"
    assert result["path_state"]["calculation_subtype"] == "match_based"
    assert "COUNT(GasStationID)" in result["path_state"]["sql_template"]
    assert result["path_state"]["normalized_question"] == (
        "find revenue in test_dataset dataset"
    )
    assert result["path_state"]["extracted_evidence"] == ""
    assert "evidence" not in result


def test_prompt_defines_taxonomy_fallback_and_evidence_example() -> None:
    prompt = create_question_intent_prompt(
        "find revenue in test_dataset dataset",
        prediction_override=False,
        glossary=[{"name": "ARR", "description": "annual recurring revenue"}],
        existing_evidence="Revenue excludes refunds.",
    )

    assert "If the type is uncertain, choose calculation" in prompt
    assert all(subtype.value in prompt for subtype in CalculationSubtype)
    assert "SELECT Title" in prompt
    assert "ORDER BY ViewCount DESC" in prompt
    assert 'Do not return question_type "prediction"' in prompt
    assert 'rewritten_question: "find revenue"' in prompt
    assert 'target_db: "test_dataset"' in prompt
    assert "belongs only in target_db" in prompt
    assert "ARR: annual recurring revenue" in prompt
    assert "Revenue excludes refunds." in prompt
    assert "do not repeat it in extracted_evidence" in prompt
    assert '"How is revenue calculated/defined?" is information' in prompt
    assert '"Calculate revenue for 2026" is calculation' in prompt


def test_calculation_only_skips_type_selection_but_keeps_subtype_and_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested_models: list[type] = []

    def invoke(_llm: object, messages: list, model: type) -> object:
        requested_models.append(model)
        prompt = str(messages[0].content)
        assert "Perform three tasks" in prompt
        assert "Do not select a top-level question type" in prompt
        assert "## Question types" not in prompt
        return CalculationOnlyQuestionIntentModel(
            calculation_subtype=CalculationSubtype.AGGREGATION,
            rewritten_question="Find revenue by region",
            extracted_evidence="Revenue excludes refunds.",
            target_db="finance",
        )

    monkeypatch.setattr(question_intent, "invoke_with_structured_output", invoke)
    state = _state(
        initial_question="Find revenue by region in the finance database",
        calculation_only=True,
        prediction_override=True,
        evidence="Use booked revenue.",
        connectors=[SimpleNamespace(database_name="finance")],
    )

    result = QuestionIntentAgent().execute(state)

    assert requested_models == [CalculationOnlyQuestionIntentModel]
    assert result["path_state"]["question_type"] == "calculation"
    assert result["path_state"]["calculation_subtype"] == "aggregation"
    assert result["path_state"]["normalized_question"] == "Find revenue by region"
    assert result["path_state"]["target_db"] == "finance"
    assert result["path_state"]["sql_template"]
    assert result["evidence"] == ("Use booked revenue.\nRevenue excludes refunds.")


def test_calculation_only_failure_defaults_to_calculation_subtype(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        question_intent,
        "invoke_with_structured_output",
        lambda *_args, **_kwargs: None,
    )

    result = QuestionIntentAgent().execute(
        _state(calculation_only=True, prediction_override=True)
    )

    assert result["path_state"]["question_type"] == "calculation"
    assert result["path_state"]["calculation_subtype"] == "match_based"
    assert result["path_state"]["sql_template"]


def test_graph_entry_starts_with_intent_but_preserves_resume() -> None:
    assert _entry_router_fn({"path_state": {}}) == "question_intent"
    assert (
        _entry_router_fn({"path_state": {"_resume_from": "reconstruct_sql"}})
        == "reconstruct_sql"
    )


@pytest.mark.parametrize("question_type", ["information", "calculation"])
def test_non_prediction_types_follow_sql_evidence_route(
    question_type: str,
) -> None:
    state = {
        "path_state": {"question_type": question_type},
        "evidence": "Use authoritative formula.",
    }

    assert route_question_type_or_evidence(state) == "refine_evidence"


@pytest.mark.parametrize(
    ("question_type", "route"),
    [
        ("information", "information"),
        ("calculation", "prepare_candidates"),
        ("prediction", "prepare_candidates"),
    ],
)
def test_information_routes_before_candidate_preparation(
    question_type: str, route: str
) -> None:
    state = {"path_state": {"question_type": question_type}}

    assert route_after_candidate_retrieval(state) == route
