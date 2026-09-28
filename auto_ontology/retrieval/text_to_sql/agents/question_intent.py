# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Classify question intent and separate inline evidence before extraction."""

from __future__ import annotations

import json
from difflib import get_close_matches
from enum import StrEnum
from pathlib import Path
from typing import Any, Dict

from langchain_core.messages import SystemMessage
from pydantic import Field, model_validator

from auto_ontology.retrieval.text_to_sql.connector_routing import (
    resolve_target_database_name,
)
from auto_ontology.retrieval.text_to_sql.base import BaseAgent
from auto_ontology.retrieval.text_to_sql.state import (
    AgentState,
    get_standalone_question,
)
from auto_ontology.utils.llm_invoke import (
    StrictLLMOutputModel,
    invoke_with_structured_output,
)


class QuestionType(StrEnum):
    """Top-level route requested by the user."""

    INFORMATION = "information"
    PREDICTION = "prediction"
    CALCULATION = "calculation"


class CalculationSubtype(StrEnum):
    """Calculation taxonomy used for SQL-oriented questions."""

    MATCH_BASED = "match_based"
    RANKING = "ranking"
    COMPARISON = "comparison"
    COUNTING = "counting"
    AGGREGATION = "aggregation"
    DOMAIN_KNOWLEDGE = "domain_knowledge"
    NUMERIC_COMPUTATION = "numeric_computation"
    SYNONYM = "synonym"
    VALUE_ILLUSTRATION = "value_illustration"


def _load_calculation_subtypes() -> dict[str, dict[str, str]]:
    """Load and validate the calculation taxonomy bundled with this package."""
    path = Path(__file__).with_name("calculation_subtypes.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise RuntimeError("calculation_subtypes.json must contain a list")

    definitions: dict[str, dict[str, str]] = {}
    for entry in payload:
        if not isinstance(entry, dict):
            raise RuntimeError("calculation subtype entries must be objects")
        subtype = entry.get("type")
        description = entry.get("description")
        sql_template = entry.get("sql_template")
        if not all(
            isinstance(value, str) and value.strip()
            for value in (subtype, description, sql_template)
        ):
            raise RuntimeError(
                "calculation subtype entries require non-empty type, "
                "description, and sql_template strings"
            )
        if subtype in definitions:
            raise RuntimeError(f"duplicate calculation subtype: {subtype}")
        definitions[subtype] = {
            "description": description.strip(),
            "sql_template": sql_template.strip(),
        }

    expected = {subtype.value for subtype in CalculationSubtype}
    actual = set(definitions)
    if actual != expected:
        raise RuntimeError(
            "calculation subtype JSON does not match CalculationSubtype: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    return definitions


CALCULATION_SUBTYPE_DEFINITIONS = _load_calculation_subtypes()
DATABASE_NAME_SIMILARITY_CUTOFF = 0.8


def _resolve_question_target_database_name(
    requested_database: str,
    available_databases: list[str],
) -> str | None:
    """Resolve an explicit question hint, tolerating a close database-name typo."""
    requested = requested_database.strip()
    if not requested:
        return None

    exact_matches = [
        database
        for database in available_databases
        if database.casefold() == requested.casefold()
    ]
    if len(exact_matches) == 1:
        return exact_matches[0]

    names_by_casefold = {
        database.casefold(): database for database in available_databases
    }
    close_matches = get_close_matches(
        requested.casefold(),
        names_by_casefold,
        n=1,
        cutoff=DATABASE_NAME_SIMILARITY_CUTOFF,
    )
    return names_by_casefold[close_matches[0]] if close_matches else None


class QuestionIntentModel(StrictLLMOutputModel):
    """Structured output of the intent preprocessing call."""

    question_type: QuestionType = Field(
        ...,
        description=(
            "Whether the request is informational, predictive, or a calculation."
        ),
    )
    calculation_subtype: CalculationSubtype | None = Field(
        ...,
        description=(
            "The calculation subtype when question_type is calculation; otherwise null."
        ),
    )
    rewritten_question: str = Field(
        ...,
        description=(
            "The user's executable request with supporting inline instructions "
            "moved to extracted_evidence."
        ),
    )
    extracted_evidence: str = Field(
        ...,
        description=(
            "Supporting instruction or context removed from the question, or an "
            "empty string when none was removed."
        ),
    )
    target_db: str | None = Field(
        ...,
        description=(
            "The exact configured database explicitly requested in the question, "
            "or null when the question does not select one."
        ),
    )

    @model_validator(mode="after")
    def _subtype_matches_type(self) -> QuestionIntentModel:
        if (
            self.question_type == QuestionType.CALCULATION
            and self.calculation_subtype is None
        ):
            raise ValueError("calculation questions require a calculation_subtype")
        if (
            self.question_type != QuestionType.CALCULATION
            and self.calculation_subtype is not None
        ):
            raise ValueError(
                "calculation_subtype must be null unless question_type is calculation"
            )
        return self


class CalculationOnlyQuestionIntentModel(StrictLLMOutputModel):
    """Intent fields needed when the caller guarantees a calculation."""

    calculation_subtype: CalculationSubtype = Field(
        ...,
        description="The calculation subtype for this question.",
    )
    rewritten_question: str = Field(
        ...,
        description=(
            "The user's executable request with supporting inline instructions "
            "moved to extracted_evidence."
        ),
    )
    extracted_evidence: str = Field(
        ...,
        description=(
            "Supporting instruction or context removed from the question, or an "
            "empty string when none was removed."
        ),
    )
    target_db: str | None = Field(
        ...,
        description=(
            "The exact configured database explicitly requested in the question, "
            "or null when the question does not select one."
        ),
    )


def create_question_intent_prompt(
    question: str,
    *,
    prediction_override: bool | None = None,
    calculation_only: bool = False,
    glossary: list[dict[str, str]] | None = None,
    existing_evidence: str = "",
    available_databases: list[str] | None = None,
) -> str:
    """Build the classification and evidence-separation prompt."""
    override_instruction = ""
    if calculation_only:
        override_instruction = (
            "\nThis request is calculation-only. Do not select a top-level question "
            "type. Choose its calculation subtype and still perform rewriting and "
            "evidence extraction.\n"
        )
    elif prediction_override is True:
        override_instruction = (
            "\nThe caller explicitly selected prediction mode. Return question_type "
            '"prediction", but still perform rewriting and evidence extraction.\n'
        )
    elif prediction_override is False:
        override_instruction = (
            "\nThe caller explicitly selected SQL mode. Do not return question_type "
            '"prediction"; classify the request as information or calculation.\n'
        )

    glossary_entries = [
        f"- {name}: {(entry.get('description') or '').strip()}"
        for entry in glossary or []
        if (name := (entry.get("name") or "").strip())
    ]
    glossary_section = ""
    if glossary_entries:
        glossary_section = (
            "\n## Glossary context\n"
            "Use these definitions to recognize internal terms and synonym usage. "
            "Do not copy them into extracted_evidence unless the question itself "
            "contains the instruction.\n" + "\n".join(glossary_entries) + "\n"
        )

    evidence_section = ""
    if existing_evidence.strip():
        evidence_section = (
            "\n## Existing evidence\n"
            "This evidence is already stored separately. Use it for interpretation "
            "but do not repeat it in extracted_evidence.\n"
            f"{existing_evidence.strip()}\n"
        )

    database_names = list(dict.fromkeys(available_databases or []))
    database_section = (
        "\n## Available databases\n"
        + (
            "\n".join(f"- {name}" for name in database_names)
            if database_names
            else "(none)"
        )
        + "\n\n"
        "Set target_db to an exact name from this list only when the question "
        "explicitly asks to use that database. Correct an obvious typo or very close "
        "variant to the exact listed name. If no listed name is close, return null. "
        "A database selection is routing metadata: remove its phrase from "
        "rewritten_question and do not put it in extracted_evidence. Do not infer a "
        "database merely because its name resembles a business entity.\n"
    )

    subtype_section = "\n".join(
        (
            f"- {subtype.value}: {CALCULATION_SUBTYPE_DEFINITIONS[subtype.value]['description']}\n"
            "  SQL shape example:\n"
            + "\n".join(
                f"    {line}"
                for line in CALCULATION_SUBTYPE_DEFINITIONS[subtype.value][
                    "sql_template"
                ].splitlines()
            )
        )
        for subtype in CalculationSubtype
    )

    tasks = (
        """Perform three tasks in one structured response:
1. Choose exactly one calculation subtype.
2. Rewrite the executable question without supporting inline instructions.
3. Return any removed supporting instruction as extracted_evidence."""
        if calculation_only
        else """Perform four tasks in one structured response:
1. Classify the question as information, prediction, or calculation.
2. For calculation only, choose exactly one calculation subtype.
3. Rewrite the executable question without supporting inline instructions.
4. Return any removed supporting instruction as extracted_evidence."""
    )
    question_types = (
        ""
        if calculation_only
        else """
## Question types
- information: asks about catalog or semantic metadata, including what a dataset,
  table, or column means, contains, or connects to, and how an existing measure is
  defined or calculated. "How is revenue calculated/defined?" is information because
  it asks for the stored definition rather than requesting a result.
- prediction: asks to predict, forecast, estimate, project, or assess the likelihood
  of a future or currently unknown outcome. A query about existing historical rows
  is not prediction merely because it contains a date.
- calculation: asks the data system to match/filter, rank, compare, count, aggregate,
  compute, or interpret values. "Calculate revenue for 2026" is calculation because
  it requests a derived value. If the type is uncertain, choose calculation.
"""
    )

    return f"""You prepare a question for a data agent.

{tasks}
{override_instruction}
{question_types}

## Calculation subtypes
Use each SQL example only to understand the subtype's structural pattern. The
identifiers and literal values belong to the example and are not instructions for
the user's question.

{subtype_section}

## Evidence separation
Move text to extracted_evidence only when it instructs the agent how or where to
interpret/calculate the answer, such as a table hint, formula, definition, mapping,
or domain rule. A configured database hint belongs only in target_db, never in
extracted_evidence. Keep actual business filters, dates, entities, requested measures,
grouping, ranking, and output constraints in rewritten_question.

Do not invent evidence. If no evidence is embedded in the question, return an empty
string. The rewritten question must remain understandable on its own.

Examples:
- "find revenue in test_dataset dataset"
  → rewritten_question: "find revenue"
  → extracted_evidence: ""
  → target_db: "test_dataset" (only when it appears in Available databases)
- "What is revenue by region, where revenue means gross_sales minus refunds?"
  → rewritten_question: "What is revenue by region?"
  → extracted_evidence: "Revenue means gross_sales minus refunds."
- "List the top 5 products by revenue in 2025"
  → rewritten_question: unchanged
  → extracted_evidence: ""

{database_section}{glossary_section}{evidence_section}
Question:
{question}"""


def _merge_evidence(existing: str, extracted: str) -> str:
    """Append extracted evidence without replacing or duplicating caller evidence."""
    existing = existing.strip()
    extracted = extracted.strip()
    if not extracted:
        return existing
    if not existing:
        return extracted
    if extracted.casefold() in existing.casefold():
        return existing
    return f"{existing}\n{extracted}"


class QuestionIntentAgent(BaseAgent):
    """Classify intent and prepare question/evidence for downstream extraction."""

    def __init__(self) -> None:
        super().__init__("question_intent")

    def validate_input(self, state: AgentState) -> bool:
        if not get_standalone_question(state):
            self.logger.warning("No question found, skipping question intent")
            return False
        return True

    def execute(self, state: AgentState) -> Dict[str, Any]:
        question = get_standalone_question(state)
        path_state = state.get("path_state", {})
        override = state.get("prediction_override")
        calculation_only = state.get("calculation_only", False)
        existing_evidence = state.get("evidence") or ""
        connectors = list(state.get("connectors") or [])
        available_databases = list(
            dict.fromkeys(
                str(database_name)
                for connector in connectors
                if (database_name := getattr(connector, "database_name", None))
            )
        )

        messages = [
            SystemMessage(
                content=create_question_intent_prompt(
                    question,
                    prediction_override=override,
                    calculation_only=calculation_only,
                    glossary=state.get("glossary"),
                    existing_evidence=existing_evidence,
                    available_databases=available_databases,
                )
            )
        ]
        intent = invoke_with_structured_output(
            state["llm"],
            messages,
            (
                CalculationOnlyQuestionIntentModel
                if calculation_only
                else QuestionIntentModel
            ),
        )

        if intent is None:
            self.logger.warning(
                "Question intent returned None — defaulting to calculation"
            )
            question_type = (
                QuestionType.PREDICTION
                if override is True and not calculation_only
                else QuestionType.CALCULATION
            )
            subtype = (
                None
                if question_type == QuestionType.PREDICTION
                else CalculationSubtype.MATCH_BASED
            )
            rewritten = question
            extracted = ""
            selected_target_db = None
        else:
            question_type = (
                QuestionType.CALCULATION if calculation_only else intent.question_type
            )
            subtype = intent.calculation_subtype
            rewritten = intent.rewritten_question.strip() or question
            extracted = intent.extracted_evidence.strip()
            selected_target_db = intent.target_db

            if calculation_only:
                question_type = QuestionType.CALCULATION
            elif override is True:
                question_type = QuestionType.PREDICTION
                subtype = None
            elif override is False and question_type == QuestionType.PREDICTION:
                question_type = QuestionType.CALCULATION
                subtype = CalculationSubtype.NUMERIC_COMPUTATION

        if selected_target_db:
            resolved_target_db = _resolve_question_target_database_name(
                selected_target_db, available_databases
            )
            if resolved_target_db is not None:
                existing_target_db = path_state.get("target_db")
                if existing_target_db:
                    resolved_existing_target_db = resolve_target_database_name(
                        str(existing_target_db), connectors
                    )
                    if resolved_existing_target_db != resolved_target_db:
                        raise ValueError(
                            "Database selected in the question conflicts with the "
                            "caller-selected target_db "
                            f"{resolved_existing_target_db!r}: "
                            f"{resolved_target_db!r} was requested."
                        )
                path_state["target_db"] = resolved_target_db

        path_state["question_type"] = question_type.value
        path_state["calculation_subtype"] = subtype.value if subtype else None
        path_state["sql_template"] = (
            CALCULATION_SUBTYPE_DEFINITIONS[subtype.value]["sql_template"]
            if subtype
            else None
        )
        path_state["extracted_evidence"] = extracted
        path_state["normalized_question"] = rewritten

        merged_evidence = _merge_evidence(existing_evidence, extracted)
        self.logger.info(
            "Question intent: type=%s subtype=%s target_db=%s rewritten=%r "
            "evidence_extracted=%s",
            question_type.value,
            subtype.value if subtype else None,
            path_state.get("target_db"),
            rewritten,
            bool(extracted),
        )

        result: Dict[str, Any] = {"path_state": path_state}
        if merged_evidence or existing_evidence or extracted:
            result["evidence"] = merged_evidence
        return result


__all__ = [
    "CalculationOnlyQuestionIntentModel",
    "CalculationSubtype",
    "CALCULATION_SUBTYPE_DEFINITIONS",
    "QuestionIntentAgent",
    "QuestionIntentModel",
    "QuestionType",
    "create_question_intent_prompt",
]
