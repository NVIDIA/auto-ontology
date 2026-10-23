# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Combined question sanitization + entity extraction (single LLM call)."""

from __future__ import annotations

import logging
from typing import Any, Dict

from langchain_core.messages import SystemMessage

from auto_ontology.retrieval.entity_coverage.models import (
    ExtractedEntity,
    QuestionExtractionLiteModel,
    QuestionExtractionModel,
)
from auto_ontology.retrieval.entity_coverage.prompts import (
    create_question_extraction_prompt,
)
from auto_ontology.retrieval.text_to_sql.base import BaseAgent
from auto_ontology.retrieval.text_to_sql.state import (
    AgentState,
    get_question_for_processing,
)
from auto_ontology.utils.llm_invoke import invoke_with_structured_output

logger = logging.getLogger(__name__)


def _word_tokens(text: str) -> set[str]:
    """Case-insensitive alphanumeric words without regex-specific semantics."""
    normalized = "".join(
        character.casefold() if character.isalnum() else " " for character in text
    )
    return {token for token in normalized.split() if token}


_ORDERED_COMPARISON_MARKERS = (
    " over ",
    " under ",
    " above ",
    " below ",
    " more than ",
    " less than ",
    " greater than ",
    " fewer than ",
    " at least ",
    " at most ",
    " before ",
    " after ",
    " between ",
    ">",
    "<",
)


def _is_ordered_comparison_value(original_text: str, question: str) -> bool:
    """Whether a proposed exact value is an ordered/range threshold."""
    question_folded = f" {question.casefold()} "
    value_folded = original_text.casefold()
    start = question_folded.find(value_folded)
    if start < 0:
        return False
    preceding = question_folded[max(0, start - 64) : start]
    return any(marker in preceding for marker in _ORDERED_COMPARISON_MARKERS)


def _is_context_only_base_field(
    base: ExtractedEntity,
    qualified: ExtractedEntity,
    question: str,
) -> bool:
    """Whether *base* only qualifies an earlier, more specific requested field."""
    base_meaning = _word_tokens(base.meaning)
    qualified_meaning = _word_tokens(qualified.meaning)
    if not base_meaning or not base_meaning < qualified_meaning:
        return False

    question_folded = question.casefold()
    qualified_text = qualified.original_text.casefold()
    base_text = base.original_text.casefold()
    qualified_start = question_folded.find(qualified_text)
    base_start = question_folded.find(base_text, qualified_start + len(qualified_text))
    if qualified_start < 0 or base_start < 0:
        return False
    between = f" {question_folded[qualified_start + len(qualified_text) : base_start]} "
    return " of " in between or " for " in between


def normalize_extracted_entities(
    entities: list[ExtractedEntity],
    *,
    question: str = "",
) -> list[dict[str, str]]:
    """Deduplicate entities and keep value-span words out of field entities."""
    cleaned: list[ExtractedEntity] = []
    seen: set[tuple[str, str, str]] = set()
    for entity in entities:
        original_text = " ".join(entity.original_text.split()).strip()
        meaning = " ".join(entity.meaning.split()).strip()
        key = (entity.type, original_text.casefold(), meaning.casefold())
        if not original_text or not meaning or key in seen:
            continue
        seen.add(key)
        cleaned.append(
            ExtractedEntity(
                type=entity.type,
                original_text=original_text,
                meaning=meaning,
            )
        )

    cleaned = [
        entity
        for entity in cleaned
        if entity.type != "exact_value"
        or not _is_ordered_comparison_value(entity.original_text, question)
    ]
    value_tokens: set[str] = set()
    for entity in cleaned:
        if entity.type == "exact_value":
            value_tokens.update(_word_tokens(entity.original_text))
    exact_value_meanings = {
        entity.meaning.casefold() for entity in cleaned if entity.type == "exact_value"
    }
    redundant_fields = {
        index
        for index, entity in enumerate(cleaned)
        if entity.type == "field"
        and any(
            other_index != index
            and other.type == "field"
            and _is_context_only_base_field(entity, other, question)
            for other_index, other in enumerate(cleaned)
        )
    }

    return [
        entity.model_dump()
        for index, entity in enumerate(cleaned)
        if index not in redundant_fields
        and (
            entity.type == "exact_value"
            or (
                entity.meaning.casefold() not in exact_value_meanings
                and not (_word_tokens(entity.original_text) & value_tokens)
            )
        )
    ]


def _fallback_entity(text: str) -> dict[str, str]:
    return {
        "type": "field",
        "original_text": text,
        "meaning": text,
    }


def _filter_glossary_by_used_names(
    glossary: list[dict[str, str]],
    llm_names: list[str],
) -> list[dict[str, str]]:
    """Keep configured glossary entries the LLM reported using, without duplicates."""
    by_name = {
        name.casefold(): entry
        for entry in glossary
        if (name := (entry.get("name") or "").strip())
    }
    used: list[dict[str, str]] = []
    seen: set[str] = set()
    for llm_name in llm_names:
        key = (llm_name or "").strip().casefold()
        entry = by_name.get(key)
        if entry is not None and key not in seen:
            used.append(entry)
            seen.add(key)
    return used


class QuestionExtractionAgent(BaseAgent):
    """Sanitize the question and extract entities in one structured LLM call.

    When ``include_subject`` is False (entity-coverage path), the LLM is not asked
    for subject or used_glossary_names; glossary is still injected for sanitization.
    """

    def __init__(self, *, include_subject: bool = True) -> None:
        super().__init__("question_extraction")
        self._include_subject = include_subject

    def validate_input(self, state: AgentState) -> bool:
        question = get_question_for_processing(state)
        if not question:
            self.logger.warning("No question found, skipping question extraction")
            return False
        return True

    def execute(self, state: AgentState) -> Dict[str, Any]:
        llm = state["llm"]
        path_state = state.get("path_state", {})
        original_question = get_question_for_processing(state)
        glossary = state.get("glossary") or []
        evidence = state.get("evidence") or ""
        # After this node (full mode), ``glossary`` means only the entries used.
        result: Dict[str, Any] = {"path_state": path_state, "glossary": []}

        try:
            messages = [
                SystemMessage(
                    content=create_question_extraction_prompt(
                        original_question,
                        glossary,
                        evidence=evidence,
                        include_subject=self._include_subject,
                    )
                )
            ]
            schema = (
                QuestionExtractionModel
                if self._include_subject
                else QuestionExtractionLiteModel
            )
            extraction = invoke_with_structured_output(llm, messages, schema)

            if extraction is None:
                self.logger.warning(
                    "Question extraction returned None, using original question"
                )
                path_state["normalized_question"] = original_question
                path_state["entities"] = [_fallback_entity(original_question)]
                if self._include_subject:
                    path_state["subject"] = original_question
                return result

            sanitized = (extraction.sanitized_question or "").strip()
            if not sanitized:
                self.logger.warning(
                    "Question extraction returned empty sanitize, using original"
                )
                sanitized = original_question

            entities = normalize_extracted_entities(
                extraction.required_entity_name or [],
                question=sanitized,
            )
            if not entities:
                self.logger.warning(
                    "Question extraction returned empty entities — using question"
                )
                entities = [_fallback_entity(sanitized)]

            path_state["normalized_question"] = sanitized
            path_state["entities"] = entities

            if self._include_subject:
                subject = (getattr(extraction, "subject", None) or "").strip()
                if subject:
                    path_state["subject"] = subject
                else:
                    self.logger.warning(
                        "Question extraction returned empty subject — continuing without it"
                    )
                glossary = _filter_glossary_by_used_names(
                    glossary,
                    getattr(extraction, "used_glossary_names", None) or [],
                )
                result["glossary"] = glossary
                self.logger.info(
                    "Extracted question:\n  raw: %s\n  sanitized: %s\n  entities: %s"
                    "\n  subject: %s\n  glossary: %s",
                    original_question,
                    sanitized,
                    entities,
                    subject or None,
                    [e.get("name") for e in glossary],
                )
            else:
                self.logger.info(
                    "Extracted question (lite):\n  raw: %s\n  sanitized: %s"
                    "\n  entities: %s",
                    original_question,
                    sanitized,
                    entities,
                )
        except Exception as exc:
            self.logger.warning(
                "Question extraction failed: %s, using original question",
                exc,
            )
            path_state["normalized_question"] = original_question
            path_state["entities"] = [_fallback_entity(original_question)]
            if self._include_subject:
                path_state["subject"] = original_question

        return result


__all__ = [
    "QuestionExtractionAgent",
    "normalize_extracted_entities",
]
