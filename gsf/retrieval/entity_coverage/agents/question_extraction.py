# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Combined question sanitization + entity extraction (single LLM call)."""

from __future__ import annotations

import logging
from typing import Any, Dict

from langchain_core.messages import SystemMessage

from gsf.retrieval.entity_coverage.models import (
    QuestionExtractionLiteModel,
    QuestionExtractionModel,
)
from gsf.retrieval.entity_coverage.prompts import create_question_extraction_prompt
from gsf.retrieval.text_to_sql.base import BaseAgent
from gsf.retrieval.text_to_sql.state import AgentState, get_question_for_processing
from gsf.utils.llm_invoke import invoke_with_structured_output

logger = logging.getLogger(__name__)


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
        # After this node (full mode), ``glossary`` means only the entries used.
        result: Dict[str, Any] = {"path_state": path_state, "glossary": []}

        try:
            messages = [
                SystemMessage(
                    content=create_question_extraction_prompt(
                        original_question,
                        glossary,
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
                path_state["entities"] = [original_question]
                if self._include_subject:
                    path_state["subject"] = original_question
                return result

            sanitized = (extraction.sanitized_question or "").strip()
            if not sanitized:
                self.logger.warning(
                    "Question extraction returned empty sanitize, using original"
                )
                sanitized = original_question

            entities = extraction.required_entity_name or []
            if not entities:
                self.logger.warning(
                    "Question extraction returned empty entities — using question"
                )
                entities = [sanitized]

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
            path_state["entities"] = [original_question]
            if self._include_subject:
                path_state["subject"] = original_question

        return result
