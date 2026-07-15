# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Question understanding for text-to-SQL retrieval.

A single LLM call that both sanitizes the user's request and extracts structured
entities, storing in path_state:
- normalized_question: concise, SQL-ready rewrite (all factual constraints kept)
- metadata_entities:   schema-level concepts (list[str])
- value_entities:      concrete filter targets, concise form (list[str])
- entities:            united list[str] (metadata + values) for retrieval
"""

import logging
from typing import Any, Dict

from langchain_core.messages import SystemMessage
from pydantic import BaseModel, ConfigDict, Field

from gsf.retrieval.text_to_sql.base import BaseAgent
from gsf.retrieval.text_to_sql.prompts import create_question_understanding_prompt
from gsf.retrieval.text_to_sql.state import AgentState, get_original_question
from gsf.utils.llm_invoke import invoke_with_structured_output

logger = logging.getLogger(__name__)


class QuestionUnderstandingModel(BaseModel):
    """Sanitized question plus metadata/value entity phrases."""

    model_config = ConfigDict(extra="forbid")

    normalized_question: str = Field(
        ...,
        description=(
            "Concise, SQL-ready rewrite of the user's request. Preserve every "
            "factual constraint (numbers, thresholds, names); remove narrative fluff."
        ),
    )
    metadata: list[str] = Field(
        default_factory=list,
        description=(
            "Schema-level concept phrases (columns/tables/relationships), each a "
            "single string."
        ),
    )
    values: list[str] = Field(
        default_factory=list,
        description=(
            "Concrete items/values the user searches or filters for, each a single "
            "concise phrase (short, product-style; not the verbose form)."
        ),
    )


class QuestionUnderstandingAgent(BaseAgent):
    """Sanitize the question and extract metadata/value entities in one LLM call."""

    def __init__(self):
        super().__init__("question_understanding")

    def validate_input(self, state: AgentState) -> bool:
        """Validate that a question is available."""
        question = get_original_question(state)
        if not question:
            self.logger.warning("No question found, skipping question understanding")
            return False
        return True

    def execute(self, state: AgentState) -> Dict[str, Any]:
        """Sanitize the question and extract structured entities."""
        llm = state.get("entity_llm") or state["llm"]
        path_state = state.get("path_state", {})
        original_question = get_original_question(state)

        result: Dict[str, Any] = {"path_state": path_state}

        try:
            messages = [
                SystemMessage(
                    content=create_question_understanding_prompt(original_question)
                )
            ]
            understanding = invoke_with_structured_output(
                llm,
                messages,
                QuestionUnderstandingModel,
            )

            if understanding is None:
                self.logger.warning(
                    "Question understanding returned None — using fallbacks"
                )
                path_state["normalized_question"] = original_question
                path_state["metadata_entities"] = []
                path_state["value_entities"] = []
                path_state["entities"] = [original_question]
                return result

            sanitized = (understanding.normalized_question or "").strip()
            if not sanitized:
                self.logger.warning(
                    "Empty normalized_question — using original question"
                )
                sanitized = original_question

            metadata_entities = [m.strip() for m in understanding.metadata if m.strip()]
            value_entities = [v.strip() for v in understanding.values if v.strip()]

            entities = metadata_entities + value_entities
            if not entities:
                self.logger.warning(
                    "No entities extracted — using question as fallback"
                )
                entities = [sanitized]

            path_state["normalized_question"] = sanitized
            path_state["metadata_entities"] = metadata_entities
            path_state["value_entities"] = value_entities
            path_state["entities"] = entities

            self.logger.info(
                "Question understanding:\n  raw: %s\n  sanitized: %s\n"
                "  metadata: %s\n  values: %s",
                original_question,
                sanitized,
                metadata_entities,
                value_entities,
            )
        except Exception as e:
            self.logger.warning(
                "Question understanding failed: %s — using fallbacks", e
            )
            path_state["normalized_question"] = original_question
            path_state["metadata_entities"] = []
            path_state["value_entities"] = []
            path_state["entities"] = [original_question]

        return result
