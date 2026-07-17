# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Question sanitization for text-to-SQL retrieval.

Rewrites conversational user requests into concise, SQL-ready questions and stores
the result in path_state["normalized_question"] for downstream retrieval steps.
"""

import logging
from typing import Any, Dict

from langchain_core.messages import SystemMessage
from pydantic import BaseModel, ConfigDict, Field

from gsf.retrieval.text_to_sql.base import BaseAgent
from gsf.retrieval.text_to_sql.prompts import create_question_sanitization_prompt
from gsf.retrieval.text_to_sql.state import AgentState, get_original_question
from gsf.utils.llm_invoke import invoke_with_structured_output

logger = logging.getLogger(__name__)


class QuestionSanitizationModel(BaseModel):
    """Sanitized question for retrieval and SQL intent."""

    model_config = ConfigDict(extra="forbid")

    sanitized_question: str = Field(
        ...,
        description=(
            "Concise, SQL-ready rewrite of the user's request. "
            "Preserve factual constraints and remove narrative fluff."
        ),
    )


class QuestionSanitizationAgent(BaseAgent):
    """Rewrite conversational questions into concise SQL/search intent."""

    def __init__(self):
        super().__init__("question_sanitization")

    def validate_input(self, state: AgentState) -> bool:
        """Validate that a question is available."""
        question = get_original_question(state)
        if not question:
            self.logger.warning("No question found, skipping question sanitization")
            return False
        return True

    def execute(self, state: AgentState) -> Dict[str, Any]:
        """Sanitize the user's question for retrieval-oriented downstream steps."""
        llm = state["llm"]
        path_state = state.get("path_state", {})
        original_question = get_original_question(state)

        result: Dict[str, Any] = {"path_state": path_state}

        try:
            messages = [
                SystemMessage(
                    content=create_question_sanitization_prompt(original_question)
                )
            ]
            sanitization_result = invoke_with_structured_output(
                llm,
                messages,
                QuestionSanitizationModel,
            )

            if sanitization_result is None:
                self.logger.warning(
                    "Question sanitization returned None, using original question"
                )
                path_state["normalized_question"] = original_question
                return result

            sanitized = (sanitization_result.sanitized_question or "").strip()
            if not sanitized:
                self.logger.warning(
                    "Question sanitization returned empty text, using original question"
                )
                sanitized = original_question

            path_state["normalized_question"] = sanitized
            self.logger.info(
                "Sanitized question:\n  raw: %s\n  sanitized: %s",
                original_question,
                sanitized,
            )
        except Exception as e:
            self.logger.warning(
                "Question sanitization failed: %s, using original question", e
            )
            path_state["normalized_question"] = original_question

        return result
