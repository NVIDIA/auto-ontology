# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Visualization Agent

After ``format_and_respond``, optionally attach ResultChart specs on
``final_response["charts"]`` (client renders them as a separate message, like
illumex). Soft-fails on any error so the prose answer still ships.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from gsf.retrieval.text_to_sql.base import BaseAgent
from gsf.retrieval.text_to_sql.state import AgentState, get_original_question
from gsf.retrieval.text_to_sql.visualization import analyze_and_visualize

logger = logging.getLogger(__name__)


class VisualizationAgent(BaseAgent):
    """Attach chart specs to ``final_response`` when visualization is enabled."""

    def __init__(self) -> None:
        super().__init__("visualization")

    def validate_input(self, state: AgentState) -> bool:
        path_state = state.get("path_state", {})
        if not path_state.get("final_response"):
            self.logger.warning("No final_response — skipping visualization")
            return False
        return True

    def execute(self, state: AgentState) -> Dict[str, Any]:
        path_state = state.get("path_state", {})
        final_response = path_state.get("final_response")
        if not isinstance(final_response, dict):
            return {}

        if path_state.get("visualization_enabled", True) is False:
            self.logger.info("Visualization disabled by user setting — skip")
            return {}

        sql_response = path_state.get("sql_response_from_db")
        if not sql_response:
            self.logger.info("No SQL result rows — skip visualization")
            return {}

        llm_response = path_state.get("sql_generation_result")
        sql_code = (
            path_state.get("sql_code") or getattr(llm_response, "sql_code", "") or ""
        )
        question = get_original_question(state)
        llm = state.get("non_reasoning_llm") or state.get("llm")
        if llm is None:
            self.logger.warning("No LLM available for visualization — skip")
            return {}

        specs = analyze_and_visualize(
            llm=llm,
            question=question,
            sql=sql_code,
            sql_response_from_db=sql_response,
        )
        if not specs:
            return {}

        # Keep prose in ``response``; charts are a separate payload so the UI can
        # render illumex-style Message 1 (text+SQL) / Message 2 (charts).
        updated_response = {
            **final_response,
            "charts": specs,
        }

        self.logger.info("Attached %d chart spec(s) to final_response", len(specs))
        return {
            "path_state": {
                **path_state,
                "final_response": updated_response,
                "charts": specs,
            },
        }
