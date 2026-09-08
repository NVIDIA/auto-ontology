# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
KumoRFM prediction agent (terminal node, prediction phase 2).

Runs after ``prepare_prediction_graph`` has built the KumoRFM graph/model into
``path_state["prediction_context"]``. Generates + repairs the PQL, predicts, and
stores the result in ``path_state["final_response"]`` in the same shape the SQL
response uses, so the frontend renders it with no special-casing.
"""

import logging
from typing import Any, Dict

from langchain_core.messages import AIMessage

from gsf.retrieval.kumo import run_prediction
from gsf.retrieval.text_to_sql.base import BaseAgent
from gsf.retrieval.text_to_sql.state import AgentState, get_standalone_question

logger = logging.getLogger(__name__)


def _error_response(message: str) -> Dict[str, Any]:
    return {
        "response": message,
        "sql_code": "",
        "sql_columns": [],
        "custom_analyses_used": [],
        "sql_response_from_db": None,
    }


class KumoPredictionAgent(BaseAgent):
    """Answer a prediction question with KumoRFM (using the prepared graph)."""

    def __init__(self):
        super().__init__("kumo_prediction")

    def execute(self, state: AgentState) -> Dict[str, Any]:
        path_state = state.get("path_state", {})
        # Predict from the unsanitized question (per requirement), resolved against
        # conversation history.
        question = get_standalone_question(state)
        context = path_state.get("prediction_context")

        if context is None:
            # Only reached when prepare_prediction_graph signalled ready, so this
            # is a defensive guard rather than an expected path.
            final_response = _error_response(
                "The prediction graph was not prepared for this question."
            )
        else:
            try:
                final_response = run_prediction(question, state["llm"], context)
            except Exception as exc:
                self.logger.exception("KumoRFM prediction failed")
                final_response = _error_response(
                    "I couldn't produce a prediction for this question. "
                    f"({type(exc).__name__}: {exc})"
                )

        markdown = final_response.get("response", "")
        # Drop the (heavy, non-serializable) prediction context now that we're done.
        new_path_state = {
            **path_state,
            "formatted_response": markdown,
            "final_response": final_response,
        }
        new_path_state.pop("prediction_context", None)
        return {
            "messages": state["messages"] + [AIMessage(content=markdown)],
            "path_state": new_path_state,
        }
