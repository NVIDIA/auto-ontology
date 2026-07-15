# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Prediction decision tree.

Runs right after question sanitization. Classifies whether the user's question
asks for a *prediction* (a future/unknown value to forecast) versus a normal
question answerable by querying existing data. Sets ``state["decision"]`` to
``"prediction"`` or ``"sql"`` so the graph can branch to the KumoRFM tool.
"""

import logging
from typing import Any, Dict

from langchain_core.messages import SystemMessage
from pydantic import BaseModel, ConfigDict, Field

from gsf.retrieval.text_to_sql.base import BaseAgent
from gsf.retrieval.text_to_sql.prompts import create_prediction_classification_prompt
from gsf.retrieval.text_to_sql.state import AgentState, get_original_question
from gsf.utils.llm_invoke import invoke_with_structured_output

logger = logging.getLogger(__name__)


class PredictionDecisionModel(BaseModel):
    """Whether the question is a prediction/forecasting request."""

    model_config = ConfigDict(extra="forbid")

    is_prediction: bool = Field(
        ...,
        description=(
            "True ONLY if the question asks to predict, forecast, estimate, or "
            "project a future or currently-unknown value (e.g. 'how many orders "
            "will customer X place next month', 'predict churn', 'expected "
            "revenue next quarter'). False for questions answerable from existing "
            "data (counts, lookups, aggregations of what already happened)."
        ),
    )


class PredictionClassificationAgent(BaseAgent):
    """Route prediction questions to KumoRFM and the rest to text-to-SQL."""

    def __init__(self):
        super().__init__("prediction_classification")

    def execute(self, state: AgentState) -> Dict[str, Any]:
        # Classify on the raw question (per requirement), not the sanitized one.
        question = get_original_question(state)
        if not question:
            return {"decision": "sql"}

        messages = [
            SystemMessage(content=create_prediction_classification_prompt(question))
        ]
        result = invoke_with_structured_output(
            state["llm"], messages, PredictionDecisionModel
        )
        if result is None:
            self.logger.warning("Prediction classification failed — defaulting to SQL")
            return {"decision": "sql"}

        decision = "prediction" if result.is_prediction else "sql"
        self.logger.info("Prediction classification → %s", decision)
        return {"decision": decision}
