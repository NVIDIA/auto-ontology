# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
KumoRFM graph-preparation node.

The prediction path is split into two nodes so the (potentially slow) infra
step streams its own progress:

  1. ``prepare_prediction_graph`` (this agent) — load the relevant tables, build
     the KumoRFM ``LocalGraph`` + model, and stash the resulting
     :class:`~gsf.retrieval.kumo.predictor.PredictionContext` in ``path_state``.
  2. ``kumo_predict`` — generate/repair the PQL, predict, and format.

On failure (no relevant tables, KumoRFM unavailable on this platform, etc.) it
writes a graceful ``final_response`` and signals ``predict_failed`` so the graph
routes straight to END without attempting a prediction.
"""

import logging
from typing import Any, Dict

from langchain_core.messages import AIMessage

from gsf.retrieval.kumo import PredictionContext, build_prediction_context
from gsf.retrieval.text_to_sql.base import BaseAgent
from gsf.retrieval.text_to_sql.state import AgentState

logger = logging.getLogger(__name__)


def _error_response(message: str) -> Dict[str, Any]:
    return {
        "response": message,
        "sql_code": "",
        "sql_columns": [],
        "custom_analyses_used": [],
        "sql_response_from_db": None,
    }


class PredictionGraphAgent(BaseAgent):
    """Build the KumoRFM graph/model for the relevant tables (prediction phase 1)."""

    def __init__(self):
        super().__init__("prediction_graph")

    def execute(self, state: AgentState) -> Dict[str, Any]:
        path_state = state.get("path_state", {})
        connectors = state.get("connectors", []) or []
        # Scope the KumoRFM graph to the tables candidate-preparation found relevant,
        # and use the catalog-derived join paths as the graph's table relationships.
        relevant_tables = path_state.get("relevant_tables") or []
        join_paths = path_state.get("attribute_join_paths") or []

        try:
            context = build_prediction_context(
                connectors,
                relevant_tables,
                join_paths=join_paths,
            )
        except Exception as exc:
            self.logger.exception("KumoRFM graph preparation failed")
            context = _error_response(
                "I couldn't prepare a prediction for this question. "
                f"({type(exc).__name__}: {exc})"
            )

        if isinstance(context, PredictionContext):
            return {
                "decision": "predict_ready",
                "path_state": {**path_state, "prediction_context": context},
            }

        # build_prediction_context returned a graceful error response dict —
        # terminate the prediction path with it.
        markdown = context.get("response", "")
        return {
            "decision": "predict_failed",
            "messages": state["messages"] + [AIMessage(content=markdown)],
            "path_state": {
                **path_state,
                "formatted_response": markdown,
                "final_response": context,
            },
        }
