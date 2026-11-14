# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The pipeline step that answers a question with one ReAct episode.

It hands a query to ``format_and_respond`` in the shape that node reads, ends
the run itself with a KumoRFM prediction's response, or hands off to
``unconstructable_sql_response`` when the episode ends without an answer.
"""

from __future__ import annotations

from typing import Any, Dict

from langchain_core.messages import AIMessage

from auto_ontology.retrieval.text_to_sql.base import BaseAgent, record_thought
from auto_ontology.retrieval.text_to_sql.models import SQLGenerationModel
from auto_ontology.retrieval.text_to_sql.react.context import ToolContext
from auto_ontology.retrieval.text_to_sql.react.loop import run_episode
from auto_ontology.retrieval.text_to_sql.react.mode import ToolName, episode_mode
from auto_ontology.retrieval.text_to_sql.react.run_sql import connector_for
from auto_ontology.retrieval.text_to_sql.state import (
    AgentState,
    get_question_for_processing,
    get_standalone_question,
)

STEP_NAME = "react_sql_agent"

_UNANSWERED = (
    "I couldn't write a query that answers this question from the data I could "
    "find. Try rephrasing it, or name the table or measure you mean."
)

_NO_ASSUMPTIONS = "No assumptions were needed beyond what the question states."


def _tool_thought(call: dict[str, Any]) -> str:
    args = call.get("args") or {}
    name = call.get("name")
    if name == ToolName.RETRIEVE:
        return f"Searching for: {args.get('query', '')}"
    if name == ToolName.RUN_SQL and str(args.get("final")).lower() != "true":
        return "Running a query to check the data."
    if name == ToolName.PREDICT:
        return "Predicting with KumoRFM."
    if name == ToolName.GREP:
        return f"Searching a saved result for: {args.get('pattern', '')}"
    return "Submitting the final query."


class ReactSQLAgent(BaseAgent):
    """Answer the question by letting the model search, query and submit."""

    def __init__(self) -> None:
        super().__init__("react_sql_agent", failure_decision="unconstructable")

    def validate_input(self, state: AgentState) -> bool:
        return bool(state.get("initial_question")) and bool(state.get("connectors"))

    def execute(self, state: AgentState) -> Dict[str, Any]:
        path_state = dict(state.get("path_state") or {})
        ctx = ToolContext(
            retriever=state.get("semantic_retriever") or state.get("data_retriever"),
            data_retriever=state.get("data_retriever"),
            connectors=list(state.get("connectors") or []),
            database_name=path_state.get("target_db"),
            numeric_type_alignment=bool(state.get("numeric_type_alignment")),
            question=get_standalone_question(state),
            prediction_llm=state.get("llm"),
        )
        connector = connector_for(ctx)
        result = run_episode(
            state.get("reasoning_llm") or state["llm"],
            ctx,
            question=get_question_for_processing(state),
            evidence=state.get("evidence"),
            dialect=getattr(connector, "dialect", None),
            mode=episode_mode(state.get("prediction_override")),
            custom_prompts=state.get("custom_prompts"),
            sql_examples=state.get("sql_examples"),
            value_anchors=state.get("value_anchors"),
        )

        for step in result.trajectory:
            for call in step.get("tool_calls") or []:
                record_thought(path_state, STEP_NAME, _tool_thought(call))
        path_state["react_trajectory"] = result.trajectory
        path_state["react_end_reason"] = result.reason
        path_state["relevant_tables"] = list(ctx.seen_tables.values())

        if result.prediction is not None:
            response = result.prediction.response
            markdown = str(response.get("response") or "")
            path_state["formatted_response"] = markdown
            path_state["final_response"] = response
            return {
                "decision": "predicted",
                "messages": list(state.get("messages") or [])
                + [AIMessage(content=markdown)],
                "path_state": path_state,
            }

        submission = result.submission
        if submission is None:
            self.logger.info("ReAct episode ended without an answer: %s", result.reason)
            path_state["unconstructable_explanation"] = (
                ctx.prediction_refusal or _UNANSWERED
            )
            return {"decision": "unconstructable", "path_state": path_state}

        path_state["sql_code"] = submission.sql
        path_state["sql_generation_result"] = SQLGenerationModel(
            thought=submission.assumptions or _NO_ASSUMPTIONS,
            sql_code=submission.sql,
            response=submission.response or "Here is the result of the query.",
        )
        path_state["sql_response_from_db"] = submission.db_response.result
        path_state.setdefault("candidates", [])
        if ctx.numeric_type_alignment:
            path_state["result_numeric_kinds"] = submission.db_response.numeric_kinds
            path_state["result_numeric_kinds_sql"] = submission.sql
            path_state["result_values_digest"] = submission.db_response.values_digest
        return {"decision": "valid_sql", "path_state": path_state}
