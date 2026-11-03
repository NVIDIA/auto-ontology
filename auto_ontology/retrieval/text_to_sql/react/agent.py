# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The graph node that answers a question with one ReAct episode.

It replaces the whole fixed pipeline — intent, extraction, retrieval,
preparation, construction, validation, execution and repair — and hands its
result to ``format_and_respond`` in the shape that node already reads, so
everything downstream of the answer is unchanged.
"""

from __future__ import annotations

from typing import Any, Dict

from auto_ontology.retrieval.text_to_sql.base import BaseAgent, record_thought
from auto_ontology.retrieval.text_to_sql.models import SQLGenerationModel
from auto_ontology.retrieval.text_to_sql.react.context import ToolContext
from auto_ontology.retrieval.text_to_sql.react.loop import run_episode
from auto_ontology.retrieval.text_to_sql.react.run_sql import connector_for
from auto_ontology.retrieval.text_to_sql.state import (
    AgentState,
    get_question_for_processing,
)

NODE_NAME = "react_sql_agent"

_UNANSWERED = (
    "I couldn't write a query that answers this question from the data I could "
    "find. Try rephrasing it, or name the table or measure you mean."
)

_NO_ASSUMPTIONS = "No assumptions were needed beyond what the question states."


def _tool_thought(call: dict[str, Any]) -> str:
    args = call.get("args") or {}
    if call.get("name") == "retrieve":
        return f"Searching for: {args.get('query', '')}"
    if call.get("name") == "run_sql":
        return "Running a query to check the data."
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
            connectors=list(state.get("connectors") or []),
            database_name=path_state.get("target_db"),
            numeric_type_alignment=bool(state.get("numeric_type_alignment")),
        )
        connector = connector_for(ctx)
        result = run_episode(
            state.get("reasoning_llm") or state["llm"],
            ctx,
            question=get_question_for_processing(state),
            evidence=state.get("evidence"),
            dialect=getattr(connector, "dialect", None),
        )

        for step in result.trajectory:
            for call in step.get("tool_calls") or []:
                record_thought(path_state, NODE_NAME, _tool_thought(call))
        path_state["react_trajectory"] = result.trajectory
        path_state["react_end_reason"] = result.reason
        path_state["relevant_tables"] = list(ctx.seen_tables.values())

        submission = result.submission
        if submission is None:
            self.logger.info("ReAct episode ended without an answer: %s", result.reason)
            path_state["unconstructable_explanation"] = _UNANSWERED
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
