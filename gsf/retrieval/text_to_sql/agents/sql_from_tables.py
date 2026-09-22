# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
SQL Generation from Tables Agent

This agent generates SQL queries from table schemas when no snippets are available.
Used as a fallback when attribute snippets don't exist or aren't sufficient.

Responsibilities:
- Generate SQL from table schemas and relationships
- Find similar questions from conversation history
- Handle cases where snippets are not available
- Store SQL response in path_state

Design Decisions:
- Used when no suitable snippets are found
- Relies on table schemas
- Can incorporate similar questions from history for context
"""

import logging
from typing import Any, Dict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from gsf.retrieval.text_to_sql.connector_routing import resolve_connector_from_tables
from gsf.retrieval.text_to_sql.formatters_util import format_tables_for_prompt
from gsf.utils.llm_invoke import invoke_with_structured_output
from gsf.retrieval.text_to_sql.base import BaseAgent
from gsf.retrieval.text_to_sql.models import SQLGenerationModel
from gsf.retrieval.text_to_sql.state import (
    AgentState,
    get_original_question,
    get_question_for_processing,
)
from gsf.retrieval.text_to_sql.prompts import (
    create_sql_from_candidates_prompt,
    create_sql_general_prompt,
    create_sql_user_prompt,
    format_authoritative_evidence,
    format_dialect_rules,
    format_dual_question_block,
    format_projection_rules,
)
from gsf.retrieval.data_access.relevant_tables import get_relevant_tables

logger = logging.getLogger(__name__)


class SQLFromTablesAgent(BaseAgent):
    """
    Agent that generates SQL from table schemas.

    This agent is used when no suitable semantic entities are available.
    It builds SQL from table schemas and similar questions.

    Input Requirements:
    - path_state["relevant_tables"]: Optional relevant tables (if not provided, will search)
    - path_state["error"]: Optional error from previous attempt (for reconstruction)
    - state["initial_question"]: User's question
    - state["connectors"]: List of connectors (the first is used for dialect/database_name)

    Output:
    - path_state["sql_generation_result"]: SQL response with SQL code
    - path_state["relevant_tables"]: Relevant tables used
    - decision: "constructable" or "unconstructable"
    """

    def __init__(self):
        super().__init__("sql_from_tables")

    def execute(self, state: AgentState) -> Dict[str, Any]:
        """
        Generate SQL from table schemas.

        Uses table schemas and similar questions to generate SQL
        when no semantic entities are available.

        Args:
            state: Current agent state

        Returns:
            Dictionary with:
            - path_state: Contains SQL response, tables, connection
            - messages: Adds SQL response to messages
            - decision: "constructable" or "unconstructable"
        """
        path_state = state.get("path_state", {})
        llm = state["llm"]
        connectors = state.get("connectors") or []
        original_question = get_original_question(state)
        question = get_question_for_processing(state)
        evidence = state["evidence"]
        has_evidence = bool(evidence)
        main_question = (
            format_dual_question_block(original_question, question)
            if has_evidence
            else question
        )

        # Get relevant tables (search if not already available)
        relevant_tables = path_state.get("relevant_tables", [])
        if not len(relevant_tables):
            relevant_tables = get_relevant_tables(
                state["data_retriever"],
                question,
                database_name=path_state.get("target_db"),
            )
        similar_questions = []

        connector = resolve_connector_from_tables(relevant_tables, connectors)
        dialect = getattr(connector, "dialect", None)
        target_db = path_state.get("target_db")

        if has_evidence:
            system_prompt = create_sql_from_candidates_prompt(
                dialect=dialect,
                target_db=target_db,
                has_sql_examples=bool(state.get("sql_examples")),
            )
        else:
            system_prompt = create_sql_general_prompt

        observation_block = ""

        # Build user prompt with formatted tables
        user_prompt = create_sql_user_prompt.format(
            dialect=dialect,
            dialect_rules=format_dialect_rules(dialect),
            projection_rules=format_projection_rules(
                state.get("detailed_answers", True)
            ),
            main_question=main_question,
            observation_block=observation_block,
            queries=[],
            tables=format_tables_for_prompt(
                relevant_tables,
                target_db=path_state.get("target_db"),
                dialect=dialect,
            ),
            important_columns="No semantically matched columns.",
            join_paths="",
            qa_from_conversations=similar_questions,
            custom_analyses="",
        )

        messages = state["messages"] + [SystemMessage(content=system_prompt)]
        if evidence:
            messages.append(
                SystemMessage(content=format_authoritative_evidence(evidence))
            )
        messages.append(HumanMessage(content=user_prompt))

        response = invoke_with_structured_output(llm, messages, SQLGenerationModel)

        self.logger.info(
            "SQL generated from tables: %s...",
            response.sql_code[:100] if response and response.sql_code else "None",
        )

        if response and response.sql_code:
            return {
                "messages": messages + [AIMessage(content=response.response)],
                "path_state": {
                    **path_state,
                    "sql_generation_result": response,
                    "relevant_tables": relevant_tables,
                },
                "decision": "constructable",
            }
        else:
            return {
                "path_state": {
                    **path_state,
                    "unconstructable_explanation": getattr(
                        response, "response", "LLM failed to produce SQL."
                    ),
                },
                "decision": "unconstructable",
            }
