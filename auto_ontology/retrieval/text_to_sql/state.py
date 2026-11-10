# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Agent state and API payload types.

Kept separate from ``pipeline.py`` to avoid circular imports (agents
import state; the pipeline imports agents).
"""

from __future__ import annotations

from typing import NotRequired, TypedDict

from langchain_core.messages import HumanMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from auto_ontology.utils.retriever import Retriever
from auto_ontology.connectors.base import SQLDatabase


class TextToSQLPayload(TypedDict):
    """Payload for the text-to-SQL agent flow (data + semantic retrievers)."""

    question: str
    evidence: NotRequired[str]
    # For graders that compare numbers exactly: rewrite explicit float casts in
    # the generated Postgres SQL to NUMERIC (see ``numeric_types``).
    numeric_type_alignment: NotRequired[bool]
    processing_question: NotRequired[str]
    data_retriever: Retriever
    semantic_retriever: NotRequired[Retriever]
    path_state: NotRequired[dict]
    connectors: NotRequired[list[SQLDatabase]]
    target_db: NotRequired[str]
    # True forces a KumoRFM prediction, False forbids one, None lets the agent
    # choose when KumoRFM is configured.
    prediction: NotRequired[bool | None]
    # Accepted from callers but not read by the ReAct agent.
    sql_examples: NotRequired[list[dict[str, str]]]
    value_anchors: NotRequired[list[dict[str, str]]]
    acronyms: NotRequired[list[dict[str, str]]]
    custom_prompts: NotRequired[str]


class AgentState(TypedDict):
    """State object passed from node to node."""

    llm: ChatNVIDIA
    reasoning_llm: NotRequired[ChatNVIDIA]
    initial_question: str
    evidence: NotRequired[str]
    numeric_type_alignment: NotRequired[bool]
    messages: list[HumanMessage]
    decision: str
    connectors: list[SQLDatabase]
    path_state: dict
    data_retriever: Retriever
    semantic_retriever: Retriever
    prediction_override: NotRequired[bool | None]
    # Read by the entity-coverage pipeline, which shares this state type.
    domain_rules: NotRequired[list[dict[str, str]]]
    # User-curated Glossary definitions (the ``acronyms`` table), kept apart from
    # domain_rules so prompts can inject them without the custom-analysis SQL.
    # After ``question_extraction``, this is narrowed to the entries the LLM used.
    glossary: NotRequired[list[dict[str, str]]]


def get_standalone_question(state: AgentState) -> str:
    """
    Question carrying the full intent, before sanitization.

    Uses the caller-provided ``path_state["processing_question"]`` (a follow-up
    resolved against conversation history) when set, otherwise the exact submitted
    ``initial_question``. Use this where a node needs the complete question but not
    the retrieval-oriented rewrite ``get_question_for_processing`` returns.
    """
    path_state = state.get("path_state", {})
    initial_question = state.get("initial_question", "")
    processing_question = path_state.get("processing_question")

    return processing_question or initial_question


def get_question_for_processing(state: AgentState) -> str:
    """
    Question string for retrieval and semantic search.

    Uses ``path_state["normalized_question"]`` when set (e.g. after extraction),
    otherwise falls back to the standalone question.
    """
    path_state = state.get("path_state", {})
    normalized_question = path_state.get("normalized_question")

    return normalized_question or get_standalone_question(state)


__all__ = [
    "TextToSQLPayload",
    "AgentState",
    "get_standalone_question",
    "get_question_for_processing",
]
