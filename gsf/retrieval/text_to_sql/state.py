# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
LangGraph agent state and API payload types.

Kept separate from ``graph.py`` to avoid circular imports (agents import state;
``graph`` imports agents).
"""

from __future__ import annotations

from typing import NotRequired, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from nemo_retriever.graph.retriever import Retriever
from nemo_retriever.tabular_data.sql_database import SQLDatabase


class AgentPayload(TypedDict):
    """Payload for the ingest/legacy retrieval flow (single retriever)."""

    question: str
    retriever: Retriever
    path_state: NotRequired[dict]
    connectors: NotRequired[list[SQLDatabase]]
    acronyms: NotRequired[list[dict[str, str]]]
    custom_prompts: NotRequired[str]


class TextToSQLPayload(TypedDict):
    """Payload for the text-to-SQL agent flow (data + semantic retrievers)."""

    question: str
    data_retriever: Retriever
    semantic_retriever: NotRequired[Retriever]
    path_state: NotRequired[dict]
    connectors: NotRequired[list[SQLDatabase]]
    acronyms: NotRequired[list[dict[str, str]]]
    custom_prompts: NotRequired[str]
    target_db: NotRequired[str]


class AgentState(TypedDict):
    """State object passed through the LangGraph."""

    llm: ChatNVIDIA
    non_reasoning_llm: NotRequired[BaseChatModel]
    initial_question: str
    messages: list[HumanMessage]
    decision: str
    connectors: list[SQLDatabase]
    path_state: dict
    data_retriever: Retriever
    semantic_retriever: Retriever
    domain_rules: list[dict[str, str]]


def get_original_question(state: AgentState) -> str:
    """Raw user question as submitted, without sanitization."""
    return state.get("initial_question", "")


def get_question_for_processing(state: AgentState) -> str:
    """
    Question string for retrieval and semantic search.

    Uses ``path_state["normalized_question"]`` when set (e.g. after sanitization),
    otherwise ``initial_question``.
    """
    path_state = state.get("path_state", {})
    normalized_question = path_state.get("normalized_question")
    if normalized_question:
        return normalized_question
    return state.get("initial_question", "")


def rules_to_text(rules: list[dict[str, str]]) -> str:
    """Convert a list of ``{"name": ..., "description": ...}`` rules to a prompt string."""
    if not rules:
        return ""
    parts = []
    for rule in rules:
        parts.append(f"## {rule['name']}\n{rule['description']}")
    return "\n\n".join(parts) + "\n\n"


__all__ = [
    "AgentPayload",
    "TextToSQLPayload",
    "AgentState",
    "get_original_question",
    "get_question_for_processing",
    "rules_to_text",
]
