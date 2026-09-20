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

from langchain_core.messages import HumanMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from nemo_retriever.graph.retriever import Retriever
from gsf.connectors.base import SQLDatabase


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
    evidence: NotRequired[str]
    # Question/SQL pairs retrieved as structural precedent for this question,
    # each a ``{"question": ..., "sql": ...}`` mapping. Unlike `evidence` these
    # are advisory and usually come from *other* databases, so they inform
    # query shape only — never identifiers or filter values.
    sql_examples: NotRequired[list[dict[str, str]]]
    value_anchors: NotRequired[list[dict[str, str]]]
    # The clarified/merged question on its own — no hint blocks or SQL
    # references mixed in (unlike `evidence`, which carries those too).
    # Omitted by callers that never enrich the question.
    enriched_question: NotRequired[str]
    processing_question: NotRequired[str]
    data_retriever: Retriever
    semantic_retriever: NotRequired[Retriever]
    path_state: NotRequired[dict]
    connectors: NotRequired[list[SQLDatabase]]
    acronyms: NotRequired[list[dict[str, str]]]
    custom_prompts: NotRequired[str]
    target_db: NotRequired[str]
    # Force the branch instead of classifying: True -> prediction, False -> SQL,
    # None/absent -> classify.
    prediction: NotRequired[bool | None]


class AgentState(TypedDict):
    """State object passed through the LangGraph."""

    llm: ChatNVIDIA
    initial_question: str
    evidence: NotRequired[str]
    sql_examples: NotRequired[list[dict[str, str]]]
    value_anchors: NotRequired[list[dict[str, str]]]
    enriched_question: NotRequired[str]
    messages: list[HumanMessage]
    decision: str
    # Caller-supplied branch override; see TextToSQLPayload.prediction.
    prediction_override: NotRequired[bool | None]
    connectors: list[SQLDatabase]
    path_state: dict
    data_retriever: Retriever
    semantic_retriever: Retriever
    domain_rules: list[dict[str, str]]
    # User-curated Glossary definitions (the ``acronyms`` table), kept apart from
    # domain_rules so prompts can inject them without the custom-analysis SQL.
    # After ``question_extraction``, this is narrowed to the entries the LLM used.
    glossary: NotRequired[list[dict[str, str]]]


def get_original_question(state: AgentState) -> str:
    """Raw user question as submitted, without sanitization."""
    return state.get("initial_question", "")


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
    "get_standalone_question",
    "get_question_for_processing",
    "rules_to_text",
]
