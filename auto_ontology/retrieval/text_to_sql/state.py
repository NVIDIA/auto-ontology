# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Agent state and API payload types.

Kept separate from ``pipeline.py`` to avoid circular imports (agents
import state; the pipeline imports agents).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NotRequired, TypedDict

from langchain_core.messages import HumanMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from auto_ontology.utils.retriever import Retriever
from auto_ontology.connectors.base import SQLDatabase


class AgentRequest(TypedDict):
    """One question for the agent, and which sources it may answer it from.

    Named for what it is rather than for what it used to be. The agent began
    as a text-to-SQL flow and grew two sources that write no SQL at all --
    *information* answers from the catalog in words, *conversation* from this
    database's past transcripts -- so a caller may switch *calculation* off
    entirely and never reach a query. ``TextToSQLPayload`` is kept below as an
    alias because the name is part of the published surface.
    """

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
    # The three things the agent may do, each the caller's to switch. At least
    # one must be on.
    #
    # *calculation* is querying the data, and needs ``connectors``. *information*
    # is answering in words from the catalog. *conversation* is answering in
    # words from this database's past conversations; it defaults off because
    # most databases have no conversation memory, and a source offered with
    # nothing behind it is a distraction the model pays for on every question.
    # The other two default on, which is what the agent has always done.
    calculation: NotRequired[bool]
    information: NotRequired[bool]
    conversation: NotRequired[bool]
    # The older spelling of ``calculation=True, information=False,
    # conversation=False``: every answer must be a query, which is what a
    # grader that scores the SQL text needs. Overrides all three rather than
    # sitting beside them.
    calculation_only: NotRequired[bool]
    acronyms: NotRequired[list[dict[str, str]]]
    custom_prompts: NotRequired[str]
    # Solved question/SQL pairs from other databases, shown as query-shape
    # precedent: ``question``, ``sql`` and optional ``evidence`` and ``db``.
    sql_examples: NotRequired[list[dict[str, str]]]
    # Question phrases looked up in the target database: ``phrase``, ``tbl``,
    # ``col``, ``stored_value``, and optional ``kind`` (value, contains or
    # absent), ``n_rows`` and ``description``.
    value_anchors: NotRequired[list[dict[str, str]]]
    # True for graders that compare result columns: adds strict output-shape rules.
    shorten_answer: NotRequired[bool]


#: The name this type was published under, kept working.
#:
#: Callers live outside this repository and a TypedDict is structural, so a
#: rename that only moved the name would break their imports while their
#: dicts stayed valid -- the least useful kind of breakage.
TextToSQLPayload = AgentRequest


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
    calculation: NotRequired[bool]
    information: NotRequired[bool]
    conversation: NotRequired[bool]
    calculation_only: NotRequired[bool]
    # The payload's custom prompts and acronyms as one system-prompt section.
    custom_prompts: NotRequired[str]
    sql_examples: NotRequired[list[dict[str, str]]]
    value_anchors: NotRequired[list[dict[str, str]]]
    shorten_answer: NotRequired[bool]
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


#: What each of the three may do when the caller says nothing.
#:
#: The defaults differ because the history does. Querying the data and
#: answering from the catalog have always been on, and turning either off would
#: change what existing callers get; conversation search is new, and most
#: databases have no conversation memory to search.
SOURCE_DEFAULTS = {"calculation": True, "information": True, "conversation": False}


def resolve_sources(state: Mapping[str, Any]) -> dict[str, bool]:
    """Which of the three an episode may use, from a payload or an agent state.

    One function for both, because the payload's keys are copied into the state
    unchanged and two readings of the same three flags would eventually
    disagree.

    ``calculation_only`` is the older spelling of the first alone, and
    overrides the three rather than sitting beside them: it is set by graders
    that score the SQL text, and a grader that cannot read a prose answer
    cannot read one the caller also asked for. Two spellings disagreeing is a
    caller's mistake, and the safe reading is the one that still produces a
    query.
    """
    if state.get("calculation_only"):
        return {"calculation": True, "information": False, "conversation": False}
    return {
        name: bool(state.get(name, default))
        for name, default in SOURCE_DEFAULTS.items()
    }


__all__ = [
    "AgentRequest",
    "TextToSQLPayload",
    "AgentState",
    "SOURCE_DEFAULTS",
    "resolve_sources",
    "get_standalone_question",
    "get_question_for_processing",
]
