# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LangGraph for the entity-coverage pipeline."""

from __future__ import annotations

import logging

from langgraph.graph import END, StateGraph

from auto_ontology.retrieval.entity_coverage.agents.coverage import CoverageGradeAgent
from auto_ontology.retrieval.entity_coverage.agents.question_extraction import (
    QuestionExtractionAgent,
)
from auto_ontology.retrieval.text_to_sql.agents.candidates_retrieval import (
    CandidateRetrievalAgent,
)
from auto_ontology.retrieval.text_to_sql.base import agent_wrapper
from auto_ontology.retrieval.text_to_sql.state import AgentState

logger = logging.getLogger(__name__)


def create_graph() -> StateGraph:
    """Build: question_extraction → retrieve_candidates → coverage_grade → END."""
    graph = StateGraph(AgentState)

    graph.add_node(
        "question_extraction",
        agent_wrapper(QuestionExtractionAgent(include_subject=False)),
    )
    graph.add_node(
        "retrieve_candidates",
        agent_wrapper(CandidateRetrievalAgent()),
    )
    graph.add_node(
        "coverage_grade",
        agent_wrapper(CoverageGradeAgent()),
    )

    graph.set_entry_point("question_extraction")
    graph.add_edge("question_extraction", "retrieve_candidates")
    graph.add_edge("retrieve_candidates", "coverage_grade")
    graph.add_edge("coverage_grade", END)

    logger.info("Entity-coverage graph created")
    return graph
