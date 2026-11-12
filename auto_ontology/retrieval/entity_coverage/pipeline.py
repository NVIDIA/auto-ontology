# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The entity-coverage pipeline: question_extraction → retrieve_candidates → coverage_grade."""

from __future__ import annotations

from typing import Any, Callable

from auto_ontology.retrieval.entity_coverage.agents.coverage import CoverageGradeAgent
from auto_ontology.retrieval.entity_coverage.agents.question_extraction import (
    QuestionExtractionAgent,
)
from auto_ontology.retrieval.text_to_sql.agents.candidates_retrieval import (
    CandidateRetrievalAgent,
)
from auto_ontology.retrieval.text_to_sql.base import agent_wrapper
from auto_ontology.retrieval.text_to_sql.state import AgentState

Step = Callable[[AgentState], dict[str, Any]]


def create_steps() -> list[Step]:
    """The steps in the order they run."""
    return [
        agent_wrapper(QuestionExtractionAgent(include_subject=False)),
        agent_wrapper(CandidateRetrievalAgent()),
        agent_wrapper(CoverageGradeAgent()),
    ]


def run(state: AgentState, steps: list[Step]) -> AgentState:
    """Run every step on a copy of ``state`` and return the final state.

    Each step's output overwrites the matching top-level keys.
    """
    final_state = dict(state)
    for step in steps:
        final_state.update(step(final_state) or {})  # type: ignore[arg-type]
    return final_state  # type: ignore[return-value]
