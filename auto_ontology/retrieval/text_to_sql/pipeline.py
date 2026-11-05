# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The text-to-SQL pipeline: the ReAct agent, then the answer or the refusal.

The agent decides everything about the query itself; the pipeline only carries
its result to the node that formats the reply, and reports each step.
"""

import logging
from typing import Any, Callable, Dict, Iterator, Literal

from auto_ontology.retrieval.text_to_sql.state import (
    TextToSQLPayload,
    AgentState,
    get_question_for_processing,
)
from auto_ontology.retrieval.text_to_sql.agents.response import ResponseAgent
from auto_ontology.retrieval.text_to_sql.agents.sql_unconstructable import (
    SQLUnconstructableAgent,
)
from auto_ontology.retrieval.text_to_sql.base import agent_wrapper
from auto_ontology.retrieval.text_to_sql.react.agent import (
    STEP_NAME as REACT_STEP,
    ReactSQLAgent,
)

logger = logging.getLogger(__name__)

Step = Callable[[AgentState], Dict[str, Any]]
StepEvent = tuple[Literal["start", "end"], str, Dict[str, Any] | None]


def route_react(state: AgentState) -> str:
    """An answered episode is formatted; anything else is reported as unanswerable."""
    if state.get("decision") == "valid_sql":
        return "format_and_respond"
    return "unconstructable_sql_response"


def log_step_visit(state, step_name: str):
    """
    Track how many times each step was visited during a run.
    """
    path_state = state.get("path_state", {})
    counts = path_state.get("node_visit_counts", {})
    counts[step_name] = counts.get(step_name, 0) + 1
    path_state["node_visit_counts"] = counts
    state["path_state"] = path_state
    total = sum(counts.values())
    logger.info(f"🔁 Node visits: {counts} | Total visits this run: {total}")


def create_steps() -> dict[str, Step]:
    return {
        REACT_STEP: agent_wrapper(ReactSQLAgent()),
        "format_and_respond": agent_wrapper(ResponseAgent()),
        "unconstructable_sql_response": agent_wrapper(SQLUnconstructableAgent()),
    }


def run_steps(state: AgentState, steps: dict[str, Step]) -> Iterator[StepEvent]:
    """Run the agent and then the step that reports its outcome.

    Yields ``("start", step, None)`` before a step runs, so a client can label
    the work in progress, and ``("end", step, output)`` once it returns. Each
    output overwrites the matching top-level keys of ``state``, which is
    updated in place.
    """
    step_name: str | None = REACT_STEP
    while step_name:
        yield "start", step_name, None
        log_step_visit(state, step_name)
        output = steps[step_name](state) or {}
        state.update(output)  # type: ignore[typeddict-item]
        yield "end", step_name, output
        step_name = route_react(state) if step_name == REACT_STEP else None


__all__ = [
    "TextToSQLPayload",
    "AgentState",
    "create_steps",
    "get_question_for_processing",
    "route_react",
    "run_steps",
]
