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
    NODE_NAME as REACT_NODE_NAME,
    ReactSQLAgent,
)

logger = logging.getLogger(__name__)

Node = Callable[[AgentState], Dict[str, Any]]
Step = tuple[Literal["start", "end"], str, Dict[str, Any] | None]


def route_react(state: AgentState) -> str:
    """An answered episode is formatted; anything else is reported as unanswerable."""
    if state.get("decision") == "valid_sql":
        return "format_and_respond"
    return "unconstructable_sql_response"


def log_node_visit(state, node_name: str):
    """
    Track how many times each node was visited during a run.
    """
    path_state = state.get("path_state", {})
    counts = path_state.get("node_visit_counts", {})
    counts[node_name] = counts.get(node_name, 0) + 1
    path_state["node_visit_counts"] = counts
    state["path_state"] = path_state
    total = sum(counts.values())
    logger.info(f"🔁 Node visits: {counts} | Total visits this run: {total}")


def create_nodes() -> dict[str, Node]:
    return {
        REACT_NODE_NAME: agent_wrapper(ReactSQLAgent()),
        "format_and_respond": agent_wrapper(ResponseAgent()),
        "unconstructable_sql_response": agent_wrapper(SQLUnconstructableAgent()),
    }


def run_steps(state: AgentState, nodes: dict[str, Node]) -> Iterator[Step]:
    """Run the agent and then the node that reports its outcome.

    Yields ``("start", node, None)`` before a node runs, so a client can label
    the work in progress, and ``("end", node, output)`` once it returns. Each
    output overwrites the matching top-level keys of ``state``, which is
    updated in place.
    """
    node_name: str | None = REACT_NODE_NAME
    while node_name:
        yield "start", node_name, None
        log_node_visit(state, node_name)
        output = nodes[node_name](state) or {}
        state.update(output)  # type: ignore[typeddict-item]
        yield "end", node_name, output
        node_name = route_react(state) if node_name == REACT_NODE_NAME else None


__all__ = [
    "TextToSQLPayload",
    "AgentState",
    "create_nodes",
    "get_question_for_processing",
    "route_react",
    "run_steps",
]
