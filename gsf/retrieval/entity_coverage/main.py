# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Entry point for the entity-coverage LangGraph pipeline."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage

from gsf.retrieval.data_access.custom_analyses import fetch_custom_analyses
from gsf.retrieval.entity_coverage.graph import create_graph
from gsf.retrieval.entity_coverage.state import (
    DEFAULT_MAX_DISTANCE,
    EntityCoveragePayload,
)
from gsf.retrieval.text_to_sql.state import AgentState
from gsf.utils.llm_invoke import get_llm_client

logger = logging.getLogger(__name__)

try:
    llm_client = get_llm_client()
except (ValueError, EnvironmentError, OSError) as exc:
    logger.error("Failed to initialize LLM client: %s", exc)
    llm_client = None

graph = create_graph()
app = graph.compile()


def _build_state(payload: EntityCoveragePayload) -> AgentState:
    connectors = payload.get("connectors", [])
    if not connectors:
        raise ValueError(
            "EntityCoveragePayload is missing required 'connectors'. "
            "Provide a non-empty list of database connectors."
        )

    data_retriever = payload.get("data_retriever")
    if data_retriever is None:
        raise ValueError("EntityCoveragePayload is missing required 'data_retriever'.")

    semantic_retriever = payload.get("semantic_retriever")
    if semantic_retriever is None:
        logger.warning(
            "No 'semantic_retriever' in payload — candidate searches will be skipped."
        )

    acronyms = payload.get("acronyms", [])
    domain_rules = fetch_custom_analyses() + list(acronyms or [])

    initial_path_state = dict(payload.get("path_state") or {})
    max_distance = payload.get("max_distance", DEFAULT_MAX_DISTANCE)
    initial_path_state["max_distance"] = float(max_distance)
    initial_path_state["return_uncovered_entities"] = bool(
        payload.get("return_uncovered_entities", False)
    )

    target_db = payload.get("target_db")
    if target_db:
        initial_path_state["target_db"] = target_db
    elif len(connectors) == 1:
        connector_db = getattr(connectors[0], "database_name", None)
        if connector_db:
            initial_path_state["target_db"] = connector_db

    state: dict[str, Any] = {
        "llm": llm_client,
        "initial_question": payload["question"],
        "connectors": connectors,
        "messages": [HumanMessage(content=payload["question"])],
        "path_state": initial_path_state,
        "data_retriever": data_retriever,
        "semantic_retriever": semantic_retriever,
        "decision": "",
        "domain_rules": domain_rules,
        "glossary": list(acronyms or []),
    }
    return state  # type: ignore[return-value]


def get_coverage_response(payload: EntityCoveragePayload) -> dict:
    """Run the entity-coverage graph and return coverage and ranked candidates."""
    if llm_client is None:
        raise RuntimeError("LLM client is not configured.")

    state = _build_state(payload)
    final_state = app.invoke(state, config={"recursion_limit": 10})

    path_state = final_state.get("path_state", {})
    final_response = path_state.get("final_response")
    if isinstance(final_response, dict):
        return final_response
    return {
        "coverage": 0.0,
        "candidates": [],
    }


__all__ = [
    "get_coverage_response",
    "app",
    "graph",
    "llm_client",
]
