# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Chat-route helpers: request models and node labels."""

from __future__ import annotations

from pydantic import BaseModel, Field

# Maps LangGraph node names from
# nemo_retriever.tabular_data.retrieval.text_to_sql.text_to_sql_graph
# to a single user-facing label per agent (1-to-1 with the agent classes
# instantiated inside ``create_graph``). Unknown nodes fall through to the
# raw node_name in the router so we never display a blank thinking step.
NODE_LABELS: dict[str, str] = {
    "entities_extraction": "Extracting entities",
    "retrieve_candidates": "Retrieving candidates",
    "prepare_candidates": "Preparing candidates",
    "construct_sql_from_candidates": "Constructing SQL from candidates",
    "construct_sql_not_from_snippets": "Constructing SQL from tables",
    "reconstruct_sql": "Reconstructing SQL",
    "validate_sql_query": "Validating SQL",
    "validate_intent": "Validating intent",
    "execute_sql_query": "Executing SQL",
    "format_and_respond": "Formatting response",
    "unconstructable_sql_response": "SQL could not be constructed",
}


class ChatRequest(BaseModel):
    """Payload sent by the frontend to start a chat completion."""

    question: str = Field(..., min_length=1)
    zone_ids: list[str] = Field(
        default=[],
        description="Zone IDs the requesting user has access to; used for scoped retrieval",
    )
