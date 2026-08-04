# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Chat-route helpers: request models and node labels."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

# Maps LangGraph node names from
# nemo_retriever.tabular_data.retrieval.text_to_sql.text_to_sql_graph
# to a single user-facing label per agent (1-to-1 with the agent classes
# instantiated inside ``create_graph``). Unknown nodes fall through to the
# raw node_name in the router so we never display a blank thinking step.
NODE_LABELS: dict[str, str] = {
    "sanitize_question": "Understanding the question",
    "classify_prediction": "Checking for a prediction",
    "prepare_prediction_graph": "Preparing prediction graph",
    "kumo_predict": "Predicting with KumoRFM",
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
    conversation_id: str | None = None
    # Force the prediction/SQL branch instead of classifying the question:
    # True  -> go straight to the KumoRFM prediction flow
    # False -> go straight to the regular text-to-SQL flow
    # None  -> classify as usual (default)
    prediction: bool | None = None
    # Scope retrieval/SQL to one connected database. When omitted (and more
    # than one connector is loaded), the pipeline does not pin a database.
    target_db: str | None = None


class VisualizeRequest(BaseModel):
    """Payload for the second step: chart generation.

    Sent by the client once it already has the SQL and its executed result
    from step 1 (``ChatRequest`` / ``/chat/completions``).
    """

    question: str = Field(..., min_length=1)
    sql: str = Field(default="")
    result: Any = Field(
        default=None,
        description=(
            "The executed SQL result, i.e. the `sql_response_from_db` from the "
            "step 1 answer — either a one-item list containing a JSON-records "
            "string, or a list of row dicts."
        ),
    )
