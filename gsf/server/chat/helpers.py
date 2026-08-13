# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Chat-route helpers: request models and node labels."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

# Re-exported so the router (and any other server-side consumer) can keep
# importing NODE_LABELS from here. The dict itself now lives next to the
# graph definition (``gsf.retrieval.text_to_sql.node_labels``) so the agent
# pipeline can also use it — e.g. when building the final run's ``thoughts``
# summary in ``stream_agent_response`` — without the retrieval layer having
# to depend on the server layer.
from gsf.retrieval.text_to_sql.node_labels import NODE_LABELS

__all__ = ["NODE_LABELS", "ChatRequest", "VisualizeRequest"]


class ChatRequest(BaseModel):
    """Payload sent by the frontend to start a chat completion."""

    question: str = Field(..., min_length=1)
    conversation_id: UUID | None = Field(
        default=None,
        description=(
            "Stable thread UUID. When supplied through the authenticated gateway, "
            "prior completed turns are loaded and the new turn is persisted."
        ),
    )
    # Force the prediction/SQL branch instead of classifying the question:
    # True  -> go straight to the KumoRFM prediction flow
    # False -> go straight to the regular text-to-SQL flow
    # None  -> classify as usual (default)
    prediction: bool | None = None
    # Scope retrieval/SQL to one connected database. When omitted (and more
    # than one connector is loaded), the pipeline does not pin a database.
    target_db: str | None = Field(
        default=None,
        description=(
            "Catalog database UUID or database name used to scope retrieval and SQL."
        ),
    )


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
