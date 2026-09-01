# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Chat-route helpers: request models, node labels, and Message-2 formatting."""

from __future__ import annotations

import json
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

__all__ = [
    "NODE_LABELS",
    "ChatRequest",
    "build_result_message",
    "charts_to_fenced_content",
    "stringify_sql_response",
]


class ChatRequest(BaseModel):
    """Payload sent by the frontend to start a chat completion."""

    question: str = Field(..., min_length=1)
    conversation_id: UUID | None = Field(
        default=None,
        description=(
            "Stable thread UUID. When supplied, prior completed turns for it are "
            "loaded and the new turn — including the chart/table bubble, once "
            "ready — is persisted to it. An unknown UUID creates the "
            "conversation; one owned by another user is rejected with 404. "
            "Omitting it runs the question statelessly, with no history and no "
            "chart step."
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


# Message 2 (the chart, or the raw result table when there's nothing to plot)
# is built here and here alone — ``_pump`` calls these directly and persists
# the result itself, so there is no separate client-side formatter to keep in
# sync with anymore (see ``gsf/server/chat/router.py``'s ``_build_charts_event``).


def stringify_sql_response(value: Any) -> str | None:
    """Normalise an executed-SQL result into the string persisted to Postgres.

    ``value`` is usually a one-item list containing a JSON-records string
    (``sql_response_from_db``), but may already be a plain string. Compact
    JSON for anything else, matching ``stringifySqlResponse`` in
    ``answerMessages.ts`` so the DB format is identical either way.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value if value.strip() else None
    try:
        return json.dumps(value)
    except (TypeError, ValueError):
        return None


def charts_to_fenced_content(charts: list[dict[str, Any]]) -> str:
    """Embed each chart spec in a ```chart fenced block the UI knows to render."""
    return "\n\n".join(f"```chart\n{json.dumps(spec)}\n```" for spec in charts)


def build_result_message(
    sql_response_from_db: Any, charts: list[dict[str, Any]] | None
) -> tuple[str, str | None] | None:
    """Build Message 2's ``(content, sql_response)`` for persistence + SSE.

    Returns ``None`` when there is nothing to show — no chart and no
    executed result to fall back to as a table.
    """
    if charts:
        return charts_to_fenced_content(charts), None
    sql_response = stringify_sql_response(sql_response_from_db)
    if sql_response:
        return "", sql_response
    return None
