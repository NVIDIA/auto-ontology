# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapter between GSF and the ``nemo_retriever`` text-to-SQL agent.

GSF carries a *list* of database connectors (``get_connectors()``) so it stays
multi-database ready, but the ``nemo_retriever`` text-to-SQL agent operates on a
single connector: its ``_build_state`` reads ``payload["connector"]`` and needs
``connector.dialect``. This shim lets call sites pass the plural ``connectors``
and derives the singular ``connector`` the agent requires, in one place.

If a future ``nemo_retriever`` version consumes ``connectors`` directly, only
this adapter needs to change — not the worker payload.
"""

from __future__ import annotations

from typing import Any, Generator

from nemo_retriever.tabular_data.retrieval.text_to_sql.main import (
    stream_agent_response as _nemo_stream_agent_response,
)


def stream_agent_response(payload: dict[str, Any]) -> Generator[Any, None, None]:
    """Run the text-to-SQL agent, accepting a plural ``connectors`` payload.

    Accepts either ``connector`` (singular, passed straight through) or
    ``connectors`` (a list, from which the first is used, since the agent
    operates on a single database at a time). The ``connectors`` key is
    removed before delegating so the underlying agent only sees what it reads.
    """

    adapted = dict(payload)
    connectors = adapted.pop("connectors", None)
    if not adapted.get("connector") and connectors:
        adapted["connector"] = connectors[0]
    return _nemo_stream_agent_response(adapted)
