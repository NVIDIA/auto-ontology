# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""User-facing labels for pipeline node names.

Lives next to ``pipeline.py`` (the source of truth for node names)
rather than in ``auto_ontology.server.chat`` so the agent pipeline itself — e.g.
``main.py``'s ``stream_agent_response`` when it builds the final
``thoughts`` summary — can use the same labels without depending on the
server layer. ``auto_ontology.server.chat.helpers`` re-exports this for the router.

1-to-1 with the nodes added inside ``create_steps``. Unknown
nodes fall through to the raw node_name in callers so we never display a
blank thinking step.
"""

from __future__ import annotations

NODE_LABELS: dict[str, str] = {
    "react_sql_agent": "Searching the data and writing SQL",
    "format_and_respond": "Formatting response",
    "unconstructable_sql_response": "SQL could not be constructed",
}

__all__ = ["NODE_LABELS"]
