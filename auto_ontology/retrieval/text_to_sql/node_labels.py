# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""User-facing labels for LangGraph node names.

Lives next to ``text_to_sql_graph.py`` (the source of truth for node names)
rather than in ``auto_ontology.server.chat`` so the agent pipeline itself — e.g.
``main.py``'s ``stream_agent_response`` when it builds the final
``thoughts`` summary — can use the same labels without depending on the
server layer. ``auto_ontology.server.chat.helpers`` re-exports this for the router.

1-to-1 with the agent classes instantiated inside ``create_graph``. Unknown
nodes fall through to the raw node_name in callers so we never display a
blank thinking step.
"""

from __future__ import annotations

NODE_LABELS: dict[str, str] = {
    "question_intent": "Classifying the question",
    "question_extraction": "Understanding the question",
    "prepare_prediction_graph": "Preparing prediction",
    "kumo_predict": "Preparing prediction",
    "retrieve_candidates": "Retrieving candidates",
    "prepare_candidates": "Preparing candidates",
    "refine_evidence": "Refining evidence",
    "precheck_combined": "Checking joins and filter values",
    "construct_sql_from_candidates": "Constructing SQL",
    "reconstruct_sql": "Reconstructing SQL",
    "validate_sql_query": "Validating SQL and intent",
    "execute_sql_query": "Executing SQL",
    "check_empty_like_result": "Checking results",
    "check_value_repair": "Checking values",
    "format_and_respond": "Formatting response",
    "unconstructable_sql_response": "SQL could not be constructed",
}

__all__ = ["NODE_LABELS"]
