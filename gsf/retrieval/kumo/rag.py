# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Retrieval of verified PQL examples (few-shot RAG for PQL generation).

Semantic-searches the ``PqlAnalysis`` corpus for questions similar to the user's,
then returns them as ``{question, query, reasoning}`` dicts — the exact shape
:func:`gsf.retrieval.kumo.prompts.build_pql_prompt` renders in its "Verified
examples" section. Best-effort: any failure returns ``[]`` so prediction still runs
with no examples.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from gsf.dal.pql_analyses import fetch_pql_analyses_by_ids
from gsf.retrieval.data_access.semantic_search import search_semantic_index
from gsf.semantic.constants import LABEL_PQL_ANALYSIS

logger = logging.getLogger(__name__)

_MAX_EXAMPLES = int(os.environ.get("KUMO_MAX_PQL_EXAMPLES", "5"))


def fetch_pql_examples(
    semantic_retriever: Any,
    question: str,
    k: int = _MAX_EXAMPLES,
) -> list[dict[str, str]]:
    """Return up to *k* verified PQL examples most similar to *question*.

    Each example is ``{question, query, reasoning}`` (reasoning omitted here).
    Returns ``[]`` when there is no retriever, no question, or on any error.
    """
    if semantic_retriever is None or not (question or "").strip():
        return []

    try:
        rows = search_semantic_index(
            semantic_retriever,
            question,
            label_filter=[LABEL_PQL_ANALYSIS],
            per_label_k={LABEL_PQL_ANALYSIS: k},
        )
    except Exception:
        logger.warning("fetch_pql_examples: semantic search failed", exc_info=True)
        return []

    ids = [str(r["id"]) for r in rows if r.get("id") is not None]
    if not ids:
        return []

    analyses = fetch_pql_analyses_by_ids(ids)

    examples: list[dict[str, str]] = []
    for row in rows:  # preserve retrieval (best-first) order
        analysis = analyses.get(str(row.get("id")))
        if not analysis or not analysis.get("pql"):
            continue
        # name is the natural-language question; description is the reasoning.
        name = analysis.get("name") or ""
        reasoning = analysis.get("description") or ""
        example = {"question": name or reasoning, "query": analysis["pql"]}
        if reasoning:
            example["reasoning"] = reasoning
        examples.append(example)
    logger.info("kumo: retrieved %d verified PQL example(s)", len(examples))
    return examples
