# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CustomAnalysis fetch + selection helpers.

Read-only operations on the ``CustomAnalysis`` / ``Sql`` subgraph plus the
small selection / rendering helpers used by the text-to-SQL agents to
filter classified analyses and turn them into markdown.

All direct Neo4j calls live in gsf/dal/custom_analyses.py.
This module only keeps the pure-Python helpers.
"""

from __future__ import annotations

import logging

from gsf.catalog.constants import Labels

from gsf.dal.custom_analyses import fetch_custom_analyses

logger = logging.getLogger(__name__)

__all__ = [
    "fetch_custom_analyses",
    "get_custom_analyses_ids",
    "build_custom_analyses_section",
    "get_relevant_queries",
]


def get_custom_analyses_ids(items):
    """Filter custom analyses by classification flag and return their IDs."""
    if not items:
        return []

    def _get(obj, key, default=None):
        """Safe getter for both Pydantic-style objects and plain dicts."""
        if hasattr(obj, key):
            return getattr(obj, key, default)
        if isinstance(obj, dict):
            return obj.get(key, default)
        return default

    classified_ids_and_labels = []
    for item in items:
        is_relevant = bool(_get(item, "classification", False))
        if not is_relevant:
            continue
        item_id = _get(item, "id")
        item_label = _get(item, "label")
        if item_id and item_label:
            classified_ids_and_labels.append({"id": item_id, "label": item_label})

    return classified_ids_and_labels


def build_custom_analyses_section(items, candidates):
    """Build a markdown section listing custom analyses that were used."""
    if not items:
        return ""

    def _get(obj, key, default=None):
        return getattr(
            obj, key, obj.get(key, default) if isinstance(obj, dict) else default
        )

    by_id = {_get(c, "id"): c for c in candidates if _get(c, "id")}

    matched_lines = []
    for item in items:
        cid = _get(item, "id")
        candidate = by_id.get(cid)
        if not candidate:
            continue

        name = _get(candidate, "name", "<unknown name>")
        relevant = _get(item, "classification", False)
        if relevant:
            matched_lines.append(f"- [[[{name}/{cid}]]]")

    if not matched_lines:
        return ""

    return "\n\n**Semantic items used**:\n" + "\n".join(matched_lines)


def get_relevant_queries(candidates):
    """Collect SQL snippets from custom-analysis candidates (deduped, in order)."""
    snippet_queries = []
    for candidate in candidates:
        if candidate.get("label", "") == Labels.CUSTOM_ANALYSIS:
            analysis_sql = candidate.get("sql", [])
            if not analysis_sql:
                continue
            s_query = analysis_sql[0].get("sql_code", "")
            if s_query and s_query not in snippet_queries:
                snippet_queries.append(s_query)
    return snippet_queries
