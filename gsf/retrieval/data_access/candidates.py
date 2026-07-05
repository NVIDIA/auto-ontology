# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Candidate retrieval: vector hits + Neo4j graph enrichment.

* :func:`gsf.dal.candidates.expand_info` pulls graph properties for the
  (label, id) pairs returned by :func:`semantic_search.search_semantic_index`.
* :func:`_get_candidates_information` glues the two together for a single
  question string.
* :func:`extract_candidates` runs the per-entity / per-query-with-values
  fan-out, deduplicates by (label, id) keeping the lowest vector distance,
  and splits into ``(custom_analysis, column, sql_attribute)`` streams.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from nemo_retriever.tabular_data.ingestion.model.reserved_words import Labels

from gsf.semantic.constants import LABEL_SQL_ATTRIBUTE

from gsf.dal.candidates import expand_info
from gsf.retrieval.data_access.relevant_tables import (
    _normalize_table_to_relevant_shape,
)
from gsf.retrieval.data_access.semantic_search import (
    MAX_CALCULATION_CANDIDATES,
    PER_LABEL_LIMIT,
    PER_LABEL_LIMITS,
    search_semantic_index,
)

if TYPE_CHECKING:
    from nemo_retriever.graph.retriever import Retriever

logger = logging.getLogger(__name__)


def _get_candidates_information(
    retriever: "Retriever",
    entity: str,
    list_of_semantic: list | None = None,
    database_name: str | None = None,
    per_label_k: "int | dict[str, int]" = PER_LABEL_LIMIT,
):
    """Vector search, then merge graph properties from :func:`expand_info`.

    Runs one query per label with a server-side ``where`` predicate
    (label + *database_name*) keeping at most *per_label_k* per label,
    then enriches each hit with Neo4j graph properties.
    """
    results: list[dict] = list(
        search_semantic_index(
            retriever,
            entity,
            label_filter=list_of_semantic,
            database_name=database_name,
            per_label_k=per_label_k,
        )
    )

    ids_and_labels = [{"label": x["label"], "id": x["id"]} for x in results]
    props_by_id = expand_info(ids_and_labels)
    for c in results:
        cid = c.get("id")
        if cid is None:
            continue
        extra = props_by_id.get(cid) or props_by_id.get(str(cid))
        if isinstance(extra, dict):
            c.update(extra)
            rel_tabs = c.get("relevant_tables")
            if isinstance(rel_tabs, list):
                c["relevant_tables"] = [
                    _normalize_table_to_relevant_shape(t)
                    for t in rel_tabs
                    if isinstance(t, dict)
                ]

    results.sort(
        key=lambda item: float(
            item.get("score") if item.get("score") is not None else float("inf")
        )
    )
    return results


def _dedupe_best_score_sort_cap(combined: list[dict]) -> list[dict]:
    """Deduplicate by (label, id), keep lowest ``score`` (L2 distance), sort ascending, cap."""
    best_by_key: dict[tuple[str | None, str], dict] = {}
    for c in combined:
        cid = c.get("id")
        if cid is None:
            continue
        key = (c.get("label"), str(cid))
        dist = c.get("score")
        score = float(dist) if dist is not None else float("inf")
        prev = best_by_key.get(key)
        prev_d = prev.get("score") if prev is not None else None
        prev_score = float(prev_d) if prev_d is not None else float("inf")
        if prev is None or score < prev_score:
            best_by_key[key] = c

    unique = list(best_by_key.values())
    unique.sort(
        key=lambda x: (
            float(x.get("score")) if x.get("score") is not None else float("inf")
        )
    )
    return unique[:MAX_CALCULATION_CANDIDATES]


def extract_candidates(
    retriever: "Retriever",
    entities: list[str],
    query_with_values: str = "",
    database_name: str | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """One semantic search per pull string (``query_with_values`` and each entity name).

    Each search fetches custom-analysis, column, and sql-attribute candidates
    in a single vector-store call, then splits by label in Python.

    Merge streams, dedupe by (label, id) keeping the lowest vector distance
    (``score``), sort ascending by distance, cap at ``MAX_CALCULATION_CANDIDATES``
    per stream.

    Returns:
        ``(custom_analysis_candidates, column_candidates, sql_attribute_candidates)``
    """
    target_labels = [Labels.CUSTOM_ANALYSIS, Labels.COLUMN, LABEL_SQL_ATTRIBUTE]

    pulls: list[str] = []
    if qwv := (query_with_values or "").strip():
        pulls.append(qwv)
    for ent in entities or []:
        if t := (ent or "").strip():
            pulls.append(t)

    combined_custom: list[dict] = []
    combined_columns: list[dict] = []
    combined_sql_attrs: list[dict] = []

    for text in pulls:
        hits = (
            _get_candidates_information(
                retriever,
                text,
                list_of_semantic=target_labels,
                database_name=database_name,
                per_label_k=PER_LABEL_LIMITS,
            )
            or []
        )
        for hit in hits:
            lab = str(hit.get("label") or "")
            if lab == Labels.CUSTOM_ANALYSIS:
                combined_custom.append(hit)
            elif lab == Labels.COLUMN:
                combined_columns.append(hit)
            elif lab == LABEL_SQL_ATTRIBUTE:
                combined_sql_attrs.append(hit)

    out_custom = _dedupe_best_score_sort_cap(combined_custom)
    out_columns = _dedupe_best_score_sort_cap(combined_columns)
    out_sql_attrs = _dedupe_best_score_sort_cap(combined_sql_attrs)

    logger.info(
        "extract_candidates: %d custom_analysis, %d column, %d sql_attribute "
        "(max %d each), %d pulls",
        len(out_custom),
        len(out_columns),
        len(out_sql_attrs),
        MAX_CALCULATION_CANDIDATES,
        len(pulls),
    )

    return out_custom, out_columns, out_sql_attrs
