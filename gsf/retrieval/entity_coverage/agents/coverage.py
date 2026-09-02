# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic coverage grade: distance filter, the store enrich, 1/n entity coverage."""

from __future__ import annotations

import logging
from typing import Any, Dict

from gsf.dal.attributes import fetch_attr_column_contexts
from gsf.dal.custom_analyses import fetch_custom_analyses_with_sql
from gsf.dal.sql_attributes import fetch_sql_attributes_with_sql
from gsf.retrieval.entity_coverage.state import DEFAULT_MAX_DISTANCE
from gsf.retrieval.text_to_sql.base import BaseAgent
from gsf.retrieval.text_to_sql.state import AgentState
from gsf.semantic.constants import LABEL_COLUMN_ATTRIBUTE, LABEL_SQL_ATTRIBUTE
from gsf.catalog.constants import Labels

logger = logging.getLogger(__name__)


def _within_distance(hit: dict[str, Any], max_distance: float) -> bool:
    try:
        return float(hit.get("score") or float("inf")) <= max_distance
    except (TypeError, ValueError):
        return False


def _filter_by_distance(
    hits: list[dict[str, Any]],
    max_distance: float,
) -> list[dict[str, Any]]:
    return [h for h in hits if _within_distance(h, max_distance)]


def _entity_keys_for_hit(hit: dict[str, Any]) -> set[str]:
    """Entities this ColumnAttribute hit covers (supports multi-entity dedupe)."""
    keys: set[str] = set()
    entities = hit.get("query_entities")
    if isinstance(entities, list):
        keys.update(str(e) for e in entities if e)
    qe = hit.get("query_entity")
    if qe:
        keys.add(str(qe))
    return keys


def _covered_entities(
    entities: list[str],
    col_attr_hits: list[dict[str, Any]],
) -> set[str]:
    covered: set[str] = set()
    for hit in col_attr_hits:
        covered.update(_entity_keys_for_hit(hit))
    return {e for e in entities if e in covered}


def _compute_coverage(
    entities: list[str],
    col_attr_hits: list[dict[str, Any]],
) -> float:
    if not entities:
        return 0.0
    return len(_covered_entities(entities, col_attr_hits)) / len(entities)


def _uncovered_entities(
    entities: list[str],
    col_attr_hits: list[dict[str, Any]],
) -> list[str]:
    """Entities with no covering ColumnAttribute hit (order preserved)."""
    covered = _covered_entities(entities, col_attr_hits)
    return [e for e in entities if e not in covered]


def _enrich_column_attributes(
    hits: list[dict[str, Any]],
    *,
    database_name: str | None,
) -> list[dict[str, Any]]:
    attr_ids = [str(h["id"]) for h in hits if h.get("id")]
    contexts = fetch_attr_column_contexts(
        attr_ids,
        database_name=database_name,
    )
    enriched: list[dict[str, Any]] = []
    for hit in hits:
        aid = str(hit.get("id") or "")
        ctx = contexts.get(aid) or {}
        enriched.append(
            {
                "id": aid,
                "label": hit.get("label") or LABEL_COLUMN_ATTRIBUTE,
                "score": hit.get("score"),
                "name": ctx.get("attr_name") or hit.get("name") or "",
                "term_name": ctx.get("term_name") or "",
            }
        )
    return enriched


def _enrich_sql_attributes(
    hits: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    attr_ids = [str(h["id"]) for h in hits if h.get("id")]
    details = {d["id"]: d for d in fetch_sql_attributes_with_sql(attr_ids)}
    enriched: list[dict[str, Any]] = []
    for hit in hits:
        aid = str(hit.get("id") or "")
        detail = details.get(aid) or {}
        enriched.append(
            {
                "id": aid,
                "label": hit.get("label") or LABEL_SQL_ATTRIBUTE,
                "score": hit.get("score"),
                "name": detail.get("name") or hit.get("name") or "",
                "term_name": detail.get("term_name") or "",
            }
        )
    return enriched


def _enrich_custom_analyses(
    hits: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    analysis_ids = [str(h["id"]) for h in hits if h.get("id")]
    details = {d["id"]: d for d in fetch_custom_analyses_with_sql(analysis_ids)}
    enriched: list[dict[str, Any]] = []
    for hit in hits:
        aid = str(hit.get("id") or "")
        detail = details.get(aid) or {}
        enriched.append(
            {
                "id": aid,
                "label": hit.get("label") or Labels.CUSTOM_ANALYSIS,
                "score": hit.get("score"),
                "name": detail.get("name") or hit.get("name") or "",
                "term_name": "",
            }
        )
    return enriched


def _rank_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        candidates,
        key=lambda h: float(h.get("score") or float("inf")),
    )


def _to_response_candidate(hit: dict[str, Any]) -> dict[str, Any]:
    """Slim ranked-list item: label, attribute, term, id."""
    return {
        "label": hit.get("label") or "",
        "attribute": hit.get("name") or "",
        "term": hit.get("term_name") or "",
        "id": hit.get("id") or "",
    }


class CoverageGradeAgent(BaseAgent):
    """Filter by distance, enrich via the store, compute 0–1 entity coverage grade."""

    def __init__(self) -> None:
        super().__init__("coverage_grade")

    def validate_input(self, state: AgentState) -> bool:
        return True

    def execute(self, state: AgentState) -> Dict[str, Any]:
        path_state = state.get("path_state", {})
        entities: list[str] = list(path_state.get("entities") or [])
        max_distance = float(path_state.get("max_distance", DEFAULT_MAX_DISTANCE))

        col_hits = _filter_by_distance(
            list(path_state.get("retrieved_column_attributes") or []),
            max_distance,
        )
        sql_hits = _filter_by_distance(
            list(path_state.get("retrieved_sql_attributes") or []),
            max_distance,
        )
        custom_hits = _filter_by_distance(
            list(path_state.get("retrieved_custom_analyses") or []),
            max_distance,
        )

        coverage = _compute_coverage(entities, col_hits)

        ranked = _rank_candidates(
            _enrich_column_attributes(
                col_hits,
                database_name=path_state.get("target_db"),
            )
            + _enrich_sql_attributes(sql_hits)
            + _enrich_custom_analyses(custom_hits)
        )
        candidates = [_to_response_candidate(hit) for hit in ranked]

        final_response: dict[str, Any] = {
            "coverage": round(coverage, 4),
            "candidates": candidates,
        }
        if path_state.get("return_uncovered_entities"):
            final_response["uncovered_entities"] = _uncovered_entities(
                entities, col_hits
            )
        path_state["final_response"] = final_response
        path_state["coverage"] = coverage

        self.logger.info(
            "Coverage grade=%.4f (%d/%d entities covered), %d candidates after "
            "distance filter (max_distance=%.3f)",
            coverage,
            int(round(coverage * len(entities))) if entities else 0,
            len(entities),
            len(candidates),
            max_distance,
        )
        return {"path_state": path_state}
