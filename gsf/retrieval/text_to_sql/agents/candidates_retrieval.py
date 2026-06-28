# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Candidate Retrieval Agent

Searches both VDBs per extracted entity, applies an LLM intent filter on each
entity's raw hits, and stores typed results in path_state.

Responsibilities:
- Search the semantic VDB (ontology_retriever) for ColumnAttribute candidates.
- Search the semantic VDB (semantic_retriever) for CustomAnalysis candidates.
- Filter each entity's hits by intent using the LLM (full question, not entity).
- Deduplicate across entities and store results in path_state.
"""

import logging
from typing import Any, Dict

from langchain_core.messages import SystemMessage

from nemo_retriever.tabular_data.ingestion.model.reserved_words import Labels
from gsf.retrieval.data_access.semantic_search import search_semantic_index
from gsf.utils.llm_invoke import invoke_with_structured_output
from gsf.retrieval.text_to_sql.base import BaseAgent
from gsf.retrieval.text_to_sql.models import (
    CandidateFilterModel,
    ColumnAttributeSpec,
    CustomAnalysisFilterModel,
)
from gsf.retrieval.text_to_sql.state import (
    AgentState,
    get_question_for_processing,
)

logger = logging.getLogger(__name__)

# Label used in the semantic VDB for column-attribute nodes.
_COLUMN_ATTRIBUTE_LABEL = "ColumnAttribute"


# ---------------------------------------------------------------------------
# Search helpers
# ---------------------------------------------------------------------------


def _search_column_attributes(ontology_retriever, entity: str, k: int) -> list[dict]:
    """Return up to *k* ColumnAttribute hits from the semantic VDB for *entity*."""
    try:
        return list(
            search_semantic_index(
                ontology_retriever,
                entity,
                label_filter=[_COLUMN_ATTRIBUTE_LABEL],
                per_label_k=k,
            )
        )
    except Exception:
        logger.warning(
            "ColumnAttribute search failed for entity %r", entity, exc_info=True
        )
        return []


def _search_custom_analyses(retriever, entity: str, k: int) -> list[dict]:
    """Return up to *k* CustomAnalysis hits from the semantic VDB for *entity*."""
    try:
        return list(
            search_semantic_index(
                retriever,
                entity,
                label_filter=[Labels.CUSTOM_ANALYSIS],
                per_label_k=k,
            )
        )
    except Exception:
        logger.warning(
            "CustomAnalysis search failed for entity %r", entity, exc_info=True
        )
        return []


# ---------------------------------------------------------------------------
# LLM intent filter
# ---------------------------------------------------------------------------

_FILTER_PROMPT_TEMPLATE = """\
You are selecting the single best candidate that matches a user's question.

User question: {question}
Entity being searched: {entity}

Candidates:
{candidates_block}

Return the ID of the single best matching candidate.
Return null if none genuinely match the intent of the question.
"""


def _llm_filter(llm, question: str, entity: str, candidates: list[dict]) -> list[str]:
    """Use the LLM to pick the single best candidate by relevance to *question*.

    Each candidate must have an ``id`` field in its metadata.
    Returns a list with the single best ID, or all IDs on LLM failure.
    """
    if not candidates:
        return []

    all_ids = [str(c.get("id") or "") for c in candidates if c.get("id")]

    candidates_block = "\n".join(
        f"- id: {c.get('id')} | {c.get('text', '')}" for c in candidates if c.get("id")
    )

    messages = [
        SystemMessage(
            content=_FILTER_PROMPT_TEMPLATE.format(
                question=question,
                entity=entity,
                candidates_block=candidates_block,
            )
        )
    ]

    result = invoke_with_structured_output(llm, messages, CandidateFilterModel)
    if result is None or not result.best_id:
        return all_ids

    return [result.best_id]


_CUSTOM_ANALYSIS_FILTER_PROMPT = """\
You are filtering candidate custom analyses for relevance to a user question.
Keep only analyses that could meaningfully contribute to answering the question.
Remove any that share no common domain, idea, or intent with the question.

User question: {question}

Candidates:
{candidates_block}

Return the list of IDs to KEEP. If none are relevant, return an empty list.
"""


def _llm_filter_custom_analyses(
    llm, question: str, candidates: list[dict]
) -> list[dict]:
    """Use the LLM to keep only custom analysis candidates relevant to *question*.

    Returns the filtered list; falls back to the original list on LLM failure.
    """
    if not candidates:
        return []

    candidates_block = "\n".join(
        f"- id: {c.get('id')} | {c.get('text', '')}" for c in candidates if c.get("id")
    )

    messages = [
        SystemMessage(
            content=_CUSTOM_ANALYSIS_FILTER_PROMPT.format(
                question=question,
                candidates_block=candidates_block,
            )
        )
    ]

    result = invoke_with_structured_output(llm, messages, CustomAnalysisFilterModel)
    if result is None:
        return candidates

    kept_ids = set(result.kept_ids)
    filtered = [c for c in candidates if str(c.get("id") or "") in kept_ids]
    logger.debug(
        "Custom analysis filter: %d → %d (kept ids: %s)",
        len(candidates),
        len(filtered),
        kept_ids,
    )
    return filtered


# ---------------------------------------------------------------------------
# ColumnAttributeSpec builder
# ---------------------------------------------------------------------------


def _build_column_attribute_spec(hit: dict) -> ColumnAttributeSpec | None:
    """Build a :class:`ColumnAttributeSpec` from a raw VDB hit dict.

    Expects the hit (or its ``metadata`` sub-dict) to contain ``name`` and
    ``source_column``. Returns ``None`` when required fields are absent.
    """
    # The hit may carry fields directly or nested under ``metadata``.
    meta: dict = hit.get("metadata") or {}
    if isinstance(meta, str):
        import json as _json

        try:
            meta = _json.loads(meta)
        except Exception:
            meta = {}

    def _get(key: str) -> Any:
        return hit.get(key) or meta.get(key)

    name = _get("name") or (hit.get("text") or "").strip() or None
    source_column = _get("source_column")

    if not name or not source_column:
        return None

    return ColumnAttributeSpec(
        name=name,
        source_column=source_column,
        display_name=_get("display_name") or "",
        datatype=_get("datatype") or "",
        description=_get("description"),
    )


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class CandidateRetrievalAgent(BaseAgent):
    """Retrieve ColumnAttribute and CustomAnalysis candidates.

    - ColumnAttributes: searched per entity from the semantic VDB, then
      LLM-filtered by the full question intent.
    - CustomAnalysis: searched once with the full question from the data VDB.

    Deduplicate across entities and store:
    - ``path_state["retrieved_column_attributes"]``: ``list[ColumnAttributeSpec]``
    - ``path_state["retrieved_custom_analyses"]``:   ``list[dict]`` (same shape as before)
    """

    def __init__(self):
        super().__init__("candidate_retrieval")

    def validate_input(self, state: AgentState) -> bool:
        question = get_question_for_processing(state)
        if not question:
            self.logger.warning("No question available for retrieval")
            return False
        return True

    def execute(self, state: AgentState) -> Dict[str, Any]:
        path_state = state.get("path_state", {})
        question = get_question_for_processing(state)
        entities: list[str] = path_state.get("entities") or []
        llm = state["llm"]
        semantic_retriever = state.get("semantic_retriever")

        all_col_attr_hits: list[dict] = []

        # CustomAnalysis: search once with the full question, not per entity.
        # CustomAnalysis rows live in the semantic-layer collection, so query
        # the semantic retriever rather than the data retriever.
        all_custom_hits = (
            _search_custom_analyses(semantic_retriever, question, 3)
            if semantic_retriever is not None
            else []
        )
        all_custom_hits = _llm_filter_custom_analyses(llm, question, all_custom_hits)

        for entity in entities:
            entity = (entity or "").strip()
            if not entity:
                continue

            # 1. Search semantic VDB for ColumnAttributes per entity
            col_attr_raw = _search_column_attributes(semantic_retriever, entity, 2)

            # 2. Collect all ColumnAttribute hits without LLM filtering
            all_col_attr_hits.extend(col_attr_raw)

        # Deduplicate ColumnAttribute hits by id (keep first occurrence, i.e. lowest score).
        seen_col_ids: set[str] = set()
        deduped_col_attr: list[dict] = []
        for hit in all_col_attr_hits:
            key = str(hit.get("id") or "")
            if key and key not in seen_col_ids:
                seen_col_ids.add(key)
                deduped_col_attr.append(hit)

        # Deduplicate CustomAnalysis hits by id (keep lowest score).
        best_custom: dict[str, dict] = {}
        for hit in all_custom_hits:
            hid = hit.get("id")
            if hid is None:
                continue
            key = str(hid)
            prev = best_custom.get(key)
            if prev is None or float(hit.get("score") or float("inf")) < float(
                prev.get("score") or float("inf")
            ):
                best_custom[key] = hit
        deduped_custom = sorted(
            best_custom.values(),
            key=lambda h: float(h.get("score") or float("inf")),
        )

        path_state["retrieved_column_attributes"] = deduped_col_attr
        path_state["retrieved_custom_analyses"] = deduped_custom

        self.logger.info(
            "Retrieved %d ColumnAttributes and %d CustomAnalysis candidates "
            "(%d entities queried)",
            len(deduped_col_attr),
            len(deduped_custom),
            len(entities),
        )

        return {"path_state": path_state}
