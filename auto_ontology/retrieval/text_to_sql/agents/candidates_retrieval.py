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
- Search the semantic VDB for one Term hit using path_state["subject"].
- Filter each entity's hits by intent using the LLM (full question, not entity).
- Deduplicate across entities and store results in path_state.
"""

import logging

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict

from langchain_core.messages import SystemMessage

from auto_ontology.catalog.constants import Labels

from auto_ontology.semantic.constants import (
    LABEL_COLUMN_ATTRIBUTE,
    LABEL_SQL_ATTRIBUTE,
    LABEL_TERM,
    SQL_ATTR_SOURCE_BRIDGE,
)

from auto_ontology.dal.custom_analyses import custom_analysis_exists
from auto_ontology.retrieval.data_access.semantic_search import search_semantic_index
from auto_ontology.utils.llm_invoke import invoke_with_structured_output
from auto_ontology.retrieval.text_to_sql.base import BaseAgent
from auto_ontology.retrieval.text_to_sql.models import (
    CandidateFilterModel,
    ColumnAttributeSpec,
    CombinedCandidateFilterModel,
    CustomAnalysisFilterModel,
)
from auto_ontology.retrieval.text_to_sql.state import (
    AgentState,
    get_question_for_processing,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Search / dedup helpers
# ---------------------------------------------------------------------------


def _search_by_label(
    retriever: object,
    entity: str,
    label: str,
    k: int,
    database_name: str | None = None,
) -> list[dict]:
    """Return up to *k* VDB hits for *label*."""
    try:
        return list(
            search_semantic_index(
                retriever,
                entity,
                label_filter=[label],
                per_label_k=k,
                database_name=database_name,
            )
        )
    except Exception:
        logger.warning("%s search failed for entity %r", label, entity, exc_info=True)
        return []


def _dedupe_best_score(hits: list[dict]) -> list[dict]:
    """Deduplicate by id, keeping the hit with the lowest score.

    When hits carry ``query_entity``, accumulate all entities that retrieved
    the same id into ``query_entities`` so per-entity coverage is preserved.
    """
    best: dict[str, dict] = {}
    entities_by_id: dict[str, set[str]] = {}
    for hit in hits:
        hid = hit.get("id")
        if hid is None:
            continue
        key = str(hid)
        qe = hit.get("query_entity")
        if qe:
            entities_by_id.setdefault(key, set()).add(str(qe))
        prev = best.get(key)
        if prev is None or float(hit.get("score") or float("inf")) < float(
            prev.get("score") or float("inf")
        ):
            best[key] = hit
    result: list[dict] = []
    for key, hit in best.items():
        out = dict(hit)
        ents = entities_by_id.get(key)
        if ents:
            out["query_entities"] = sorted(ents)
        result.append(out)
    return sorted(
        result,
        key=lambda h: float(h.get("score") or float("inf")),
    )


def _database_name(hit: dict) -> str | None:
    """Return a normalized database name from a semantic hit."""
    database_name = str(hit.get("database_name") or "").strip()
    return database_name or None


def _score(hit: dict) -> float:
    """Return a sortable semantic distance, treating missing scores as worst."""
    score = hit.get("score")
    return float(score) if score is not None else float("inf")


def _select_candidate_database(
    column_hits: list[dict],
    custom_hits: list[dict],
    sql_attribute_hits: list[dict],
) -> tuple[str | None, dict[str, dict[str, Any]]]:
    """Select the database that covers the most searched entities.

    Databases are ranked by distinct ColumnAttribute ``query_entity`` coverage,
    then total hits across all candidate types, aggregate semantic distance,
    and finally database name for deterministic ties.
    """
    stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"entities": set(), "hit_count": 0, "score_total": 0.0}
    )

    for hit in column_hits:
        database_name = _database_name(hit)
        if database_name is None:
            continue
        entity = str(hit.get("query_entity") or "").strip()
        if entity:
            stats[database_name]["entities"].add(entity)
        stats[database_name]["hit_count"] += 1
        stats[database_name]["score_total"] += _score(hit)

    for hit in [*custom_hits, *sql_attribute_hits]:
        database_name = _database_name(hit)
        if database_name is None:
            continue
        stats[database_name]["hit_count"] += 1
        stats[database_name]["score_total"] += _score(hit)

    if not stats:
        return None, {}

    selected_database = min(
        stats,
        key=lambda database_name: (
            -len(stats[database_name]["entities"]),
            -stats[database_name]["hit_count"],
            stats[database_name]["score_total"],
            database_name,
        ),
    )
    return selected_database, dict(stats)


def _filter_hits_to_database(hits: list[dict], database_name: str) -> list[dict]:
    """Keep only semantic hits belonging to *database_name*."""
    return [hit for hit in hits if _database_name(hit) == database_name]


def _covered_entities(column_hits: list[dict]) -> set[str]:
    """Return entities represented by the retained ColumnAttribute hits."""
    return {
        str(hit.get("query_entity")).strip()
        for hit in column_hits
        if str(hit.get("query_entity") or "").strip()
    }


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


_CANDIDATE_FILTER_PROMPT = """\
You are filtering candidate {candidate_type} for relevance to a user question.
Keep only candidates that could meaningfully contribute to answering the question.
Remove any that share no common domain, idea, or intent with the question.

User question: {question}

Candidates:
{candidates_block}

Return the list of IDs to KEEP. If none are relevant, return an empty list.
"""

_COMBINED_FILTER_PROMPT = """\
You are filtering two sets of candidates for relevance to a user question.
Keep only candidates that could meaningfully contribute to answering the question.
Remove any that share no common domain, idea, or intent with the question.

User question: {question}

Custom analyses:
{custom_block}

SQL attributes:
{sql_attr_block}

Return the IDs to KEEP for each set separately. Use empty lists if none are relevant.
"""


def _llm_filter_candidates(
    llm, question: str, candidates: list[dict], candidate_type: str
) -> list[dict]:
    """Keep only candidates relevant to *question* via LLM.

    Falls back to the original list on LLM failure.
    """
    if not candidates:
        return []

    candidates_block = "\n".join(
        f"- id: {c.get('id')} | {c.get('text', '')}" for c in candidates if c.get("id")
    )

    messages = [
        SystemMessage(
            content=_CANDIDATE_FILTER_PROMPT.format(
                candidate_type=candidate_type,
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
        "%s filter: %d → %d (kept ids: %s)",
        candidate_type,
        len(candidates),
        len(filtered),
        kept_ids,
    )
    return filtered


def _llm_filter_both(
    llm,
    question: str,
    custom_hits: list[dict],
    sql_attr_hits: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Filter custom analyses and SQL attributes in a single LLM call.

    Falls back to the original lists on LLM failure.
    """
    if not custom_hits and not sql_attr_hits:
        return [], []

    def _fmt(candidates: list[dict]) -> str:
        lines = [
            f"- id: {c.get('id')} | {c.get('text', '')}"
            for c in candidates
            if c.get("id")
        ]
        return "\n".join(lines) if lines else "(none)"

    messages = [
        SystemMessage(
            content=_COMBINED_FILTER_PROMPT.format(
                question=question,
                custom_block=_fmt(custom_hits),
                sql_attr_block=_fmt(sql_attr_hits),
            )
        )
    ]

    result = invoke_with_structured_output(llm, messages, CombinedCandidateFilterModel)
    if result is None:
        return custom_hits, sql_attr_hits

    kept_custom = set(result.custom_analysis_ids)
    kept_sql = set(result.sql_attribute_ids)
    filtered_custom = [c for c in custom_hits if str(c.get("id") or "") in kept_custom]
    filtered_sql = [c for c in sql_attr_hits if str(c.get("id") or "") in kept_sql]

    logger.debug(
        "combined filter: custom %d→%d, sql_attr %d→%d",
        len(custom_hits),
        len(filtered_custom),
        len(sql_attr_hits),
        len(filtered_sql),
    )
    return filtered_custom, filtered_sql


def _all_bridge_sourced(sql_attr_hits: list[dict]) -> bool:
    """Whether every hit in *sql_attr_hits* is a bridge-table structural join.

    Bridge-table SqlAttributes (``source="bridgeTable"``) are LLM-generated,
    schema-only join patterns with no business filter to judge for intent —
    they are always structurally relevant when retrieved. Skipping the LLM
    filter for a pure-bridge batch avoids its latency; any other source (or
    an empty/mixed batch) still goes through the filter as usual. ``source``
    rides along on the hit's own VDB metadata (see ``embed_docs_into_vdb``),
    so this needs no extra lookup.
    """
    if not sql_attr_hits:
        return False
    return all(h.get("source") == SQL_ATTR_SOURCE_BRIDGE for h in sql_attr_hits)


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
    """Retrieve ColumnAttribute, CustomAnalysis, SqlAttribute, and subject Term candidates.

    - ColumnAttributes: searched per entity from the semantic VDB.
    - CustomAnalysis: searched once with the full question from the semantic VDB.
    - SqlAttribute: searched once with the full question from the semantic VDB.
    - Subject Term: searched once with ``path_state["subject"]`` (top-1 hit)
      when that key is a non-empty string; skipped when absent/None/blank.

    Deduplicate across entities and store:
    - ``path_state["retrieved_column_attributes"]``: ``list[dict]``
      (each ColumnAttribute hit may include ``query_entity`` /
      ``query_entities`` naming the extraction string(s) that retrieved it)
    - ``path_state["retrieved_custom_analyses"]``:   ``list[dict]``
    - ``path_state["retrieved_sql_attributes"]``:    ``list[dict]``
    - ``path_state["retrieved_subject_term"]``:      ``dict | None``
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
        raw_entities = path_state.get("entities")
        entities: list[str] = [
            e.strip()
            for e in (raw_entities if isinstance(raw_entities, list) else [])
            if isinstance(e, str) and e.strip()
        ]
        raw_subject = path_state.get("subject")
        subject = (
            raw_subject.strip()
            if isinstance(raw_subject, str) and raw_subject.strip()
            else ""
        )
        llm = state["llm"]
        semantic_retriever = state.get("semantic_retriever")
        target_db = path_state.get("target_db")
        retrieval_database = target_db

        all_col_attr_hits: list[dict] = []
        all_custom_hits: list[dict] = []
        all_sql_attr_hits: list[dict] = []
        subject_term_hits: list[dict] = []

        if semantic_retriever is not None:
            clean_entities = entities

            has_custom = custom_analysis_exists(target_db)
            if not has_custom:
                self.logger.info(
                    "No CustomAnalysis nodes found for database %r — skipping VDB search",
                    target_db,
                )

            search_tasks: list[tuple[str, Any]] = [
                *(
                    [
                        (
                            "custom",
                            (
                                semantic_retriever,
                                question,
                                Labels.CUSTOM_ANALYSIS,
                                3,
                                target_db,
                            ),
                        )
                    ]
                    if has_custom
                    else []
                ),
                (
                    "sql_attr",
                    (
                        semantic_retriever,
                        question,
                        LABEL_SQL_ATTRIBUTE,
                        3,
                        target_db,
                    ),
                ),
                *[
                    (
                        f"col_attr:{entity}",
                        (
                            semantic_retriever,
                            entity,
                            LABEL_COLUMN_ATTRIBUTE,
                            2,
                            target_db,
                        ),
                    )
                    for entity in clean_entities
                ],
            ]
            if subject:
                search_tasks.append(
                    (
                        "subject_term",
                        (
                            semantic_retriever,
                            subject,
                            LABEL_TERM,
                            1,
                            target_db,
                        ),
                    )
                )

            with ThreadPoolExecutor(max_workers=len(search_tasks) or 1) as pool:
                futures = {
                    pool.submit(_search_by_label, *args): key
                    for key, args in search_tasks
                }
                for future in as_completed(futures):
                    key = futures[future]
                    result = future.result()
                    if key == "custom":
                        all_custom_hits = result
                    elif key == "sql_attr":
                        all_sql_attr_hits = result
                    elif key == "subject_term":
                        subject_term_hits = result
                    else:
                        # key is "col_attr:{entity}" — tag each hit for coverage.
                        entity = key.split(":", 1)[1]
                        for hit in result:
                            tagged = dict(hit)
                            tagged["query_entity"] = entity
                            all_col_attr_hits.append(tagged)

            if target_db is None:
                selected_database, database_stats = _select_candidate_database(
                    all_col_attr_hits,
                    all_custom_hits,
                    all_sql_attr_hits,
                )
                if selected_database is None:
                    self.logger.warning(
                        "No database-scoped semantic hits found; dropping %d "
                        "hits without database_name",
                        len(all_col_attr_hits)
                        + len(all_custom_hits)
                        + len(all_sql_attr_hits),
                    )
                    all_col_attr_hits = []
                    all_custom_hits = []
                    all_sql_attr_hits = []
                else:
                    retrieval_database = selected_database
                    initial_hit_count = (
                        len(all_col_attr_hits)
                        + len(all_custom_hits)
                        + len(all_sql_attr_hits)
                    )
                    selected_stats = database_stats[selected_database]
                    self.logger.info(
                        "Selected candidate database %r: %d entities covered, "
                        "%d hits, aggregate score %.4f",
                        selected_database,
                        len(selected_stats["entities"]),
                        selected_stats["hit_count"],
                        selected_stats["score_total"],
                    )

                    all_col_attr_hits = _filter_hits_to_database(
                        all_col_attr_hits, selected_database
                    )
                    all_custom_hits = _filter_hits_to_database(
                        all_custom_hits, selected_database
                    )
                    all_sql_attr_hits = _filter_hits_to_database(
                        all_sql_attr_hits, selected_database
                    )
                    retained_hit_count = (
                        len(all_col_attr_hits)
                        + len(all_custom_hits)
                        + len(all_sql_attr_hits)
                    )
                    self.logger.info(
                        "Dropped %d semantic hits outside candidate database %r",
                        initial_hit_count - retained_hit_count,
                        selected_database,
                    )

                    covered_entities = _covered_entities(all_col_attr_hits)
                    uncovered_entities = [
                        entity
                        for entity in clean_entities
                        if entity not in covered_entities
                    ]
                    if uncovered_entities:
                        self.logger.info(
                            "Backfilling %d uncovered entities in database %r: %s",
                            len(uncovered_entities),
                            selected_database,
                            uncovered_entities,
                        )
                        with ThreadPoolExecutor(
                            max_workers=len(uncovered_entities)
                        ) as pool:
                            futures = {
                                pool.submit(
                                    _search_by_label,
                                    semantic_retriever,
                                    entity,
                                    LABEL_COLUMN_ATTRIBUTE,
                                    2,
                                    selected_database,
                                ): entity
                                for entity in uncovered_entities
                            }
                            for future in as_completed(futures):
                                entity = futures[future]
                                for hit in _filter_hits_to_database(
                                    future.result(), selected_database
                                ):
                                    tagged = dict(hit)
                                    tagged["query_entity"] = entity
                                    all_col_attr_hits.append(tagged)

        if _all_bridge_sourced(all_sql_attr_hits):
            # All hits are structural bridge-table joins — nothing to judge
            # for intent, so only run the (cheaper) single-list filter on
            # custom analyses and skip the combined LLM call entirely.
            all_custom_hits = _llm_filter_candidates(
                llm, question, all_custom_hits, "custom analyses"
            )
        else:
            all_custom_hits, all_sql_attr_hits = _llm_filter_both(
                llm, question, all_custom_hits, all_sql_attr_hits
            )

        deduped_col_attr = _dedupe_best_score(all_col_attr_hits)
        deduped_custom = _dedupe_best_score(all_custom_hits)
        deduped_sql_attr = _dedupe_best_score(all_sql_attr_hits)
        subject_term = subject_term_hits[0] if subject_term_hits else None

        path_state["retrieved_column_attributes"] = deduped_col_attr
        path_state["retrieved_custom_analyses"] = deduped_custom
        path_state["retrieved_sql_attributes"] = deduped_sql_attr
        if subject:
            path_state["retrieved_subject_term"] = subject_term
        else:
            path_state.pop("retrieved_subject_term", None)
        if retrieval_database:
            path_state["retrieval_database"] = retrieval_database
        else:
            path_state.pop("retrieval_database", None)

        self.logger.info(
            "Retrieved %d ColumnAttributes, %d CustomAnalysis, "
            "%d SqlAttribute candidates, and subject Term %s "
            "(%d entities queried, subject=%r)",
            len(deduped_col_attr),
            len(deduped_custom),
            len(deduped_sql_attr),
            (
                f"id={subject_term.get('id')!r} score={subject_term.get('score')}"
                if subject_term
                else "None"
            ),
            len(entities),
            subject,
        )

        return {"path_state": path_state}
