# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Order and annotate the tables SQL generation sees, entity by entity.

The SQL-generating model strongly prefers tables listed early in AVAILABLE
TABLES, and picks an early near-duplicate over a later correct table. The
collection order of candidate preparation says nothing about relevance, so
the right table for the question's last entity could sit at the bottom.

Ranking therefore happens per entity and is then interleaved: every entity's
best table comes before any entity's second best. Each table also records
which entities it matches and whether it is that entity's best match, so the
prompt can tell the model which of several look-alike tables carries a concept.
"""

import logging

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from auto_ontology.dal.attributes import fetch_attr_column_contexts
from auto_ontology.retrieval.data_access.semantic_search import search_semantic_index
from auto_ontology.semantic.constants import LABEL_COLUMN_ATTRIBUTE

logger = logging.getLogger(__name__)

#: ColumnAttribute hits pulled per entity. Wide enough that a table several
#: places down an entity's ranking still gets scored for it.
ENTITY_SCORE_K = 30

#: Concurrent per-entity scorers. Each one checks out store connections in
#: ``fetch_attr_column_contexts``; kept below the DAL pool size.
_SCORE_WORKERS = 4

#: ``{entity: {table_id: {"score": distance, "column": column name}}}``.
EntityTableScores = dict[str, dict[str, dict[str, Any]]]


def unique_entities(entities: list[Any]) -> list[str]:
    """Non-empty entity strings, stripped, first occurrence kept."""
    seen: set[str] = set()
    result: list[str] = []
    for entity in entities or []:
        if not isinstance(entity, str):
            continue
        text = entity.strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _score_one_entity(
    retriever: Any, entity: str, target_db: str | None
) -> dict[str, dict[str, Any]]:
    rows = search_semantic_index(
        retriever,
        entity,
        label_filter=[LABEL_COLUMN_ATTRIBUTE],
        per_label_k=ENTITY_SCORE_K,
        database_name=target_db,
    )
    attr_ids = [str(r["id"]) for r in rows if r.get("id")]
    if not attr_ids:
        return {}
    contexts = fetch_attr_column_contexts(attr_ids, database_name=target_db)
    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        score = row.get("score")
        ctx = contexts.get(str(row.get("id") or ""))
        table_id = str(ctx.get("table_id") or "") if ctx else ""
        if score is None or not table_id:
            continue
        score = float(score)
        if table_id not in best or score < best[table_id]["score"]:
            best[table_id] = {"score": score, "column": ctx.get("col_name") or ""}
    return best


def score_tables_by_entity(
    retriever: Any, entities: list[str], target_db: str | None
) -> EntityTableScores:
    """Best (lowest-distance) ColumnAttribute hit per table, separately per entity.

    An entity whose search fails is left out rather than failing the rest; the
    callers treat a missing entity as "no signal" and keep their prior order.
    """
    if retriever is None or not entities:
        return {}
    scores: EntityTableScores = {}
    with ThreadPoolExecutor(max_workers=min(len(entities), _SCORE_WORKERS)) as pool:
        futures = {
            entity: pool.submit(_score_one_entity, retriever, entity, target_db)
            for entity in entities
        }
        for entity, future in futures.items():
            try:
                scores[entity] = future.result()
            except Exception:
                logger.warning(
                    "Entity table scoring failed for %r", entity, exc_info=True
                )
    return scores


def best_score_across_entities(scores: EntityTableScores) -> dict[str, float]:
    """``{table_id: lowest distance over all entities}``."""
    best: dict[str, float] = {}
    for per_table in scores.values():
        for table_id, hit in per_table.items():
            if table_id not in best or hit["score"] < best[table_id]:
                best[table_id] = hit["score"]
    return best


def rank_tables_by_entity(
    tables: list[dict],
    scores: EntityTableScores,
    entities: list[str],
    lead_table_id: str | None = None,
) -> list[dict]:
    """Reorder *tables* round-robin across entities and annotate entity matches.

    Order: the lead (anchor) table, then rank 1 of every entity in entity
    order, then rank 2 of every entity, and so on. Tables no entity matched
    follow in their original order. Ties within an entity keep the original
    order, so the result is deterministic.

    Every matched table gets ``entity_matches``: one entry per entity it
    matches, with the matching column, whether it is that entity's best table,
    and otherwise the id of the table that is. The input dicts are not mutated.
    """
    if not tables or not scores:
        return list(tables)

    position = {str(t.get("id") or ""): i for i, t in enumerate(tables) if t.get("id")}
    ranked_per_entity: list[tuple[str, list[str]]] = []
    for entity in entities:
        per_table = scores.get(entity) or {}
        in_scope = [tid for tid in per_table if tid in position]
        in_scope.sort(key=lambda tid: (per_table[tid]["score"], position[tid]))
        if in_scope:
            ranked_per_entity.append((entity, in_scope))

    if not ranked_per_entity:
        return list(tables)

    order: list[str] = []
    placed: set[str] = set()

    def place(table_id: str) -> None:
        if table_id not in placed:
            placed.add(table_id)
            order.append(table_id)

    if lead_table_id and lead_table_id in position:
        place(lead_table_id)
    depth = max(len(ranked) for _, ranked in ranked_per_entity)
    for rank in range(depth):
        for _, ranked in ranked_per_entity:
            if rank < len(ranked):
                place(ranked[rank])

    matches: dict[str, list[dict[str, Any]]] = {}
    for entity, ranked in ranked_per_entity:
        best_id = ranked[0]
        for table_id in ranked:
            entry: dict[str, Any] = {
                "entity": entity,
                "column": scores[entity][table_id]["column"],
                "best": table_id == best_id,
            }
            if table_id != best_id:
                entry["best_table_id"] = best_id
            matches.setdefault(table_id, []).append(entry)

    by_id = {str(t.get("id") or ""): t for t in tables if t.get("id")}
    result: list[dict] = []
    for table_id in order:
        table = by_id[table_id]
        result.append(
            {**table, "entity_matches": matches[table_id]}
            if table_id in matches
            else table
        )
    result.extend(
        t for t in tables if not t.get("id") or str(t.get("id")) not in placed
    )
    return result
