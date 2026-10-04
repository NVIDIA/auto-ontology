# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Create SqlAttributes for pure-FK bridge (junction) tables.

After semantic FK resolution, discovers tables whose columns are entirely
foreign keys, generates a structural join SQL via LLM, and persists a
SqlAttribute on the better-matching of the Terms that REPRESENT the FK
target tables (chosen by reranking those Terms against the SqlAttribute
description). No new Term is created for the bridge table itself.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from nemo_retriever.operators.rerank import rerank_hits
from pydantic import BaseModel, ConfigDict

from auto_ontology.dal.datasources import (
    fetch_bridge_table_candidates,
    mark_table_as_junction,
)
from auto_ontology.dal.sql_attributes import (
    SqlAttributeExpressionConflict,
    SqlAttributeNameConflict,
    SqlAttributeSqlError,
)
from auto_ontology.dal.terms import get_term_record_for_table
from auto_ontology.semantic.constants import SQL_ATTR_SOURCE_BRIDGE
from auto_ontology.server.sql_attributes.service import create_sql_attribute
from auto_ontology.utils.llm_invoke import (
    get_llm_client,
    invoke_with_structured_output,
)
from auto_ontology.utils.rerank import get_rerank_kwargs

logger = logging.getLogger(__name__)


class _BridgeSqlAttributeProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    expression: str


def _format_fk_block(fk_pairs: list[dict[str, Any]]) -> str:
    lines = []
    for pair in fk_pairs:
        src = pair.get("source_column", "")
        tgt_schema = pair.get("target_schema", "")
        tgt_table = pair.get("target_table", "")
        tgt_col = pair.get("target_column", "")
        lines.append(
            f"  - {src} -> {tgt_schema}.{tgt_table}.{tgt_col}"
            if tgt_schema
            else f"  - {src} -> {tgt_table}.{tgt_col}"
        )
    return "\n".join(lines)


def _target_terms_for_bridge(candidate: dict[str, Any]) -> list[dict[str, str]]:
    """Terms that REPRESENT the distinct FK target tables of a bridge."""
    terms: list[dict[str, str]] = []
    seen_term_ids: set[str] = set()
    seen_table_ids: set[str] = set()
    for pair in candidate.get("fk_pairs") or []:
        table_id = pair.get("target_table_id")
        if not table_id or table_id in seen_table_ids:
            continue
        seen_table_ids.add(table_id)
        record = get_term_record_for_table(table_id)
        if not record or not record.get("id"):
            logger.warning(
                "Bridge FK target table %r (%s) has no REPRESENTS Term.",
                pair.get("target_table"),
                table_id,
            )
            continue
        term_id = record["id"]
        if term_id in seen_term_ids:
            continue
        seen_term_ids.add(term_id)
        terms.append(record)
    return terms


def _term_hit_text(term: dict[str, str]) -> str:
    name = (term.get("name") or "").strip()
    description = (term.get("description") or "").strip()
    if name and description:
        return f"{name}. {description}"
    return name or description


def _pick_owner_term_id(
    terms: list[dict[str, str]],
    sql_attr_description: str,
) -> str | None:
    """Choose the Term that best matches the SqlAttribute description.

    With a single candidate (self-referential bridge, or only one FK target
    has a Term), that Term wins without calling the reranker. With two or
    more, rerank by ``sql_attr_description`` and take the highest score.
    """
    if not terms:
        return None
    if len(terms) == 1:
        return terms[0]["id"]

    query = (sql_attr_description or "").strip()
    if not query:
        logger.warning(
            "SqlAttribute description empty — falling back to first target Term %r.",
            terms[0].get("name"),
        )
        return terms[0]["id"]

    hits = [
        {"id": t["id"], "text": _term_hit_text(t), "name": t.get("name")} for t in terms
    ]
    hits = [h for h in hits if (h.get("text") or "").strip()]
    if not hits:
        return terms[0]["id"]

    try:
        ranked = rerank_hits(query, hits, top_n=1, **get_rerank_kwargs())
    except Exception:
        logger.exception(
            "Rerank failed for bridge owner selection — falling back to first Term."
        )
        return terms[0]["id"]

    if not ranked:
        return terms[0]["id"]

    winner = ranked[0]
    logger.info(
        "Bridge owner Term chosen by rerank: %r (score=%s) among %d candidate(s).",
        winner.get("name") or winner.get("id"),
        winner.get("_rerank_score"),
        len(hits),
    )
    return winner.get("id") or terms[0]["id"]


def _generate_bridge_sql_attribute(
    candidate: dict[str, Any],
) -> _BridgeSqlAttributeProposal | None:
    schema_name = candidate.get("schema_name") or ""
    table_name = candidate.get("table_name") or ""
    fk_pairs = candidate.get("fk_pairs") or []
    qualified_bridge = f"{schema_name}.{table_name}" if schema_name else table_name

    related_tables = sorted(
        {
            (
                f"{pair['target_schema']}.{pair['target_table']}"
                if pair.get("target_schema")
                else pair["target_table"]
            )
            for pair in fk_pairs
            if pair.get("target_table")
        }
    )
    self_referential = len(related_tables) == 1
    join_hint = (
        "This is a self-referential bridge: both FK columns point at the same "
        "entity table. Alias that table twice in the JOIN (e.g. product AS buyer "
        "and product AS also_bought) so each FK role is clear."
        if self_referential
        else "Join the bridge table to each related entity table using the FK pairs."
    )

    llm = get_llm_client(temperature=0.0)
    messages = [
        SystemMessage(
            content=(
                "You are an expert data engineer defining reusable join patterns "
                "for many-to-many junction (bridge) tables. Bridge tables connect "
                "entities through foreign keys, including self-referential cases "
                "where multiple columns point at the same target table."
            )
        ),
        HumanMessage(
            content=(
                f"Bridge table: {qualified_bridge}\n"
                f"Related tables: {', '.join(related_tables) or '(unknown)'}\n"
                f"Foreign key columns (use these exact join keys):\n"
                f"{_format_fk_block(fk_pairs)}\n\n"
                f"{join_hint}\n\n"
                "Produce one SqlAttribute that captures how to join through this "
                "bridge table to reach all related entities.\n\n"
                "Rules:\n"
                "  • expression must be a valid SELECT … FROM … JOIN … statement\n"
                "  • schema-qualify every table name\n"
                "  • use ONLY the FK pairs listed above — do not invent columns "
                "or filters\n"
                "  • no WHERE clause with literal business filters; structural "
                "joins only\n"
                "  • name — user-friendly Title Case label (e.g. 'Order Products', "
                "'Also Buy')\n"
                "  • description — what relationship the junction represents\n"
            )
        ),
    ]

    result = invoke_with_structured_output(llm, messages, _BridgeSqlAttributeProposal)
    return result


def build_bridge_tables_sql_attributes(database_name: str) -> int:
    """Discover bridge tables and persist join-pattern SqlAttributes.

    Returns the number of newly created SqlAttribute nodes.
    """
    candidates = fetch_bridge_table_candidates(database_name)
    if not candidates:
        logger.info("No eligible bridge tables found for database %r.", database_name)
        return 0

    logger.info(
        "Found %d bridge table candidate(s) in database %r.",
        len(candidates),
        database_name,
    )

    # Structural qualification is sufficient evidence for the catalog flag.
    # Do this before optional Term lookup and LLM SQL generation so those
    # downstream failures cannot lose a valid junction classification.
    for candidate in candidates:
        mark_table_as_junction(candidate["table_id"])

    created = 0
    for candidate in candidates:
        table_name = candidate.get("table_name") or candidate.get("table_id")

        target_terms = _target_terms_for_bridge(candidate)
        if not target_terms:
            logger.warning(
                "Bridge table %r: no REPRESENTS Terms on FK targets — skipping.",
                table_name,
            )
            continue

        proposal = _generate_bridge_sql_attribute(candidate)
        if proposal is None:
            logger.warning(
                "Bridge table %r: LLM returned no proposal — skipping.", table_name
            )
            continue

        term_id = _pick_owner_term_id(target_terms, proposal.description)
        if not term_id:
            logger.warning(
                "Bridge table %r: could not pick an owner Term — skipping.", table_name
            )
            continue

        try:
            create_sql_attribute(
                name=proposal.name,
                description=proposal.description,
                expression=proposal.expression,
                term_id=term_id,
                connector=database_name,
                source=SQL_ATTR_SOURCE_BRIDGE,
            )
        except SqlAttributeNameConflict:
            logger.debug(
                "Bridge table %r: SqlAttribute %r already exists — skipping.",
                table_name,
                proposal.name,
            )
            continue
        except SqlAttributeExpressionConflict:
            logger.debug(
                "Bridge table %r: equivalent SQL already exists for term — skipping.",
                table_name,
            )
            continue
        except SqlAttributeSqlError as exc:
            logger.warning(
                "Bridge table %r: SQL validation failed for %r — skipping: %s",
                table_name,
                proposal.name,
                exc,
            )
            continue
        except Exception:
            logger.exception(
                "Bridge table %r: unexpected error creating SqlAttribute %r",
                table_name,
                proposal.name,
            )
            continue

        logger.info(
            "Bridge table %r: created SqlAttribute %r on Term %r.",
            table_name,
            proposal.name,
            term_id,
        )
        created += 1

    logger.info(
        "Bridge table SqlAttribute pass complete — %d new node(s) written.", created
    )
    return created
