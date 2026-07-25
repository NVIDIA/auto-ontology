# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Create SqlAttributes for pure-FK bridge (junction) tables.

After semantic FK resolution, discovers tables whose columns are entirely
foreign keys, generates a structural join SQL via LLM, and persists a
SqlAttribute owned by the bridge table's Term.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict

from gsf.dal.datasources import fetch_bridge_table_candidates
from gsf.dal.sql_attributes import (
    SqlAttributeExpressionConflict,
    SqlAttributeNameConflict,
    SqlAttributeSqlError,
)
from gsf.dal.terms import get_term_id_for_table, merge_term
from gsf.semantic.constants import SQL_ATTR_SOURCE_BRIDGE
from gsf.server.sql_attributes.service import create_sql_attribute
from gsf.utils.llm_invoke import (
    get_non_reasoning_llm_client,
    invoke_with_structured_output,
)

logger = logging.getLogger(__name__)


class _BridgeSqlAttributeProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    expression: str


def _table_name_to_term_name(table_name: str) -> str:
    return " ".join(part.capitalize() for part in table_name.replace("_", " ").split())


def _junction_description(
    bridge_table: str,
    fk_pairs: list[dict[str, Any]],
) -> str:
    related = sorted(
        {pair["target_table"] for pair in fk_pairs if pair.get("target_table")}
    )
    if not related:
        return f"Junction table represented by {bridge_table}."
    if len(related) == 1:
        return (
            f"Self-referential junction table linking rows of {related[0]} "
            f"via {bridge_table}."
        )
    joined = ", ".join(related[:-1]) + f" and {related[-1]}"
    return f"Junction table linking {joined}."


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


def _ensure_bridge_term_id(candidate: dict[str, Any]) -> str | None:
    table_id = candidate["table_id"]
    existing = get_term_id_for_table(table_id)
    if existing:
        return existing

    table_name = candidate.get("table_name") or ""
    fk_pairs = candidate.get("fk_pairs") or []
    term_name = _table_name_to_term_name(table_name)
    description = candidate.get("description") or _junction_description(
        table_name, fk_pairs
    )
    return merge_term(term_name, description, table_id)


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

    llm = get_non_reasoning_llm_client(temperature=0.0)
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

    created = 0
    for candidate in candidates:
        table_name = candidate.get("table_name") or candidate.get("table_id")
        term_id = _ensure_bridge_term_id(candidate)
        if not term_id:
            logger.warning(
                "Bridge table %r: could not resolve Term — skipping.", table_name
            )
            continue

        proposal = _generate_bridge_sql_attribute(candidate)
        if proposal is None:
            logger.warning(
                "Bridge table %r: LLM returned no proposal — skipping.", table_name
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
            "Bridge table %r: created SqlAttribute %r.", table_name, proposal.name
        )
        created += 1

    logger.info(
        "Bridge table SqlAttribute pass complete — %d new node(s) written.", created
    )
    return created
