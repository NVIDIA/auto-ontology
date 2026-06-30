"""Semantic compilation entry: flat table-by-table taxonomy pass."""

from __future__ import annotations

import logging

from gsf.neo4j.datasources import (
    fetch_all_tables_with_term,
    fetch_all_tables_without_term,
    fetch_table_context,
)
from gsf.semantic.domain import DomainSummary, load_domain_summary
from gsf.semantic.embed import SemanticEmbedder
from gsf.semantic.visit_enter import process_table

logger = logging.getLogger(__name__)


def compile_semantic_layer(
    database_name: str,
    *,
    domain_summary: DomainSummary | None = None,
    embedder: SemanticEmbedder | None = None,
) -> int:
    """Run full taxonomy compilation over every table in Neo4j."""
    summary = domain_summary or load_domain_summary(database_name)
    tables = fetch_all_tables_without_term()
    count = 0

    for table in tables:
        table_id = table["id"]
        table_name = table["name"]
        ctx = fetch_table_context(table_id)

        if not ctx.get("columns"):
            logger.warning("Table %s has no columns — skipping", table_name)
            continue

        logger.info("[%d/%d] Processing table: %s", count + 1, len(tables), table_name)

        try:
            process_table(table, ctx, domain_summary=summary, embedder=embedder)
            count += 1
        except Exception:
            logger.exception("Unexpected error processing table %s", table_name)

    logger.info("Compilation complete — %d table(s) processed", count)
    return count


def compile_sql_attributes(
    database_name: str,
    *,
    embedder: SemanticEmbedder | None = None,
) -> int:
    """Derive SqlAttributes from table columns via LLM (post-compilation step).

    Iterates tables that already have a Term but no auto-generated
    SqlAttributes. For each table the LLM proposes multi-column business
    metrics; valid proposals are persisted and optionally embedded.

    Returns the total number of SqlAttributes created.
    """
    from gsf.semantic.sql_attribute_extractor import extract_sql_attributes
    from gsf.server.sql_attributes.service import create_sql_attribute_auto

    tables = fetch_all_tables_with_term()
    if not tables:
        logger.info("No unprocessed tables with Terms — nothing to do")
        return 0

    created_ids: list[str] = []
    total = len(tables)

    for idx, table in enumerate(tables, 1):
        table_id = table["id"]
        table_name = table["name"]
        term_id = table["term_id"]
        schema_name = table.get("schema_name")

        ctx = fetch_table_context(table_id)
        if not ctx.get("columns"):
            logger.warning("Table %s has no columns — skipping", table_name)
            continue

        logger.info(
            "[%d/%d] Extracting SqlAttributes for table: %s",
            idx,
            total,
            table_name,
        )

        term = {
            "name": table.get("term_name", ""),
            "description": table.get("description", ""),
        }

        try:
            proposals = extract_sql_attributes(
                table, ctx, schema_name, term, database_name
            )
        except Exception:
            logger.exception(
                "LLM extraction failed for table %s — skipping", table_name
            )
            continue

        for proposal in proposals:
            try:
                row = create_sql_attribute_auto(
                    name=proposal.name,
                    description=proposal.description,
                    expression=proposal.expression,
                    term_id=term_id,
                    database_name=database_name,
                )
                if row is not None:
                    created_ids.append(row["id"])
                    logger.info(
                        "  Created SqlAttribute %r for table %s",
                        proposal.name,
                        table_name,
                    )
            except Exception:
                logger.warning(
                    "  Failed to persist SqlAttribute %r for table %s",
                    proposal.name,
                    table_name,
                    exc_info=True,
                )

    if created_ids and embedder is not None:
        from gsf.server.sql_attributes.service import _embed_sql_attribute
        from gsf.utils import get_embed_params

        embed_params = get_embed_params()
        vdb = embedder.vdb
        for attr_id in created_ids:
            try:
                _embed_sql_attribute(embed_params, vdb, attr_id, database_name)
            except Exception:
                logger.warning(
                    "Embedding failed for SqlAttribute %s", attr_id, exc_info=True
                )

    logger.info(
        "SqlAttribute extraction complete — %d attribute(s) created", len(created_ids)
    )
    return len(created_ids)
