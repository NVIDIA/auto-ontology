"""Semantic compilation entry: flat table-by-table taxonomy pass."""

from __future__ import annotations

import logging

from gsf.dal.datasources import fetch_table_context, fetch_all_tables_without_term
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
