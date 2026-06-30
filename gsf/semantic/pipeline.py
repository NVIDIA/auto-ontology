"""Semantic compilation entry: flat table-by-table taxonomy pass."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from gsf.dal.datasources import fetch_table_context, fetch_all_tables_without_term
from gsf.semantic.domain import DomainSummary, load_domain_summary
from gsf.semantic.embed import SemanticEmbedder
from gsf.semantic.visit_enter import process_table

logger = logging.getLogger(__name__)

_WORKERS = 3


def compile_semantic_layer(
    database_name: str,
    *,
    domain_summary: DomainSummary | None = None,
    embedder: SemanticEmbedder | None = None,
) -> int:
    """Run full taxonomy compilation over every table in Neo4j."""
    summary = domain_summary or load_domain_summary(database_name)
    tables = fetch_all_tables_without_term()

    def _process(table: dict, index: int) -> bool:
        table_id = table["id"]
        table_name = table["name"]
        ctx = fetch_table_context(table_id)

        if not ctx.get("columns"):
            logger.warning("Table %s has no columns — skipping", table_name)
            return False

        logger.info("[%d/%d] Processing table: %s", index, len(tables), table_name)
        try:
            process_table(table, ctx, domain_summary=summary, embedder=embedder)
            return True
        except Exception:
            logger.exception("Unexpected error processing table %s", table_name)
            return False

    count = 0
    if embedder is not None:
        embedder.vdb.create_index()

    with ThreadPoolExecutor(max_workers=_WORKERS) as pool:
        futures = {
            pool.submit(_process, table, i + 1): table for i, table in enumerate(tables)
        }
        for future in as_completed(futures):
            if future.result():
                count += 1

    logger.info("Compilation complete — %d table(s) processed", count)
    return count
