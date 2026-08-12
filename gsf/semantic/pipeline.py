"""Semantic compilation entry: flat table-by-table taxonomy pass."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from gsf.dal.datasources import fetch_all_tables_without_term, fetch_table_context
from gsf.semantic.domain import DomainSummary, load_domain_summary
from gsf.semantic.embed import SemanticEmbedder
from gsf.semantic.models import ProcessTableResult
from gsf.semantic.visit_enter import process_table

logger = logging.getLogger(__name__)

_WORKERS = 3


def compile_semantic_layer(
    database_name: str,
    *,
    domain_summary: DomainSummary | None = None,
    embedder: SemanticEmbedder | None = None,
) -> int:
    """Run full taxonomy compilation over every table in the store.

    Tables are processed in parallel (LLM calls for FK detection and term
    extraction run concurrently). The commit phase (VDB dedup check, the store
    writes, VDB embedding) is serialized via ``_term_commit_lock`` in
    ``visit_enter`` to prevent duplicate Terms.
    """
    summary = domain_summary or load_domain_summary(database_name)
    tables = fetch_all_tables_without_term(database_name)

    def _process(table: dict, index: int) -> ProcessTableResult | None:
        table_name = table["name"]
        ctx = fetch_table_context(table["id"])

        if not ctx.get("columns"):
            logger.warning("Table %s has no columns — skipping", table_name)
            return None

        logger.info("[%d/%d] Processing table: %s", index, len(tables), table_name)
        try:
            return process_table(
                table,
                ctx,
                domain_summary=summary,
                embedder=embedder,
                database_name=database_name,
            )
        except Exception:
            logger.exception("Unexpected error processing table %s", table_name)
            return None

    count = 0

    if embedder is not None:
        embedder.vdb.create_index()

    with ThreadPoolExecutor(max_workers=_WORKERS) as pool:
        futures = {
            pool.submit(_process, table, i + 1): table for i, table in enumerate(tables)
        }
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                count += 1
                logger.debug(
                    "  terms=%s attrs=%s sql_attrs=%s",
                    result.term_names,
                    result.attr_names,
                    result.sql_attr_names,
                )

    logger.info("Compilation complete — %d table(s) processed", count)
    return count
