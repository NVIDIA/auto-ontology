"""Unified entry point for semantic layer compilation."""

from __future__ import annotations

import logging

from gsf.semantic.domain import DomainSummary, load_domain_summary
from gsf.semantic.embed import build_semantic_embedder
from gsf.semantic.semantic_fk import resolve_semantic_fks

from gsf.semantic.pipeline import compile_semantic_layer

logger = logging.getLogger(__name__)


def run_semantic_compilation(
    database_name: str,
    *,
    domain_summary: DomainSummary | None = None,
) -> int:
    """Compile semantic taxonomy, embed all nodes into the VDB, then resolve FK edges.

    Returns the number of tables processed.
    """
    summary = domain_summary or load_domain_summary(database_name)

    embedder = build_semantic_embedder(database_name, reset=False)

    logger.info("=" * 60)
    logger.info(
        "Semantic compilation — full Neo4j graph (database=%r)",
        database_name,
    )
    logger.info("=" * 60)

    count = compile_semantic_layer(
        database_name,
        domain_summary=summary,
        embedder=embedder,
    )

    logger.info("=" * 60)
    logger.info("Semantic compilation finished — %d table visits", count)
    logger.info("=" * 60)

    logger.info("=" * 60)
    logger.info("Resolving semantic FK edges…")
    logger.info("=" * 60)
    fk_count = resolve_semantic_fks(database_name)
    logger.info("Semantic FK edges created: %d", fk_count)

    return count
