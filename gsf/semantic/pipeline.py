"""Semantic compilation entry: flat table-by-table taxonomy pass."""

from __future__ import annotations

import logging

from gsf.neo4j.datasources import (
    fetch_all_tables_without_term,
    fetch_table_context,
)
from gsf.semantic.domain import DomainSummary, load_domain_summary
from gsf.semantic.embed import SemanticEmbedder
from gsf.semantic.visit_enter import process_table

logger = logging.getLogger(__name__)


def _extract_sql_attributes_for_table(
    table: dict,
    ctx: dict,
    schema_name: str | None,
    term_ids: list[str],
    term_names: list[str],
    database_name: str,
) -> list[str]:
    """Run LLM extraction + persistence for one table. Returns created attr ids."""
    from gsf.neo4j.sql_attributes import add_term_link
    from gsf.semantic.sql_attribute_extractor import extract_sql_attributes
    from gsf.server.sql_attributes.service import create_sql_attribute_auto

    term = {
        "name": term_names[0] if term_names else "",
        "description": table.get("description", ""),
    }

    proposals = extract_sql_attributes(table, ctx, schema_name, term, database_name)
    created: list[str] = []

    for proposal in proposals:
        try:
            row = create_sql_attribute_auto(
                name=proposal.name,
                description=proposal.description,
                expression=proposal.expression,
                term_id=term_ids[0],
                database_name=database_name,
            )
            if row is not None:
                attr_id = row["id"]
                created.append(attr_id)
                for extra_term_id in term_ids[1:]:
                    add_term_link(attr_id, extra_term_id)
                logger.info(
                    "  Created SqlAttribute %r (linked to %d terms)",
                    proposal.name,
                    len(term_ids),
                )
        except Exception:
            logger.warning(
                "  Failed to persist SqlAttribute %r",
                proposal.name,
                exc_info=True,
            )

    return created


def compile_semantic_layer(
    database_name: str,
    *,
    domain_summary: DomainSummary | None = None,
    embedder: SemanticEmbedder | None = None,
) -> int:
    """Run full taxonomy compilation over every table in Neo4j."""
    from gsf.neo4j.terms import fetch_terms_and_attributes_for_table

    summary = domain_summary or load_domain_summary(database_name)
    tables = fetch_all_tables_without_term()
    count = 0
    sql_attr_ids: list[str] = []

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
            continue

        try:
            terms, _ = fetch_terms_and_attributes_for_table(table_id)
            term_ids = [t["id"] for t in terms if t.get("id")]
            term_names = [t["name"] for t in terms if t.get("name")]
            schema_name = table.get("schema_name")

            if term_ids and len(ctx.get("columns", [])) >= 2:
                ids = _extract_sql_attributes_for_table(
                    table, ctx, schema_name, term_ids, term_names, database_name
                )
                sql_attr_ids.extend(ids)
        except Exception:
            logger.warning(
                "SqlAttribute extraction failed for table %s",
                table_name,
                exc_info=True,
            )

    logger.info(
        "Compilation complete — %d table(s) processed, %d SqlAttribute(s) created",
        count,
        len(sql_attr_ids),
    )
    return count
