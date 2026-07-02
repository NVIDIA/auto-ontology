"""Per-table taxonomy compilation: FK detection, Term + ColumnAttributes."""

from __future__ import annotations

import logging
from typing import Any

from gsf.dal.attributes import merge_column_attribute
from gsf.dal.terms import fetch_terms_and_attributes_for_table, merge_term
from gsf.semantic.deterministic import column_attribute_specs
from gsf.semantic.domain import DomainSummary
from gsf.semantic.embed import SemanticEmbedder
from gsf.semantic.fk_suggester import suggest_potential_foreign_keys
from gsf.semantic.models import ColumnAttributeSpec, ProcessTableResult
from gsf.semantic.sql_attribute_extractor import extract_sql_attributes
from gsf.semantic.term_extractor import apply_display_names_to_specs, extract_term
from gsf.server.sql_attributes.service import create_sql_attribute_auto

logger = logging.getLogger(__name__)


def _terms_with_assignments(
    term_result: Any,
    spec_by_column: dict[str, ColumnAttributeSpec],
) -> list[tuple[Any, list[Any]]]:
    """Terms that have at least one resolvable column attribute."""
    persisted = []
    for term in term_result.terms:
        assignments = [a for a in term.attributes if a.source_column in spec_by_column]
        if assignments:
            persisted.append((term, assignments))
    return persisted


def _extract_sql_attributes_for_table(
    table: dict,
    ctx: dict,
    schema_name: str | None,
    term_ids: list[str],
    term_names: list[str],
    database_name: str,
) -> list[str]:
    """Run LLM extraction + persistence for one table. Returns created attr names."""
    term = {
        "name": term_names[0] if term_names else "",
        "description": table.get("description", ""),
    }

    proposals = extract_sql_attributes(table, ctx, schema_name, term, database_name)
    created_names: list[str] = []

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
                created_names.append(row["name"])
                logger.info("  Created SqlAttribute %r", proposal.name)
        except Exception:
            logger.warning(
                "  Failed to persist SqlAttribute %r",
                proposal.name,
                exc_info=True,
            )

    return created_names


def process_table(
    table: dict[str, Any],
    ctx: dict[str, Any],
    *,
    domain_summary: DomainSummary | None,
    embedder: SemanticEmbedder | None = None,
    database_name: str | None = None,
) -> ProcessTableResult:
    """Build taxonomy nodes for one table: Term and ColumnAttributes."""
    table_id = table["id"]
    table_name = table["name"]

    # --- FK detection (LLM + declared); results not written to Neo4j ---
    declared_fks = ctx.get("fks", [])
    fk_suggestions = suggest_potential_foreign_keys(table, ctx)
    suggested_fk_names = {s.column_name for s in fk_suggestions.suggestions}
    declared_fk_names = {
        fk["source_column"] for fk in declared_fks if fk.get("source_column")
    }
    all_fk_names = declared_fk_names | suggested_fk_names

    # --- Build attribute specs for non-FK columns ---
    specs = column_attribute_specs(
        ctx.get("columns", []),
        declared_fks,
        suggested_fk_columns=suggested_fk_names,
    )
    if not specs:
        logger.warning("[%s] no non-FK columns — skipping Term creation", table_name)
        return ProcessTableResult()

    # --- LLM: propose Term(s) and display names ---
    term_result = extract_term(table, ctx, specs, domain_summary=domain_summary)
    apply_display_names_to_specs(term_result, specs)
    spec_by_column = {spec.source_column: spec for spec in specs}
    persisted_terms = _terms_with_assignments(term_result, spec_by_column)

    if not persisted_terms:
        logger.warning(
            "[%s] LLM assigned no columns to any Term (%d candidates)",
            table_name,
            len(specs),
        )
        return ProcessTableResult()

    result_term_names: list[str] = []
    result_attr_names: list[str] = []
    for term, assignments in persisted_terms:
        merge_term(term.name, term.description, table_id, synonyms=term.synonyms)
        result_term_names.append(term.name)
        for assignment in assignments:
            spec = spec_by_column[assignment.source_column]
            merge_column_attribute(
                term_name=term.name,
                table_id=table_id,
                source_column=spec.source_column,
                attr_name=spec.display_name,
                datatype=spec.datatype,
                description=spec.description,
            )
            result_attr_names.append(spec.display_name)

    logger.info(
        "[%s] → Terms %s (%d attrs, %d suspected FKs)",
        table_name,
        result_term_names,
        len(result_attr_names),
        len(all_fk_names),
    )

    # --- Fetch persisted terms once — reused for embedding and SQL attr extraction ---
    try:
        terms, attrs = fetch_terms_and_attributes_for_table(table_id)
    except Exception:
        logger.warning("[%s] failed to fetch persisted terms", table_name)
        return ProcessTableResult(
            term_names=result_term_names, attr_names=result_attr_names
        )

    if embedder is not None:
        try:
            from collections import defaultdict

            attrs_by_term: dict[str, list[dict]] = defaultdict(list)
            for attr in attrs:
                if attr.get("term_name"):
                    attrs_by_term[attr["term_name"]].append(attr)
            for term in terms:
                embedder.embed_term(term, attrs_by_term.get(term["name"], []))
        except Exception:
            logger.warning("[%s] inline embed failed", table_name)

    result_sql_attr_names: list[str] = []
    if database_name is not None:
        try:
            term_ids = [t["id"] for t in terms if t.get("id")]
            term_names = [t["name"] for t in terms if t.get("name")]
            schema_name = table.get("schema_name")
            if term_ids and len(ctx.get("columns", [])) >= 2:
                result_sql_attr_names = _extract_sql_attributes_for_table(
                    table, ctx, schema_name, term_ids, term_names, database_name
                )
        except Exception:
            logger.warning(
                "[%s] SqlAttribute extraction failed", table_name, exc_info=True
            )

    return ProcessTableResult(
        term_names=result_term_names,
        attr_names=result_attr_names,
        sql_attr_names=result_sql_attr_names,
    )
