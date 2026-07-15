"""Per-table taxonomy compilation: FK detection, Term + ColumnAttributes."""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from gsf.connectors import get_connectors
from gsf.dal.attributes import merge_column_attribute
from gsf.dal.datasources import (
    store_column_sample_values,
    store_column_uniqueness,
)
from gsf.dal.terms import fetch_terms_and_attributes_for_table, merge_term
from gsf.semantic.deterministic import column_attribute_specs
from gsf.semantic.domain import DomainSummary
from gsf.semantic.embed import SemanticEmbedder
from gsf.semantic.fk_suggester import suggest_potential_foreign_keys
from gsf.semantic.models import ColumnAttributeSpec, ProcessTableResult
from gsf.semantic.sql_attribute_extractor import extract_sql_attributes
from gsf.semantic.term_extractor import apply_display_names_to_specs, extract_term
from gsf.server.sql_attributes.service import (
    SqlAttributeNameConflict,
    create_sql_attribute,
)

if TYPE_CHECKING:
    from nemo_retriever.tabular_data.sql_database import SQLDatabase

logger = logging.getLogger(__name__)

# Row cap for the per-column profiling sample.
_PROFILING_SAMPLE_LIMIT = 1000
# Most-common values kept per column.
_PROFILING_TOP_N = 5
# String sample values longer than this are not persisted.
_MAX_SAMPLE_VALUE_LEN = 30
# Declared data-type substrings whose sample values are not persisted.
_EXCLUDED_SAMPLE_TYPES = ("date", "time", "timestamp", "datetime", "uuid")

# Tables are processed in parallel (ThreadPoolExecutor in pipeline.py), but
# the commit phase must be serial: VDB search → judge → Neo4j merge → VDB embed.
# Without the lock, two threads could simultaneously propose the same Term,
# both find zero VDB hits (the first hasn't embedded yet), and create duplicates.
_term_commit_lock = threading.Lock()


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
    columns: list[dict],
    schema_name: str | None,
    term_id: str,
    term_name: str,
    database_name: str,
) -> list[str]:
    """Run LLM extraction + persistence for one term's columns. Returns created attr names."""
    term = {
        "name": term_name,
        "description": table.get("description", ""),
    }

    proposals = extract_sql_attributes(table, columns, schema_name, term, database_name)
    created_names: list[str] = []

    for proposal in proposals:
        try:
            row = create_sql_attribute(
                name=proposal.name,
                description=proposal.description,
                expression=proposal.expression,
                term_id=term_id,
                connector=database_name,
                source="table",
            )
            created_names.append(row["name"])
            logger.info("  Created SqlAttribute %r", proposal.name)
        except SqlAttributeNameConflict:
            logger.debug("SqlAttribute %r already exists — skipping", proposal.name)
        except Exception:
            logger.warning(
                "  Failed to persist SqlAttribute %r",
                proposal.name,
                exc_info=True,
            )

    return created_names


def _resolve_connector(database_name: str | None) -> "SQLDatabase | None":
    """Return the loaded connector whose ``database_name`` matches, or None."""
    if not database_name:
        return None
    key = database_name.casefold()
    for connector in get_connectors():
        db = getattr(connector, "database_name", None)
        if db is not None and db.casefold() == key:
            return connector
    return None


def _is_excluded_sample_type(data_type: str | None) -> bool:
    """Whether a column's declared type disqualifies it from sample storage."""
    if not data_type:
        return False
    lowered = data_type.lower()
    return any(token in lowered for token in _EXCLUDED_SAMPLE_TYPES)


def calculate_columns_profiling(
    table: dict[str, Any],
    columns: list[dict[str, Any]],
    connector: "SQLDatabase",
) -> dict[str, dict[str, Any]]:
    """Profile a table's columns from a live sample of up to 1000 rows.

    Runs ``SELECT * ... LIMIT 1000`` and, for every column, computes an
    ``is_unique`` flag (all non-null values distinct) and the 5 most-common
    values.

    Persists to Neo4j Column nodes: ``is_unique`` for every column, and
    ``sample_values`` for every column except those whose declared type is a
    date/time/uuid (individual string values longer than 30 chars are dropped).

    Returns ``{column_name: {"sample_values": [top-5 values], "is_unique": bool}}``
    for *all* columns (values unfiltered — includes dates, uuids and long
    strings).
    """
    schema_name = table.get("schema_name")
    table_name = table["name"]
    qualified = f"{schema_name}.{table_name}" if schema_name else table_name

    try:
        df = connector.execute(
            f"SELECT * FROM {qualified} LIMIT {_PROFILING_SAMPLE_LIMIT}"
        )
    except Exception:
        logger.warning(
            "[%s] column profiling query failed — skipping", table_name, exc_info=True
        )
        return {}

    if df is None or df.empty:
        return {}

    type_by_column = {
        col.get("name"): col.get("data_type") for col in columns if col.get("name")
    }

    profiling: dict[str, dict[str, Any]] = {}
    sample_values: dict[str, list] = {}
    uniqueness: dict[str, bool] = {}

    for column in df.columns:
        col_name = str(column)
        try:
            # Cast to string first: some columns hold unhashable values (e.g.
            # Postgres array columns come back as Python lists, JSON/JSONB as
            # dict/list), and both is_unique and value_counts hash values.
            series = df[column].dropna().map(str)

            is_unique = bool(len(series) > 0 and series.is_unique)
            top5 = list(series.value_counts().head(_PROFILING_TOP_N).index)
        except Exception:
            logger.warning(
                "[%s] profiling failed for column %r — skipping column",
                table_name,
                col_name,
                exc_info=True,
            )
            continue

        uniqueness[col_name] = is_unique
        profiling[col_name] = {"sample_values": top5, "is_unique": is_unique}

        if _is_excluded_sample_type(type_by_column.get(col_name)):
            continue
        filtered = [v for v in top5 if len(v) <= _MAX_SAMPLE_VALUE_LEN]
        if filtered:
            sample_values[col_name] = filtered

    table_id = table["id"]
    store_column_sample_values(table_id, sample_values)
    store_column_uniqueness(table_id, uniqueness)

    return profiling


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

    # Columns profiling — requires a live connector; skipped when unavailable.
    # Persists sample_values + is_unique onto Column nodes, and maps each column
    # to {"sample_values": [...], "is_unique": bool} for FK detection below.
    connector = _resolve_connector(database_name)
    columns_profiling_samples: dict[str, dict[str, Any]] = {}
    if connector is not None:
        try:
            columns_profiling_samples = calculate_columns_profiling(
                table, ctx.get("columns", []), connector
            )
        except Exception:
            logger.warning(
                "[%s] column profiling failed — continuing without it",
                table_name,
                exc_info=True,
            )

    # --- FK detection (LLM + declared); results not written to Neo4j ---
    declared_fks = ctx.get("fks", [])
    fk_suggestions = suggest_potential_foreign_keys(
        table, ctx, columns_profiling_samples
    )
    suggested_fk_names = {s.column_name for s in fk_suggestions.suggestions}
    declared_fk_names = {
        fk["source_column"] for fk in declared_fks if fk.get("source_column")
    }
    all_fk_names = declared_fk_names | suggested_fk_names

    # --- Build attribute specs for non-FK columns ---
    specs = column_attribute_specs(
        ctx.get("columns", []),
        declared_fks,
        suggested_fk_columns=all_fk_names,
        columns_profiling_samples=columns_profiling_samples,
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

    # Serialize: dedup check + Neo4j writes + VDB embed must be atomic
    # so the next thread's VDB search sees this thread's newly embedded terms.
    result_term_names: list[str] = []
    result_attr_names: list[str] = []
    terms: list = []
    attrs_by_term: dict[str, list[dict]] = defaultdict(list)

    with _term_commit_lock:
        if embedder is not None:
            for term, _ in persisted_terms:
                try:
                    candidates = embedder.search_similar_terms(
                        term.name, term.description
                    )
                    if candidates:
                        from gsf.semantic.term_judge import judge_term_overlap

                        merge_into = judge_term_overlap(
                            term.name, term.description, candidates
                        )
                        if merge_into:
                            logger.info(
                                "[%s] Merging proposed Term %r into existing %r",
                                table_name,
                                term.name,
                                merge_into,
                            )
                            term.name = merge_into
                except Exception:
                    logger.warning(
                        "[%s] Term dedup check failed for %r — proceeding as-is",
                        table_name,
                        term.name,
                        exc_info=True,
                    )

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

        # Fetch persisted terms — used for embedding and SQL attribute extraction
        try:
            terms, attrs = fetch_terms_and_attributes_for_table(table_id)
            for attr in attrs:
                if attr.get("term_name"):
                    attrs_by_term[attr["term_name"]].append(attr)
        except Exception:
            logger.warning("[%s] failed to fetch persisted terms", table_name)

        if embedder is not None and terms:
            try:
                for term in terms:
                    embedder.embed_term(term, attrs_by_term.get(term["name"], []))
            except Exception:
                logger.warning("[%s] inline embed failed", table_name)

    # --- LLM: propose SqlAttributes (outside the lock — Term writes are complete) ---
    result_sql_attr_names: list[str] = []
    if database_name is not None and terms:
        try:
            col_by_name = {c["name"]: c for c in ctx.get("columns", [])}
            schema_name = table.get("schema_name")

            for term in terms:
                term_id = term.get("id")
                term_name_str = term.get("name")
                term_col_names = [
                    a["source_column"]
                    for a in attrs_by_term.get(term_name_str, [])
                    if a.get("source_column")
                ]
                filtered_cols = [
                    col_by_name[n] for n in term_col_names if n in col_by_name
                ]
                if len(filtered_cols) < 2:
                    continue
                sql_names = _extract_sql_attributes_for_table(
                    table,
                    filtered_cols,
                    schema_name,
                    term_id,
                    term_name_str,
                    database_name,
                )
                result_sql_attr_names.extend(sql_names)
        except Exception:
            logger.warning(
                "[%s] SqlAttribute extraction failed", table_name, exc_info=True
            )

    return ProcessTableResult(
        term_names=result_term_names,
        attr_names=result_attr_names,
        sql_attr_names=result_sql_attr_names,
    )
