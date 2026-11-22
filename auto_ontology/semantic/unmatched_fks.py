# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Give suspected FK columns that found no target a ColumnAttribute.

``process_table`` keeps every suspected FK column — declared, or suggested by
the LLM — out of its table's Term, expecting semantic FK resolution to link it
to the attribute it references instead. A column resolution could not link (a
key that is also a foreign key, a reference to a table that
was never ingested, an inferred match the values or the join refuted) would
otherwise belong to nothing. One that joins other FK columns directly is in a
SEMANTIC_FK_DIRECT group instead (``auto_ontology.semantic.direct_fks``) and
is left alone. The rest, after resolution, are attached like any other
column: to its table's Term, or to a new Term when the table has none — a
junction table, whose columns were all keys, is the usual case.
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager, nullcontext
from typing import Any

from auto_ontology.dal.attributes import (
    find_unlinked_fk_columns,
    merge_column_attribute,
)
from auto_ontology.dal.datasources import fetch_table_by_id, fetch_table_context
from auto_ontology.dal.terms import fetch_terms_and_attributes_for_table
from auto_ontology.semantic.cancellation import is_cancelled
from auto_ontology.semantic.deterministic import (
    column_attribute_specs,
    fill_missing_column_descriptions,
)
from auto_ontology.semantic.domain import DomainSummary
from auto_ontology.semantic.embed import SemanticEmbedder
from auto_ontology.semantic.models import ColumnAttributeSpec, TableTermsResult
from auto_ontology.semantic.pipeline import _OrderedCommitQueue, _OrderedCommitSlot
from auto_ontology.semantic.term_extractor import (
    apply_display_names_to_specs,
    extract_term,
)
from auto_ontology.semantic.visit_enter import (
    commit_terms,
    term_commit_lock,
    terms_with_assignments,
)
from auto_ontology.utils.sample_values import parse_sample_values
from auto_ontology.vdb.records import write_embedded_rows

logger = logging.getLogger(__name__)

# Tables are independent until their writes, which run one at a time in table
# order (``_OrderedCommitQueue``); LLM load is bounded by the compile's
# in-flight limit, not by this.
_WORKERS = 4


def attach_unmatched_fk_columns(
    database_name: str,
    embedder: SemanticEmbedder | None = None,
    *,
    domain_summary: DomainSummary | None = None,
) -> int:
    """Attach every unlinked suspected FK column of *database_name*.

    The term step assigns every column it is given, so once all tables went
    through ``process_table`` a column with no ColumnAttribute is a suspected
    FK that resolution did not link, to a key or directly. Returns the number
    of ColumnAttributes written.
    """
    by_table: dict[str, set[str]] = defaultdict(set)
    for row in find_unlinked_fk_columns(database_name):
        by_table[row["table_id"]].add(row["name"])
    if not by_table:
        logger.info("attach_unmatched_fk_columns: no unmatched suspected FK columns")
        return 0

    queue = _OrderedCommitQueue()
    cancel_logged = threading.Event()

    def _run(position: int, table_id: str, column_names: set[str]) -> int:
        commit_slot = _OrderedCommitSlot(queue, position)
        try:
            if is_cancelled():
                if not cancel_logged.is_set():
                    cancel_logged.set()
                    logger.info("attach_unmatched_fk_columns: cancelled")
                return 0
            return _attach_table(
                table_id,
                column_names,
                embedder,
                domain_summary=domain_summary,
                commit_slot=commit_slot,
            )
        except Exception:
            logger.exception(
                "attach_unmatched_fk_columns: failed for table %s", table_id
            )
            return 0
        finally:
            commit_slot.skip_if_unused()

    with ThreadPoolExecutor(max_workers=_WORKERS) as pool:
        futures = [
            pool.submit(_run, position, table_id, column_names)
            for position, (table_id, column_names) in enumerate(
                sorted(by_table.items())
            )
        ]
        return sum(future.result() for future in futures)


def _attach_table(
    table_id: str,
    column_names: set[str],
    embedder: SemanticEmbedder | None,
    *,
    domain_summary: DomainSummary | None,
    commit_slot: AbstractContextManager[Any] | None = None,
) -> int:
    """Attach *column_names* of one table; only the writes are serialized."""
    table = fetch_table_by_id(table_id)
    if table is None:
        return 0
    ctx = fetch_table_context(table_id)
    columns = [col for col in ctx["columns"] if col.get("name") in column_names]
    profiling = {
        col["name"]: {"sample_values": parse_sample_values(col.get("sample_values"))}
        for col in columns
    }
    fill_missing_column_descriptions(table_id, columns, profiling)
    specs = column_attribute_specs(columns, [])
    if not specs:
        return 0

    terms, _ = fetch_terms_and_attributes_for_table(table_id)
    term_result = (
        None if terms else _extract_new_term(table, ctx, specs, domain_summary)
    )

    with commit_slot or nullcontext(), term_commit_lock:
        terms, _ = fetch_terms_and_attributes_for_table(table_id)
        if terms:
            written = _attach_to_term(table, terms, specs)
        else:
            if term_result is None:
                term_result = _extract_new_term(table, ctx, specs, domain_summary)
            written = _commit_new_term(table, term_result, specs)
    if written and embedder is not None:
        _embed_table_terms(table, embedder)
    return written


def _attach_to_term(
    table: dict[str, Any],
    terms: list[dict[str, Any]],
    specs: list[ColumnAttributeSpec],
) -> int:
    term = min(terms, key=lambda t: t["name"])
    if len(terms) > 1:
        logger.warning(
            "[%s] has %d Terms; attaching unmatched FK columns to %r",
            table["name"],
            len(terms),
            term["name"],
        )
    written = 0
    for spec in specs:
        if merge_column_attribute(
            term_name=term["name"],
            table_id=table["id"],
            source_column=spec.source_column,
            attr_name=spec.display_name or spec.name,
            datatype=spec.datatype,
        ):
            written += 1
    logger.info(
        "[%s] unmatched FK column(s) %s attached to Term %r",
        table["name"],
        sorted(spec.source_column for spec in specs),
        term["name"],
    )
    return written


def _extract_new_term(
    table: dict[str, Any],
    ctx: dict[str, Any],
    specs: list[ColumnAttributeSpec],
    domain_summary: DomainSummary | None,
) -> TableTermsResult:
    """The LLM's Term for a table that has none, with display names applied."""
    term_result = extract_term(table, ctx, specs, domain_summary=domain_summary)
    apply_display_names_to_specs(term_result, specs)
    return term_result


def _commit_new_term(
    table: dict[str, Any],
    term_result: TableTermsResult,
    specs: list[ColumnAttributeSpec],
) -> int:
    spec_by_column = {spec.source_column: spec for spec in specs}
    term_names: list[str] = []
    attr_names: list[str] = []
    commit_terms(
        terms_with_assignments(term_result, spec_by_column),
        spec_by_column,
        table["id"],
        term_names,
        attr_names,
    )
    logger.info(
        "[%s] had no Term; unmatched FK column(s) %s → Terms %s",
        table["name"],
        sorted(spec_by_column),
        term_names,
    )
    return len(attr_names)


def _embed_table_terms(table: dict[str, Any], embedder: SemanticEmbedder) -> None:
    try:
        terms, attrs = fetch_terms_and_attributes_for_table(table["id"])
        attrs_by_term: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for attr in attrs:
            if attr.get("term_name"):
                attrs_by_term[attr["term_name"]].append(attr)
        rows = [
            row
            for term in terms
            for row in embedder.embed_term_rows(
                term, attrs_by_term.get(term["name"], [])
            )
        ]
        # The VDB appends and has no transaction, so replacements are embedded
        # first and only a node that has one loses its old row.
        replaced = {row["metadata"]["id"] for row in rows if row["metadata"].get("id")}
        missing = sorted(
            node["id"] for node in (*terms, *attrs) if node["id"] not in replaced
        )
        if missing:
            logger.warning(
                "[%s] no new embedding for %s; keeping their old vectors",
                table["name"],
                missing,
            )
        for node in (*terms, *attrs):
            if node["id"] in replaced:
                embedder.vdb.delete_by_id(node["id"])
        if rows:
            write_embedded_rows(embedder.vdb, rows)
    except Exception:
        logger.warning("[%s] re-embedding after attaching failed", table["name"])
