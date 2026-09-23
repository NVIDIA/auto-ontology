"""Semantic compilation entry: flat table-by-table taxonomy pass."""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from types import TracebackType

from auto_ontology.dal.datasources import (
    fetch_all_tables_without_term,
    fetch_table_context,
)
from auto_ontology.infra.feature_flags import is_distinct_value_probing_enabled
from auto_ontology.semantic.domain import DomainSummary, load_domain_summary
from auto_ontology.semantic.embed import SemanticEmbedder
from auto_ontology.semantic.models import ProcessTableResult
from auto_ontology.semantic.visit_enter import process_table, reset_sampling_breaker

logger = logging.getLogger(__name__)

_WORKERS = 3


class _OrderedCommitQueue:
    """Coordinate table commit turns without serializing LLM extraction."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._next_position = 0

    def wait(self, position: int) -> None:
        with self._condition:
            while position != self._next_position:
                self._condition.wait()

    def advance(self, position: int) -> None:
        with self._condition:
            while position != self._next_position:
                self._condition.wait()
            self._next_position += 1
            self._condition.notify_all()


class _OrderedCommitSlot:
    """One table's turn in a compilation-wide canonical commit sequence."""

    def __init__(self, queue: _OrderedCommitQueue, position: int) -> None:
        self._queue = queue
        self._position = position
        self._entered = False
        self._completed = False

    def __enter__(self) -> None:
        self._queue.wait(self._position)
        self._entered = True

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._queue.advance(self._position)
        self._completed = True

    def skip_if_unused(self) -> None:
        """Advance this turn when processing returned before committing."""
        if self._entered or self._completed:
            return
        self._queue.advance(self._position)
        self._completed = True


def _table_commit_key(database_name: str, table: dict) -> tuple[str, ...]:
    """Canonical table order used to decide unqualified Term-name ownership."""
    parts = (
        database_name,
        str(table.get("schema_name") or ""),
        str(table.get("name") or ""),
    )
    return (*[part.casefold() for part in parts], *parts, str(table.get("id") or ""))


def compile_semantic_layer(
    database_name: str,
    *,
    domain_summary: DomainSummary | None = None,
    embedder: SemanticEmbedder | None = None,
) -> int:
    """Run full taxonomy compilation over every table in the store.

    Tables are processed in parallel (LLM calls for FK detection and term
    extraction run concurrently). Term commits run in canonical table order,
    making ownership of an unqualified colliding name deterministic while
    leaving the expensive LLM work parallel.
    """
    # A warehouse that was unreachable during an earlier run in this worker
    # must not stay un-sampled for this one.
    reset_sampling_breaker(database_name)
    summary = domain_summary or load_domain_summary(database_name)
    tables = sorted(
        fetch_all_tables_without_term(database_name),
        key=lambda table: _table_commit_key(database_name, table),
    )
    commit_queue = _OrderedCommitQueue()
    # Read once per run, not once per table: the flag lives in Postgres and
    # tables are processed in parallel, so a per-table read would be hundreds
    # of connections and could also change mid-run.
    is_probe_distinct_values = is_distinct_value_probing_enabled()
    if not is_probe_distinct_values:
        logger.info(
            "Distinct value probing is disabled in settings — sampling rows as "
            "usual, but skipping the per-column DISTINCT probes"
        )

    def _process(
        table: dict, position: int, display_index: int
    ) -> ProcessTableResult | None:
        commit_slot = _OrderedCommitSlot(commit_queue, position)
        table_name = table["name"]
        try:
            ctx = fetch_table_context(table["id"])

            if not ctx.get("columns"):
                logger.warning("Table %s has no columns — skipping", table_name)
                return None

            logger.info(
                "[%d/%d] Processing table: %s",
                display_index,
                len(tables),
                table_name,
            )
            return process_table(
                table,
                ctx,
                domain_summary=summary,
                embedder=embedder,
                database_name=database_name,
                probe_distinct_values=is_probe_distinct_values,
                commit_slot=commit_slot,
            )
        except Exception:
            logger.exception("Unexpected error processing table %s", table_name)
            return None
        finally:
            commit_slot.skip_if_unused()

    count = 0

    if embedder is not None:
        embedder.vdb.create_index()

    with ThreadPoolExecutor(max_workers=_WORKERS) as pool:
        futures = {
            pool.submit(_process, table, position, position + 1): table
            for position, table in enumerate(tables)
        }
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                count += 1
                logger.debug(
                    "  terms=%s attrs=%s",
                    result.term_names,
                    result.attr_names,
                )

    logger.info("Compilation complete — %d table(s) processed", count)
    return count
