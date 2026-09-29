# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
from auto_ontology.semantic.cancellation import is_cancelled
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
        self._skipped: set[int] = set()

    def wait(self, position: int) -> None:
        with self._condition:
            while position != self._next_position:
                self._condition.wait()

    def advance(self, position: int) -> None:
        with self._condition:
            while position != self._next_position:
                self._condition.wait()
            self._advance_locked()

    def skip(self, position: int) -> None:
        """Give up a turn **without** waiting for it to come round.

        Giving up a turn used to mean waiting for it first, which holds a pool
        worker for as long as the tables ahead take — and the workers are few
        (``_WORKERS``). Cancelling a backlog makes every queued table skip at
        once, so that cost lands exactly when the point is to stop quickly.

        Recording the skip instead releases the worker immediately, and also
        allows a position to be given up out of order, which waiting cannot
        express at all: a skip of a position whose predecessors have not had
        their turns would simply block forever.

        The skip is consumed when its turn arrives, cascading through any run
        of consecutive skips.
        """
        with self._condition:
            self._skipped.add(position)
            self._drain_skipped_locked()
            self._condition.notify_all()

    def _advance_locked(self) -> None:
        self._next_position += 1
        self._drain_skipped_locked()
        self._condition.notify_all()

    def _drain_skipped_locked(self) -> None:
        while self._next_position in self._skipped:
            self._skipped.discard(self._next_position)
            self._next_position += 1


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
        """Give up this turn when processing returned before committing.

        Non-blocking (see :meth:`_OrderedCommitQueue.skip`): waiting for the
        turn would hold a pool worker, and cancellation makes every queued
        table skip at once.
        """
        if self._entered or self._completed:
            return
        self._queue.skip(self._position)
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

    # Set the first time a table is dropped, so the "we are cancelling" line is
    # logged once rather than once per table.
    cancel_logged = threading.Event()

    def _process(
        table: dict, position: int, display_index: int
    ) -> ProcessTableResult | None:
        commit_slot = _OrderedCommitSlot(commit_queue, position)
        table_name = table["name"]
        # Checked here rather than before submitting: every table is queued up
        # front, so this is what turns the backlog into no-ops and lets the pool
        # drain in about one table's time instead of one database's.
        if is_cancelled():
            if not cancel_logged.is_set():
                cancel_logged.set()
                # Say this out loud: the tables already running cannot be
                # interrupted, and one mid-LLM-call can take minutes. Without a
                # line here the pass goes silent and looks wedged.
                logger.info(
                    "Compilation cancelled — dropping the queued tables and "
                    "waiting for the up-to-%d already running to finish",
                    _WORKERS,
                )
            commit_slot.skip_if_unused()
            return None
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

    if is_cancelled():
        logger.info(
            "Compilation cancelled — %d of %d table(s) compiled before stopping",
            count,
            len(tables),
        )
    else:
        logger.info("Compilation complete — %d table(s) processed", count)
    return count
