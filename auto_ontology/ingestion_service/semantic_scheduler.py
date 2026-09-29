# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recurring scheduler for semantic compilation.

Runs :func:`run_semantic_compilation` for every configured connection once at
startup, then again every 24h measured from the end of the previous run. An
API-triggered run also resets that 24h timer, so the next automatic run is
always one interval after the most recent run — scheduled or manual.

Every pass re-checks the ``semantic_compilation_enabled`` settings flag and
no-ops when it is off, so disabling in the UI takes effect on the next run
(scheduled or triggered) without waiting for a service restart.

A completed pass also re-applies every tagging rule — see
:meth:`SemanticScheduler._reapply_rules`. That is the one piece of work here
which is not compilation, and it lives here because it is the point at which
both the catalog and the semantic layer a rule labels are current. It is not
the only caller: a deployment with compilation switched off never reaches this
pass at all, so the ingest pass and ``trigger_ingest`` replay the rules too.
``auto_ontology.ingestion_service.rules`` holds the shared half and the reason
the three callers choose different moments.

Compilation reads the catalog (databases/schemas/tables) that
:class:`DataScheduler` writes to the store. Both schedulers start their first pass at the same moment on
service boot, so without coordination this one could race ahead and compile
against a catalog that isn't there yet — silently producing zero Terms instead
of an error. Passing the ingest scheduler as *depends_on* closes that race: the
very first pass waits for the ingest scheduler's first pass to finish (see
``IntervalScheduler.wait_first_pass``) before touching the catalog. Every pass
after that returns immediately, since the dependency is long since satisfied.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import timedelta

from auto_ontology.ingestion_service.config import is_semantic_compilation_enabled
from auto_ontology.ingestion_service.connections import resolve_database_names
from auto_ontology.ingestion_service.history import (
    RUN_FAILED,
    RUN_SUCCEEDED,
    record_run_finish,
    record_run_start,
)
from auto_ontology.ingestion_service.rules import replay_rules_in_thread
from auto_ontology.ingestion_service.scheduler import IntervalScheduler
from auto_ontology.semantic.cancellation import clear_cancel, request_cancel
from auto_ontology.semantic.compile import run_semantic_compilation

logger = logging.getLogger(__name__)

# Cadence measured from the end of the previous run rather than a fixed
# wall-clock time.
SEMANTIC_INTERVAL = timedelta(hours=24)

# Appended when a pass compiles nothing. The count is tables that *needed*
# compiling (``compile_semantic_layer`` works from the tables with no Term yet),
# so zero is a legitimate outcome — while the elapsed time can still be minutes,
# because the FK, SqlAttribute and bridge-table stages run afterwards
# regardless. Without this, "0 table(s) ... in 620.7s" reads as a stall or a
# silently failed run.
#
# Deliberately does not name a cause. Zero establishes only that nothing needed
# compiling; whether that is a settled catalog, one whose tables are all
# excluded by a connection's table filter, or an empty catalog, this cannot
# tell — and claiming the first would hide the other two.
_NOTHING_TO_COMPILE = (
    " (no table needed compiling; the elapsed time is the FK, SqlAttribute "
    "and bridge-table stages, which run on every pass)"
)


class SemanticScheduler(IntervalScheduler):
    """Drives semantic compilation on startup, on a 24h timer, and on demand."""

    name = "semantic"

    def __init__(
        self,
        interval: timedelta = SEMANTIC_INTERVAL,
        *,
        depends_on: IntervalScheduler | None = None,
    ) -> None:
        super().__init__(interval)
        # The scheduler that writes the catalog this one compiles from — see
        # the module docstring. Optional so tests/callers that don't care about
        # the startup race can still construct one on its own.
        self._depends_on = depends_on

    def abort(self) -> None:
        """Stop the in-flight pass at the next *table*, not the next database.

        The base class only sets its own flag, which the pass checks between
        databases — so an abort could take as long as the slowest single
        database (312s observed here). Cancelling the semantic work as well
        makes queued tables no-ops and skips the post-compile stages, which
        brings a stop down to roughly one table.
        """
        request_cancel()
        super().abort()

    async def _reapply_rules(self) -> None:
        """Re-label the catalog through every stored rule, after compiling it.

        A rule is a saved search plus the tags to apply to what it matches, and
        it goes on matching as the catalog grows -- a column ingested last
        night is not labelled by anything else while compilation is on. See
        ``auto_ontology.ingestion_service.rules`` for the other two callers and
        why the three disagree about when to run.

        Here rather than only in the data scheduler because a rule labels
        *both* layers: Tables and Columns, which ingest writes, and Terms and
        their attributes, which this pass writes. Running it at the end of this
        pass is the only point at which both are current -- and this pass
        already waits for ingest's (see the module docstring), so the catalog
        beneath the semantic layer is current too.

        Only reached when the pass ran to completion. A stop or a mid-run
        disable returns before this, which is the right way round: those cut
        the pass short at a database boundary, so the semantic layer is half
        written, and labels applied against half of it would be taken back
        again on the next pass.
        """
        await replay_rules_in_thread(self.name)

    async def _run_once(self) -> None:
        # A cancellation belongs to the pass that was running when it was
        # requested; leaving it set would abort this one before it starts.
        clear_cancel()

        # Re-check the settings flag on every pass so a disable is honored live:
        # the loop keeps ticking but no-ops until re-enabled, rather than running
        # until the next service restart. Mirrors the lifespan startup gate.
        if not is_semantic_compilation_enabled():
            logger.info("semantic: compilation disabled in settings; skipping run")
            return

        if self._depends_on is not None:
            if not self._depends_on.first_pass_done:
                logger.info(
                    "semantic: waiting for %s's first pass before compiling",
                    self._depends_on.name,
                )
            # No-ops once the dependency has had its first pass — every run
            # after the very first one returns immediately here.
            await self._depends_on.wait_first_pass()

        databases = resolve_database_names()
        if not databases:
            logger.info(
                "semantic: no connections configured; nothing to compile. "
                "Add a connection in Settings → Connections."
            )
            return

        logger.info("semantic: starting (%d database(s))", len(databases))
        started = time.monotonic()
        succeeded = failed = 0
        total_tables = 0
        run_id = record_run_start()
        # Starts optimistic; only ever downgraded, never upgraded back — a
        # stop/disable after a real failure must still be recorded as failed,
        # not left NULL, so a genuine bug isn't masked by the fact that the run
        # was also stopped. An explicit stop/disable downgrades to None instead,
        # which record_run_finish treats as a no-op — see history.py on why an
        # aborted run leaves the row NULL rather than storing an outcome.
        outcome: str | None = RUN_SUCCEEDED
        try:
            for index, database_name in enumerate(databases):
                # Checked per database rather than once per pass, so a stop
                # request or a disable ends the run at the next boundary instead
                # of after every database. The database in flight is not
                # abandoned — Python cannot kill its worker threads — but
                # ``abort`` also cancels the semantic work, so it stops at its
                # next table and skips the post-compile stages rather than
                # running to completion.
                if self.aborting:
                    logger.info(
                        "semantic: stopped on request; %d database(s) not compiled",
                        len(databases) - index,
                    )
                    if outcome != RUN_FAILED:
                        outcome = None
                    return
                if not is_semantic_compilation_enabled():
                    logger.info(
                        "semantic: disabled mid-run; %d database(s) not compiled",
                        len(databases) - index,
                    )
                    if outcome != RUN_FAILED:
                        outcome = None
                    return

                try:
                    database_started = time.monotonic()
                    tables_processed = await asyncio.to_thread(
                        run_semantic_compilation, database_name
                    )
                    succeeded += 1
                    total_tables += tables_processed
                    # "compiled", not "processed": the count is tables that
                    # *needed* compiling. The pass summary carries the why when
                    # it is zero; repeating it per database would be five copies
                    # of the same sentence.
                    logger.info(
                        "Finished semantic compilation successfully for database %s: "
                        "%d table(s) compiled in %.1fs",
                        database_name,
                        tables_processed,
                        time.monotonic() - database_started,
                    )
                except Exception:
                    failed += 1
                    outcome = RUN_FAILED
                    logger.exception("semantic: failed for database %s", database_name)

            await self._reapply_rules()
        finally:
            # A no-op when outcome is None (stop/disable) — see
            # record_run_finish's docstring on why that leaves the row NULL
            # rather than recording an "aborted" outcome.
            record_run_finish(run_id, outcome)
            # As in the data scheduler: per-database failures are caught so the
            # rest still compile, so the closing line has to carry the tally to
            # mean anything.
            elapsed = time.monotonic() - started
            if outcome is None:
                # The early return above already logged *why* it stopped; this
                # says how far it got before it did.
                logger.info(
                    "semantic: stopped — %d of %d database(s) compiled, "
                    "%d table(s) processed in %.1fs",
                    succeeded,
                    len(databases),
                    total_tables,
                    elapsed,
                )
            elif failed:
                logger.warning(
                    "semantic: finished with errors — %d of %d database(s) succeeded, "
                    "%d failed, %d table(s) processed in %.1fs",
                    succeeded,
                    len(databases),
                    failed,
                    total_tables,
                    elapsed,
                )
            else:
                logger.info(
                    "semantic: finished successfully — %d database(s), "
                    "%d table(s) compiled in %.1fs%s",
                    succeeded,
                    total_tables,
                    elapsed,
                    "" if total_tables else _NOTHING_TO_COMPILE,
                )
