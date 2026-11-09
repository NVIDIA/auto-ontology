# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recurring scheduler for data ingestion.

Runs :func:`run_ingest` for every configured connection once at startup, then
again every ``INGEST_INTERVAL_HOURS`` hours (default 24) measured from the end
of the previous run. Connections are reloaded on every pass so newly added
ones are picked up without a restart.

A pass also replays the tagging rules, but only when semantic compilation is
switched off — with it on, the semantic pass is the better moment and does it
instead. See ``auto_ontology.ingestion_service.rules``.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import time
from datetime import datetime, timedelta, timezone

from auto_ontology.connectors.registry import (
    get_connectors,
    invalidate_connectors_cache,
)
from auto_ontology.ingestion_service.config import is_semantic_compilation_enabled
from auto_ontology.ingestion_service.ingest import run_ingest
from auto_ontology.ingestion_service.rules import replay_rules_in_thread
from auto_ontology.ingestion_service.scheduler import IntervalScheduler

logger = logging.getLogger(__name__)

# Cadence measured from the end of the previous run rather than a fixed
# wall-clock time. Overridable via INGEST_INTERVAL_HOURS (see .env.example).
DEFAULT_INGEST_INTERVAL = timedelta(hours=24)
INGEST_INTERVAL_ENV = "INGEST_INTERVAL_HOURS"


def resolve_ingest_interval() -> timedelta:
    """The ingest cadence from ``INGEST_INTERVAL_HOURS``, or 24h.

    Unset or blank means the default. Invalid values, including intervals that
    round to zero or cannot produce a representable next-run datetime, are
    logged and ignored rather than stopping the unattended scheduler.
    """
    raw = os.environ.get(INGEST_INTERVAL_ENV, "").strip()
    if not raw:
        return DEFAULT_INGEST_INTERVAL
    try:
        hours = float(raw)
        if not math.isfinite(hours) or hours <= 0:
            raise ValueError("hours must be finite and positive")
        interval = timedelta(hours=hours)
        if interval <= timedelta(0) or interval > (
            datetime.max.replace(tzinfo=timezone.utc) - datetime.now(timezone.utc)
        ):
            raise ValueError("interval is outside the scheduling range")
    except (ValueError, OverflowError):
        logger.warning(
            "ingest: ignoring %s=%r (expected positive, schedulable hours); "
            "using the %gh default",
            INGEST_INTERVAL_ENV,
            raw,
            DEFAULT_INGEST_INTERVAL.total_seconds() / 3600,
        )
        return DEFAULT_INGEST_INTERVAL
    return interval


class DataScheduler(IntervalScheduler):
    """Drives data ingestion on startup, on a timer, and on demand.

    The timer defaults to :func:`resolve_ingest_interval`, so the cadence is
    read from the environment when the scheduler is built, not when this
    module is imported.
    """

    name = "ingest"

    def __init__(self, interval: timedelta | None = None) -> None:
        super().__init__(interval or resolve_ingest_interval())

    async def _run_once(self) -> None:
        # Reload connectors each pass so newly added connections are picked up
        # without a restart. get_connectors caches, so invalidate first; the
        # returned connectors carry each connection's schema allowlist and are
        # owned by the cache (run_ingest must not close them).
        invalidate_connectors_cache()
        connectors = get_connectors()
        if not connectors:
            logger.info(
                "ingest: no connections configured. "
                "Add a connection in Settings → Connections or set "
                "CONNECTION_STRINGS in your .env."
            )
            return

        logger.info("ingest: starting (%s connection(s))", len(connectors))
        started = time.monotonic()
        succeeded = failed = 0
        for connector in connectors:
            database_name = getattr(connector, "database_name", "?")
            try:
                await asyncio.to_thread(run_ingest, connector)
                succeeded += 1
            except Exception:
                failed += 1
                logger.exception("ingest: failed for connection %s", database_name)

        # Rules are replayed at the end of the *semantic* pass when compilation
        # is on, because a rule labels both layers and the end of that pass is
        # the only point at which both are current. With compilation off there
        # is no such pass and no semantic layer to be stale, and this becomes
        # the only thing left that can put a standing rule's tags on a column
        # ingested tonight -- without it such a deployment labels the catalog
        # once, when each rule is created, and never again.
        if not is_semantic_compilation_enabled():
            await replay_rules_in_thread(self.name)

        # A per-connection failure is caught above so the remaining connections
        # still run, which means reaching this point says nothing on its own.
        # Report the tally rather than a bare "finished": the old line was
        # indistinguishable whether all connections ingested or all of them threw.
        elapsed = time.monotonic() - started
        if failed:
            logger.warning(
                "ingest: finished with errors — %d of %d connection(s) succeeded, "
                "%d failed in %.1fs",
                succeeded,
                len(connectors),
                failed,
                elapsed,
            )
        else:
            logger.info(
                "ingest: finished successfully — %d connection(s) ingested in %.1fs",
                succeeded,
                elapsed,
            )
