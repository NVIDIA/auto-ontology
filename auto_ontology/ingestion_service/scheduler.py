# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Interval scheduler shared by the ingestion service's background jobs.

A scheduler runs its job once at startup, then again every ``interval``
measured from the *end* of the previous run. An API-triggered run happens
immediately and, because the loop always reschedules from "now" after any run,
resets that interval timer.

Subclasses implement :meth:`_run_once`. A subclass whose work depends on
another scheduler's data (e.g. semantic compilation reading the catalog that
data ingestion writes) should await that scheduler's :meth:`wait_first_pass`
at the top of its own ``_run_once`` — see ``SemanticScheduler`` — so its
startup run can never race the dependency's startup run.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

# Upper bound on a single sleep chunk so shutdown/trigger events are observed
# promptly even while waiting out a full interval.
_WAIT_CHUNK_SECONDS = 500


class IntervalScheduler:
    """Runs a job at startup, on a fixed interval, and on demand."""

    #: Short label used as a log prefix; overridden by subclasses.
    name = "scheduler"

    def __init__(self, interval: timedelta) -> None:
        self._interval = interval
        self._stop = asyncio.Event()
        self._trigger = asyncio.Event()
        self._abort = asyncio.Event()
        self._running = False
        self._paused = False
        self._first_pass_done = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    def start(self) -> bool:
        """Launch the background loop; runs one pass immediately.

        Returns ``True`` if this call started the loop (so a startup run is now
        underway), or ``False`` if it was already running. Callers that also
        want an on-demand run should only ``trigger()`` when this returns
        ``False`` — otherwise the fresh startup run and the trigger would
        compile twice.
        """
        if self._task is not None:
            return False
        self._task = asyncio.create_task(self._run_forever())
        return True

    async def stop(self) -> None:
        """Signal the loop to exit and wait for the in-flight wait to unwind."""
        self._stop.set()
        if self._task is not None:
            await self._task

    def trigger(self) -> None:
        """Request an immediate run without blocking the caller.

        The run happens on the scheduler's own task; when it finishes the loop
        reschedules the next automatic run one interval later.
        """
        self._trigger.set()

    def abort(self) -> None:
        """Ask the in-flight pass to stop at its next safe point.

        Only the current pass is cut short: the loop keeps running and the next
        automatic run happens on schedule. A queued trigger is dropped as well,
        so aborting cannot be followed immediately by another pass.
        """
        self._abort.set()
        self._trigger.clear()

    def pause(self) -> None:
        """Suppress passes until :meth:`resume`.

        ``abort`` alone is not enough to hold the scheduler off: it stops the
        *current* pass, but the loop keeps ticking and ``_run_pass`` clears the
        abort flag before each new one. A caller that has to mutate the data a
        pass reads — the semantic reset deleting the layer — would otherwise
        race a timer tick or an API trigger that starts a pass against the
        half-changed state.
        """
        self._paused = True

    def resume(self) -> None:
        """Allow passes again after :meth:`pause`."""
        self._paused = False

    @property
    def paused(self) -> bool:
        """Whether passes are currently suppressed."""
        return self._paused

    @property
    def aborting(self) -> bool:
        """Whether the pass currently running was asked to stop."""
        return self._abort.is_set()

    @property
    def running(self) -> bool:
        """Whether a pass is executing right now, as opposed to idling between ticks."""
        return self._running

    @property
    def first_pass_done(self) -> bool:
        """Whether this scheduler's first pass has finished yet.

        Lets a dependent scheduler log only when :meth:`wait_first_pass` is
        actually going to block, instead of on every one of its own passes.
        """
        return self._first_pass_done.is_set()

    async def wait_first_pass(self) -> None:
        """Wait until this scheduler's first pass — successful or not — has
        finished.

        Stays satisfied forever once that pass ends, so only a caller racing
        the very first pass ever blocks here; every call after that returns
        immediately. Meant for a dependent scheduler to await before doing any
        work of its own — see the module docstring.
        """
        await self._first_pass_done.wait()

    async def _run_once(self) -> None:
        """Perform one pass of the job. Implemented by subclasses."""
        raise NotImplementedError

    async def _run_pass(self) -> None:
        """Run one pass with a fresh abort flag, surviving unhandled errors.

        A paused scheduler skips the pass entirely rather than aborting it part
        way: the point of the pause is that the data a pass reads is being
        rewritten, so starting one at all is what has to be prevented.
        """
        if self._paused:
            logger.info("%s: paused; skipping this pass", self.name)
            return
        self._abort.clear()
        self._running = True
        try:
            await self._run_once()
        except Exception:
            logger.exception("%s: unhandled error; will retry on next tick", self.name)
        finally:
            self._running = False
            self._first_pass_done.set()

    async def _wait_for_next(self, next_run: datetime) -> None:
        """Sleep until ``next_run``, returning early on stop or trigger."""
        while not self._stop.is_set() and not self._trigger.is_set():
            remaining = (next_run - datetime.now(timezone.utc)).total_seconds()
            if remaining <= 0:
                return
            waiters = [
                asyncio.create_task(self._stop.wait()),
                asyncio.create_task(self._trigger.wait()),
            ]
            try:
                await asyncio.wait(
                    waiters,
                    timeout=min(remaining, _WAIT_CHUNK_SECONDS),
                    return_when=asyncio.FIRST_COMPLETED,
                )
            finally:
                for w in waiters:
                    w.cancel()

    async def _run_forever(self) -> None:
        # Run once at startup so a fresh deploy refreshes existing connections
        # without waiting for the next scheduled slot.
        await self._run_pass()

        while not self._stop.is_set():
            # Schedule the next run one interval out from now, so a long run or a
            # manual trigger naturally pushes the next automatic run back.
            next_run = datetime.now(timezone.utc) + self._interval
            logger.info("%s: next run at %s", self.name, next_run.isoformat())
            await self._wait_for_next(next_run)
            if self._stop.is_set():
                break

            # A trigger fired (or the interval elapsed) — consume it and run now.
            self._trigger.clear()
            await self._run_pass()
