# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cooperative cancellation for an in-flight semantic compilation pass.

Why cooperative rather than a hard kill: compilation runs in worker threads
(``asyncio.to_thread`` per database, a ``ThreadPoolExecutor`` per table), and
Python cannot kill a thread from outside. Cancelling the awaiting coroutine
only stops the *waiting* — the worker keeps running, and keeps writing Terms
and embeddings where nothing is left watching it. That is strictly worse than
letting it finish, because a reset would then delete underneath a writer it
cannot see.

So the pass checks a flag at the points where stopping is safe and cheap:

* before each table is processed, so queued tables become no-ops and the pool
  drains in about the time one table takes rather than one database;
* between the post-compile stages (FK resolution, SqlAttribute suggestion,
  bridge tables), which are the long tail — FK resolution alone has been
  measured at over 300s on a single database.

Tables already executing when the flag is set still finish; that is bounded by
the pool width, which is the price of not orphaning a writer.

The flag is process-wide because there is one compilation pass per ingestion
service, and it is a :class:`threading.Event` rather than an asyncio one so the
worker threads can read it without touching the loop.
"""

from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)

_cancelled = threading.Event()


def request_cancel() -> None:
    """Ask the in-flight pass to stop at its next safe point."""
    if not _cancelled.is_set():
        logger.info("semantic: cancellation requested; stopping at the next table")
    _cancelled.set()


def clear_cancel() -> None:
    """Drop a pending cancellation, so the next pass starts uncancelled."""
    _cancelled.clear()


def is_cancelled() -> bool:
    """Whether the running pass has been asked to stop."""
    return _cancelled.is_set()
