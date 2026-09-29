# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replaying the tagging rules once a pass has finished changing the catalog.

A rule is a standing instruction rather than a one-off labelling -- see
``auto_ontology.server.rules.service`` -- so something has to replay it over
what was ingested since the last time, and nothing else in either service
will. This is the shared half of that: containing a failure and getting the
synchronous work off the event loop. *When* to call it is the caller's
decision, and the three callers do not agree, for reasons each documents:

* The semantic pass runs it on every completed pass, because a rule labels the
  semantic layer too and that pass is the only point at which both layers are
  current.
* The ingest pass runs it only when semantic compilation is off, because then
  there is no semantic pass to do it and no semantic layer to be stale.
* ``trigger_ingest`` runs it unconditionally, because creating a connection
  ingests a whole database without going through either scheduler.
"""

from __future__ import annotations

import asyncio
import logging

from auto_ontology.server.rules.service import reapply_rules

logger = logging.getLogger(__name__)


def replay_rules(label: str) -> None:
    """Re-label the catalog through every stored rule, swallowing a failure.

    *label* prefixes the log line with the pass that asked, matching the
    scheduler log prefixes.

    A failure here must not fail the pass that called it. Per-rule failures are
    already contained one level down, so reaching this handler means the whole
    rule pass could not run -- and recording the *ingest* or the *compilation*
    as failed for it would put a red state on the settings page against work
    that in fact succeeded.
    """
    try:
        reapply_rules()
    except Exception:
        logger.exception("%s: could not re-apply tagging rules", label)


async def replay_rules_in_thread(label: str) -> None:
    """:func:`replay_rules`, off the event loop.

    In a thread for the reason the compilation and the ingest themselves are:
    this is synchronous database work, and awaiting it on the loop would block
    the scheduler's own timers.
    """
    await asyncio.to_thread(replay_rules, label)
