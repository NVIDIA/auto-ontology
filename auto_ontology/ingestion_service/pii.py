# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Carrying ``PII`` from columns onto the attributes built from them.

Attributes are written by semantic compilation, after the ingest that classified
their columns, so two passes call this: the ingest pass (attributes from an
earlier compilation, new PII columns) and the semantic pass (attributes it just
created). Both are idempotent.

Like :mod:`auto_ontology.ingestion_service.rules`, this is the shared half:
containing a failure and getting synchronous work off the event loop. A failure
here must not mark an ingest or a compilation as failed -- the work they did
succeeded.
"""

from __future__ import annotations

import asyncio
import logging

from auto_ontology.pii_detection.service import propagate_pii_to_attributes

logger = logging.getLogger(__name__)


def run_pii_propagation(label: str) -> None:
    """Tag attributes of PII columns, swallowing a failure.

    *label* names the pass or database in the log line.
    """
    try:
        propagate_pii_to_attributes()
    except Exception:
        logger.exception("%s: could not propagate PII tags to attributes", label)


async def run_pii_propagation_in_thread(label: str) -> None:
    """:func:`run_pii_propagation`, off the event loop."""
    await asyncio.to_thread(run_pii_propagation, label)
