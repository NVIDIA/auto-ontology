# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Carrying ``PII`` from columns onto the attributes built from them.

And withholding the values of those columns from the catalog and the semantic
VDB once tagged -- see :mod:`auto_ontology.semantic.pii_values`.

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
from auto_ontology.semantic.pii_values import withhold_pii_column_values

logger = logging.getLogger(__name__)


def run_pii_propagation(label: str) -> None:
    """Tag attributes of PII columns and withhold their values, swallowing a failure.

    *label* names the pass or database in the log line. The two steps fail
    independently: a column whose attributes could not be tagged must still
    have its values withheld.
    """
    try:
        propagate_pii_to_attributes()
    except Exception:
        logger.exception("%s: could not propagate PII tags to attributes", label)
    run_pii_withholding(label)


def run_pii_withholding(label: str, column_ids: list[str] | None = None) -> None:
    """Withhold the values of PII-tagged columns, swallowing a failure.

    See :func:`auto_ontology.semantic.pii_values.withhold_pii_column_values`.
    """
    try:
        withhold_pii_column_values(column_ids)
    except Exception:
        logger.exception("%s: could not withhold the values of PII columns", label)


async def run_pii_propagation_in_thread(label: str) -> None:
    """:func:`run_pii_propagation`, off the event loop."""
    await asyncio.to_thread(run_pii_propagation, label)
