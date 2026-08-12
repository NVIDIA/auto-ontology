# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Index creation — a no-op, because Alembic owns the schema.

The Neo4j implementation creates a uniqueness constraint and two indexes per
label on **every ingest**, because a schemaless store has nowhere else to put
them. Here they are declared in ``gsf/dal/pg/schema.py`` and created once by a
migration, so there is nothing for an ingest to do.

Kept as a function rather than deleted because ``write.py`` calls it
unconditionally at the top of ``populate_tabular_data``, and that call site is
storage-agnostic code the fork does not edit. Phase 11 removes both.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def add_indices():
    """Do nothing. Indexes are created by migration, not by ingestion."""
    logger.debug("add_indices: no-op on Postgres; indexes come from Alembic")
