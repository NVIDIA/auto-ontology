# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Index creation — a no-op, because Alembic owns the schema.

Indexes and constraints are declared in ``auto_ontology/dal/schema.py`` and created once
by a migration, so there is nothing for an ingest to do. Kept as a function
because ``write.populate_tabular_data`` calls it unconditionally.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def add_indices():
    """Do nothing. Indexes are created by migration, not by ingestion."""
    logger.debug("add_indices: no-op on Postgres; indexes come from Alembic")
