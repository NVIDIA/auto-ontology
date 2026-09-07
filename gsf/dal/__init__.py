# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GSF data access layer, organised by topic.

Each module is the single source of truth for a domain:
  datasources     — Database / Schema / Table / Column catalog
  terms           — Term CRUD and synonym reads (semantic compilation)
  attributes      — ColumnAttribute, SEMANTIC_FK, join path traversal
  custom_analyses — CustomAnalysis and its SQL
  sql_attributes  — SqlAttribute and its SQL
  connections     — UI-managed database connection metadata
  reset           — Deleting a database's catalog/semantic rows and embeddings
  candidates      — Vector-hit enrichment at retrieval time
  exploration     — The data- and semantic-layer Exploration graphs
  search          — Fulltext global search across catalog and semantic rows
  users           — Zone-scope helpers. Users and user-to-zone grants live in
                    Prisma's tables, not here.

``schema`` holds the SQLAlchemy Core table metadata and ``session`` the pooled
engine; everything else queries through them.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def close_store() -> None:
    """Release every connection the DAL holds. Idempotent.

    The public shutdown hook, so callers do not reach into driver internals.
    Deferred import keeps importing :mod:`gsf.dal` cheap and leaves this module
    with no import-time dependency on SQLAlchemy.

    One connection to release: the pooled engine. ``dispose_engine`` is already
    idempotent and already swallows its own failures, so there is nothing left
    for this function to add beyond being the name callers know.
    """
    from gsf.dal.session import dispose_engine

    dispose_engine()
