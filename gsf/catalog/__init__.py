# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GSF-owned catalog layer — the data catalog's vocabulary and write path.

Forked out of ``nemo_retriever.tabular_data.ingestion`` in Phase 1 of the
drop-Neo4j refactor (``docs/refactor/drop-neo4j/PLAN.md``).

The public surface is deliberately four entry points:

* :func:`gsf.catalog.ingest_catalog` — extract a connector's catalog and write it
* :func:`gsf.catalog.sql_parse.parse_query_single` — parse one SQL statement
  against the catalog
* :func:`gsf.catalog.store.queries.add_query` — persist a parsed query
* :class:`gsf.catalog.model.Schema` / :class:`gsf.catalog.model.CatalogNode`

Everything above :mod:`gsf.catalog.store` is storage-agnostic. ``store/`` is the
only subtree Phase 4 rewrites for Postgres — keep Cypher and driver types out of
everything else.

``ingest_catalog`` is imported lazily: it pulls in pandas, sqlglot and the neo4j
driver, and several modules import :mod:`gsf.catalog.constants` for nothing but
the label vocabulary.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from gsf.catalog.ingest import ingest_catalog

__all__ = ["ingest_catalog"]


def __getattr__(name: str) -> Any:
    if name == "ingest_catalog":
        from gsf.catalog.ingest import ingest_catalog

        return ingest_catalog
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
