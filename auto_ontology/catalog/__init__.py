# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Auto Ontology-owned catalog layer — the data catalog's vocabulary and write path.

The public surface is deliberately four entry points:

* :func:`auto_ontology.catalog.ingest_catalog` — extract a connector's catalog and write it
* :func:`auto_ontology.catalog.sql_parse.parse_query_single` — parse one SQL statement
  against the catalog
* :func:`auto_ontology.catalog.store.queries.add_query` — persist a parsed query
* :class:`auto_ontology.catalog.model.Schema` / :class:`auto_ontology.catalog.model.CatalogNode`

Everything above :mod:`auto_ontology.catalog.store` is storage-agnostic — keep SQL and
storage types confined to ``store/``.

``ingest_catalog`` is imported lazily: it pulls in pandas, sqlglot and the store
driver, and several modules import :mod:`auto_ontology.catalog.constants` for nothing but
the label vocabulary.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from auto_ontology.catalog.ingest import ingest_catalog

__all__ = ["ingest_catalog"]


def __getattr__(name: str) -> Any:
    if name == "ingest_catalog":
        from auto_ontology.catalog.ingest import ingest_catalog

        return ingest_catalog
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
