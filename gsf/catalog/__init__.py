# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GSF-owned catalog layer — the data catalog's vocabulary and write path.

The public surface is deliberately four entry points:

* :func:`gsf.catalog.ingest_catalog` — extract a connector's catalog and write it
* :func:`gsf.catalog.sql_parse.parse_query_single` — parse one SQL statement
  against the catalog
* :func:`gsf.catalog.store.queries.add_query` — persist a parsed query
* :class:`gsf.catalog.model.Schema` / :class:`gsf.catalog.model.CatalogNode`

Everything above :mod:`gsf.catalog.store` is storage-agnostic — keep SQL and
storage types confined to ``store/``.

``ingest_catalog`` is imported lazily: it pulls in pandas, sqlglot and the store
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
