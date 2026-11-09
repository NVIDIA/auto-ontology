# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The interface every vector store backend implements.

:func:`auto_ontology.vdb.get_vdb` returns one of these. Callers type against
:class:`VectorStore` so the backend (pgvector or Qdrant) is a deployment
choice, not a code change.
"""

from __future__ import annotations

from typing import Any, Iterable, Protocol, runtime_checkable


@runtime_checkable
class VectorStore(Protocol):
    """A collection of embedded records with metadata filters.

    A record is ``{"document_type", "metadata"}`` where ``metadata`` carries
    ``embedding``, ``content``, ``id``, ``label``, ``database_name`` and any
    other keys retrieval reads back (see :mod:`auto_ontology.vdb.records`).

    A search hit is ``{"text", "metadata", "_distance"}``. ``_distance`` is
    the cosine distance, so lower is better and an identical vector scores 0.
    Callers compare it against thresholds such as ``DEFAULT_MAX_DISTANCE``,
    so every backend must report the same scale.
    """

    #: Shape of the ``where`` filter callers pass to :meth:`retrieval`.
    #: ``"dict"`` is a flat ``{key: value | [values]}`` mapping.
    metadata_filter_format: str
    collection_name: str

    def create_index(self, **kwargs: Any) -> str:
        """Create the collection if it does not exist; return its name."""
        ...

    def write_to_index(
        self, records: list, batch_size: int = 500, **kwargs: Any
    ) -> int:
        """Insert records; return the number written."""
        ...

    def run(self, records: list) -> int:
        """Create the collection if needed, then write *records* to it."""
        ...

    def retrieval(
        self, queries: list, top_k: int = 10, **kwargs: Any
    ) -> list[list[dict]]:
        """Nearest records for each query vector, best first."""
        ...

    def delete_by_database(self, database_name: str) -> int:
        """Delete every record of *database_name*; return the count."""
        ...

    def delete_by_id(self, node_id: str) -> int:
        """Delete every record whose metadata ``id`` is *node_id*."""
        ...

    def delete_by_ids(self, node_ids: Iterable[str]) -> int:
        """Delete every record whose metadata ``id`` is in *node_ids*, in one call."""
        ...

    def delete_all(self) -> int:
        """Delete every record in the collection; return the count."""
        ...

    def close(self) -> None:
        """Release connections. Safe to call more than once."""
        ...
