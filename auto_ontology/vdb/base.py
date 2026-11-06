# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The interface every vector store implements.

Two implementations: :class:`~auto_ontology.vdb.postgres.PostgresVDB`, which
stores vectors the app computed, and
:class:`~auto_ontology.vdb.elasticsearch.ElasticsearchVDB`, which computes them
itself. ``embeds_server_side`` is how callers tell the two apart, and it
changes what ``retrieval`` takes: query *vectors* when it is false, query
*text* when it is true.
"""

from __future__ import annotations

from typing import Any, Iterable, Literal, Protocol


class VectorStore(Protocol):
    """A collection of embedded catalog rows, searchable by similarity."""

    #: Shape of ``retrieval``'s ``where`` filter; see
    #: :func:`auto_ontology.retrieval.data_access.semantic_search._build_metadata_where_clause`.
    metadata_filter_format: Literal["sql", "dict"]

    #: True when the store embeds text itself. Rows are then written without
    #: ``metadata["embedding"]`` and ``retrieval`` takes text, not vectors.
    embeds_server_side: bool

    def create_index(self, **kwargs: Any) -> str:
        """Create the collection if it is missing; returns its name."""
        ...

    def write_to_index(
        self, records: list, batch_size: int = 500, **kwargs: Any
    ) -> int:
        """Write *records* (see :mod:`auto_ontology.vdb.records`); returns rows written."""
        ...

    def run(self, records: list) -> int:
        """Create the collection if needed, then write *records*."""
        ...

    def delete_all(self) -> int:
        """Delete every row; returns how many were deleted."""
        ...

    def delete_by_id(self, node_id: str) -> int:
        """Delete every row whose ``metadata["id"]`` is *node_id*."""
        ...

    def delete_by_database(self, database_name: str) -> list[str]:
        """Delete every row for *database_name*; returns the deleted rows' ids."""
        ...

    def replace_rows(self, node_ids: Iterable[str], records: list) -> int:
        """Swap the rows for *node_ids* for *records*; returns rows written.

        A write that fails must leave the previous rows searchable, so an edit
        that cannot be embedded does not take the node out of retrieval.
        """
        ...

    def retrieval(
        self, queries: list, top_k: int = 10, **kwargs: Any
    ) -> list[list[dict]]:
        """The *top_k* nearest rows per query, each ``{"text", "metadata", "_distance"}``.

        ``_distance`` is cosine distance (``1 - cos``, lower is better) on both
        backends, so thresholds such as ``CLARIFY_MAX_DISTANCE`` mean the same
        thing whichever is configured. ``kwargs["where"]`` is the metadata
        filter.
        """
        ...

    def close(self) -> None:
        """Release connections; safe to call more than once."""
        ...
