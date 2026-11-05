# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""VDB initialization — data layer and semantic layer collections.

``AUTO_ONTOLOGY_VDB_BACKEND`` selects the backend:

* ``postgres`` (default) — pgvector in the app's own Postgres, configured by
  the ``POSTGRES_*`` variables.
* ``qdrant`` — a Qdrant instance at ``QDRANT_URL`` (default
  ``http://localhost:6333``), authenticated with ``QDRANT_API_KEY`` when set.
  ``QDRANT_TIMEOUT`` sets the request timeout in seconds (default 60).

Both return a :class:`~auto_ontology.vdb.base.VectorStore`.
"""

from __future__ import annotations

import math
import os
import threading
from typing import TYPE_CHECKING, Any

from auto_ontology.infra.postgres import get_postgres_connection_string
from auto_ontology.vdb.base import VectorStore
from auto_ontology.vdb.postgres import PostgresVDB

if TYPE_CHECKING:
    from qdrant_client import QdrantClient

DATA_VDB_COLLECTION: str = "data_objects_layer"
SEMANTIC_VDB_COLLECTION: str = "semantic_layer"
VDB_SCHEMA: str = "vdb"

BACKEND_ENV: str = "AUTO_ONTOLOGY_VDB_BACKEND"
POSTGRES_BACKEND: str = "postgres"
QDRANT_BACKEND: str = "qdrant"
_BACKENDS: tuple[str, ...] = (POSTGRES_BACKEND, QDRANT_BACKEND)

__all__ = [
    "DATA_VDB_COLLECTION",
    "SEMANTIC_VDB_COLLECTION",
    "VDB_SCHEMA",
    "VectorStore",
    "get_data_vdb",
    "get_qdrant_client",
    "get_semantic_vdb",
    "get_vdb",
    "get_vdb_backend",
]


def get_vdb_backend() -> str:
    """The configured backend name, ``postgres`` unless set otherwise."""
    backend = os.environ.get(BACKEND_ENV, POSTGRES_BACKEND).strip().lower()
    if backend not in _BACKENDS:
        raise ValueError(
            f"{BACKEND_ENV}={backend!r} is not supported; "
            f"use one of {', '.join(_BACKENDS)}."
        )
    return backend


def _qdrant_timeout() -> int:
    """``QDRANT_TIMEOUT`` in whole seconds, rounded up; 60 when unset.

    The client takes an integer, so ``7.5`` becomes 8 rather than an error on
    every vector call.
    """
    raw = os.environ.get("QDRANT_TIMEOUT", "").strip() or "60"
    try:
        seconds = float(raw)
    except ValueError:
        seconds = math.nan
    if not seconds > 0 or math.isinf(seconds):
        raise ValueError(
            f"QDRANT_TIMEOUT must be a positive number of seconds, got {raw!r}."
        )
    return math.ceil(seconds)


_qdrant_client: "QdrantClient | None" = None
_qdrant_client_lock = threading.Lock()


def get_qdrant_client() -> "QdrantClient":
    """One Qdrant client per process, shared by every store.

    ``get_vdb`` runs on most write paths, so a client per call would open a
    new connection pool each time.
    """
    global _qdrant_client
    with _qdrant_client_lock:
        if _qdrant_client is None:
            from qdrant_client import QdrantClient

            _qdrant_client = QdrantClient(
                url=os.environ.get("QDRANT_URL", "http://localhost:6333"),
                api_key=os.environ.get("QDRANT_API_KEY") or None,
                timeout=_qdrant_timeout(),
            )
        return _qdrant_client


def get_vdb(
    *,
    database_name: str | None = None,
    collection_name: str = DATA_VDB_COLLECTION,
    reset: bool = False,
) -> VectorStore:
    """Build the configured vector store for *collection_name*.

    When database_name is provided, the VDB will use it to reset old embeddings for the given database.
    """
    kwargs: dict[str, Any] = {"collection_name": collection_name}
    if database_name:
        kwargs["database_name"] = database_name

    if get_vdb_backend() == QDRANT_BACKEND:
        from auto_ontology.vdb.qdrant import QdrantVDB

        return QdrantVDB(reset=reset, client=get_qdrant_client(), **kwargs)

    kwargs["connection_string"] = get_postgres_connection_string()
    kwargs["schema_name"] = VDB_SCHEMA
    return PostgresVDB(reset=reset, **kwargs)


def get_data_vdb(
    *, database_name: str | None = None, reset: bool = False
) -> VectorStore:
    """Backward-compatible alias for :func:`get_vdb` (tabular / data layer)."""
    return get_vdb(database_name=database_name, reset=reset)


def get_semantic_vdb(
    *, database_name: str | None = None, reset: bool = False
) -> VectorStore:
    """Build the vector store for the semantic layer."""
    return get_vdb(
        database_name=database_name,
        collection_name=SEMANTIC_VDB_COLLECTION,
        reset=reset,
    )
