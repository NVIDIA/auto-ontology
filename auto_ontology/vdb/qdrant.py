# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qdrant vector store for the data and semantic layers.

Select it with ``AUTO_ONTOLOGY_VDB_BACKEND=qdrant`` (see
:func:`auto_ontology.vdb.get_vdb`). It implements the same
:class:`~auto_ontology.vdb.base.VectorStore` interface as
:class:`~auto_ontology.vdb.postgres.PostgresVDB`, so retrieval, ingestion and
reset code run unchanged on either backend.

Each record becomes one point:

* the vector is the record's ``metadata["embedding"]``, stored as float32 at
  its full dimension (2048 for the default embedding model);
* the payload is ``{"page_content": <text>, "metadata": {...}}``, where
  ``metadata`` is the record metadata without the embedding, plus
  ``document_type``. Search hits return this ``metadata`` unchanged, so callers
  read ``id``, ``label`` and ``name`` from the same place on both backends.

Filters address ``metadata.<key>``. The keys callers filter on (``label``,
``database_name``, ``schema_name`` and ``id``) get keyword payload indexes, so
Qdrant applies the filter while it searches the HNSW graph. ``database_name``
is indexed as the tenant key, which keeps each database's points together on
disk.

Qdrant reports cosine *similarity*. This store converts it to cosine distance
(``1 - similarity``) so ``_distance`` keeps the scale callers compare against
``DEFAULT_MAX_DISTANCE``.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
import weakref
from typing import Any, Iterable, Optional

from qdrant_client import QdrantClient, models

from auto_ontology.vdb.records import to_vdb_documents

logger = logging.getLogger(__name__)

_PAYLOAD_TEXT_KEY = "page_content"
_PAYLOAD_METADATA_KEY = "metadata"
_DATABASE_KEY = "database_name"
_LABEL_KEY = "label"
_SCHEMA_KEY = "schema_name"
_ID_KEY = "id"

#: Collections known to exist with every payload index, per client.
#: ``get_vdb()`` builds a new store on almost every call, so a flag on the store
#: would repeat the setup requests each time. Keyed weakly, so a closed client
#: takes its entries with it.
_ready_collections: "weakref.WeakKeyDictionary[QdrantClient, set[str]]" = (
    weakref.WeakKeyDictionary()
)
_ready_lock = threading.Lock()

#: Metadata keys that get a keyword payload index. ``database_name`` is the
#: tenant key: every query and every reset is scoped to one database.
_INDEXED_KEYS: dict[str, bool] = {
    _DATABASE_KEY: True,
    _LABEL_KEY: False,
    _SCHEMA_KEY: False,
    _ID_KEY: False,
}


def _metadata_path(key: str) -> str:
    return f"{_PAYLOAD_METADATA_KEY}.{key}"


def _condition(key: str, value: Any) -> models.FieldCondition:
    """One ``FieldCondition`` for a flat filter entry.

    A list becomes ``MatchAny`` (an ``IN`` list), the same meaning
    langchain-postgres gives a list value.
    """
    if isinstance(value, (list, tuple, set)):
        return models.FieldCondition(
            key=_metadata_path(key), match=models.MatchAny(any=list(value))
        )
    return models.FieldCondition(
        key=_metadata_path(key), match=models.MatchValue(value=value)
    )


def build_filter(where: Any) -> Optional[models.Filter]:
    """Translate a ``where`` argument into a Qdrant ``Filter``.

    ``where`` is the flat ``{key: value | [values]}`` mapping that
    :func:`~auto_ontology.retrieval.data_access.semantic_search._build_metadata_where_clause`
    builds for ``metadata_filter_format == "dict"``. Every entry must match.
    A ``models.Filter`` passes through unchanged, and an empty value means no
    filter.
    """
    if not where:
        return None
    if isinstance(where, models.Filter):
        return where
    if not isinstance(where, dict):
        raise TypeError(
            f"QdrantVDB expects a dict filter, got {type(where).__name__}: {where!r}"
        )
    return models.Filter(must=[_condition(k, v) for k, v in where.items()])


class QdrantVDB:
    """Vector store backed by Qdrant.

    Connection settings come from the kwargs, which
    :func:`auto_ontology.vdb.get_vdb` fills from ``QDRANT_URL``,
    ``QDRANT_API_KEY`` and ``QDRANT_TIMEOUT``. Pass ``client`` to share or
    inject a :class:`~qdrant_client.QdrantClient`, for example
    ``QdrantClient(":memory:")`` in tests.
    """

    #: Callers build the ``where`` filter as a flat dict; see :func:`build_filter`.
    metadata_filter_format = "dict"

    def __init__(self, *, reset: bool = False, **kwargs: Any) -> None:
        self.collection_name: str = kwargs.get(
            "collection_name", kwargs.get("index_name", "data_objects_layer")
        )
        self.vector_size: int = int(kwargs.get("vector_size") or 2048)

        self._client: Optional[QdrantClient] = kwargs.get("client")
        self._owns_client = self._client is None
        self._client_lock = threading.Lock()
        self._client_kwargs: dict[str, Any] = {
            "url": kwargs.get("url") or "http://localhost:6333",
            "api_key": kwargs.get("api_key") or None,
            "timeout": int(kwargs.get("timeout") or 60),
        }

        self.database_name: Optional[str] = kwargs.get("database_name")
        if self.database_name and reset:
            deleted = self.delete_by_database(self.database_name)
            logger.info(
                "QdrantVDB.delete_by_database: deleted %d points for database %s",
                deleted,
                self.database_name,
            )

    # ------------------------------------------------------------------
    # Collection lifecycle
    # ------------------------------------------------------------------

    @property
    def client(self) -> QdrantClient:
        """The Qdrant client, created on first use and again after :meth:`close`."""
        with self._client_lock:
            if self._client is None:
                self._client = QdrantClient(**self._client_kwargs)
            return self._client

    @property
    def _collection_ready(self) -> bool:
        """Whether the collection is known to exist with every payload index.

        Shared by every store on the same client. Writes and deletes trust it.
        A call that then finds the collection gone clears it, and the next
        write builds the collection again.
        """
        with _ready_lock:
            return self.collection_name in _ready_collections.get(self.client, ())

    @_collection_ready.setter
    def _collection_ready(self, ready: bool) -> None:
        client = self.client
        with _ready_lock:
            names = _ready_collections.setdefault(client, set())
            if ready:
                names.add(self.collection_name)
            else:
                names.discard(self.collection_name)

    def _collection_exists(self) -> bool:
        """Ask the server, without the shared cache."""
        return self.client.collection_exists(self.collection_name)

    def _missing_after_error(self) -> bool:
        """After a failed call: is the collection gone? ``False`` when unsure.

        Clears ``_collection_ready`` when it is gone. Never raises, so the
        caller can re-raise its own error rather than this check's.
        """
        try:
            missing = not self._collection_exists()
        except Exception:
            logger.debug("QdrantVDB: collection check failed", exc_info=True)
            return False
        if missing:
            self._collection_ready = False
        return missing

    def _check_vector_params(self, info: models.CollectionInfo) -> None:
        """Refuse a collection whose vectors this store cannot write or rank."""
        vectors = info.config.params.vectors
        if (
            isinstance(vectors, models.VectorParams)
            and vectors.size == self.vector_size
            and vectors.distance == models.Distance.COSINE
        ):
            return
        raise ValueError(
            f"Qdrant collection {self.collection_name} has vectors {vectors!r}, "
            f"but this store writes one unnamed {self.vector_size}-dimension "
            "vector with cosine distance. Drop the collection, or change "
            "EMBED_MODEL back to the model that filled it."
        )

    def _ensure_collection(self) -> None:
        """Create the collection if needed, then any payload index it lacks.

        The index check runs whether or not this call created the collection,
        so a collection left without indexes (a crash between the two steps,
        or one created by hand) gets them on the next write.
        """
        if self._collection_ready:
            return
        created = False
        if not self._collection_exists():
            try:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.vector_size, distance=models.Distance.COSINE
                    ),
                )
                created = True
            except Exception:
                # Another process created it between the check and the create.
                if not self._collection_exists():
                    raise

        info = self.client.get_collection(self.collection_name)
        self._check_vector_params(info)
        existing = set(info.payload_schema or {})
        for key, is_tenant in _INDEXED_KEYS.items():
            if _metadata_path(key) in existing:
                continue
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=_metadata_path(key),
                field_schema=models.KeywordIndexParams(
                    type=models.KeywordIndexType.KEYWORD, is_tenant=is_tenant
                ),
            )
        self._collection_ready = True
        if created:
            logger.info(
                "QdrantVDB: created collection %s (size=%d, cosine)",
                self.collection_name,
                self.vector_size,
            )

    def create_index(self, **kwargs: Any) -> str:
        """Ensure the collection and its payload indexes exist."""
        self._ensure_collection()
        return self.collection_name

    # ------------------------------------------------------------------
    # Read / write
    # ------------------------------------------------------------------

    def _upsert(self, points: list[models.PointStruct]) -> None:
        """Upsert *points*, rebuilding the collection once if it was dropped."""
        try:
            self.client.upsert(
                collection_name=self.collection_name, points=points, wait=True
            )
        except Exception:
            if not self._missing_after_error():
                raise
            logger.warning(
                "QdrantVDB: collection %s disappeared; creating it again",
                self.collection_name,
            )
            self._ensure_collection()
            self.client.upsert(
                collection_name=self.collection_name, points=points, wait=True
            )

    def write_to_index(
        self,
        records: list,
        batch_size: int = 500,
        **kwargs: Any,
    ) -> int:
        """Upsert records as new points; return the number written."""
        documents, skipped = to_vdb_documents(records)
        if not documents:
            logger.info(
                "QdrantVDB.write_to_index: no points to insert (skipped %d)", skipped
            )
            return 0

        points = [
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=document.embedding,
                payload={
                    _PAYLOAD_TEXT_KEY: document.text,
                    _PAYLOAD_METADATA_KEY: document.metadata,
                },
            )
            for document in documents
        ]
        self._ensure_collection()
        for start in range(0, len(points), batch_size):
            self._upsert(points[start : start + batch_size])

        logger.info(
            "QdrantVDB.write_to_index: inserted %d points into %s (skipped %d)",
            len(points),
            self.collection_name,
            skipped,
        )
        return len(points)

    def run(self, records: list) -> int:
        """Create the collection if needed, then write records to it."""
        self.create_index()
        return self.write_to_index(records)

    def retrieval(
        self,
        queries: list,
        top_k: int = 10,
        **kwargs: Any,
    ) -> list[list[dict]]:
        """Cosine k-NN search for each query vector, in one batch request.

        ``kwargs["where"]`` is a flat filter dict, for example
        ``{"label": ["Column", "Table"], "database_name": "prod"}``. A missing
        collection returns empty results, the same as an empty one.
        """
        if not queries:
            return []
        started = time.perf_counter()
        try:
            responses = self.client.query_batch_points(
                collection_name=self.collection_name,
                requests=[
                    models.QueryRequest(
                        query=[float(v) for v in query],
                        filter=build_filter(kwargs.get("where")),
                        limit=top_k,
                        with_payload=True,
                    )
                    for query in queries
                ],
            )
        except Exception as e:
            # Checked only after a failure, so a search costs one request.
            if not self._missing_after_error():
                logger.error(f"Error in retrieval: {e}")
            return [[] for _ in queries]

        results = [
            [
                {
                    "text": (point.payload or {}).get(_PAYLOAD_TEXT_KEY) or "",
                    "metadata": (point.payload or {}).get(_PAYLOAD_METADATA_KEY) or {},
                    "_distance": 1.0 - float(point.score),
                }
                for point in response.points
            ]
            for response in responses
        ]
        logger.debug(
            "vdb.retrieval total=%.3fs queries=%d",
            time.perf_counter() - started,
            len(queries),
        )
        return results

    # ------------------------------------------------------------------
    # Deletes
    # ------------------------------------------------------------------

    def _delete_matching(self, query_filter: models.Filter, what: str) -> int:
        """Delete the points *query_filter* matches; return how many there were.

        Counts first because Qdrant's delete does not report a count. Points
        written between the two calls are deleted but not counted. A
        collection known to be ready skips the existence check, so a delete
        costs two requests.
        """
        if not self._collection_ready and not self._collection_exists():
            return self._nothing_to_delete(what)
        try:
            deleted = self.client.count(
                collection_name=self.collection_name,
                count_filter=query_filter,
                exact=True,
            ).count
            if deleted:
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=models.FilterSelector(filter=query_filter),
                    wait=True,
                )
        except Exception:
            if not self._missing_after_error():
                raise
            return self._nothing_to_delete(what)
        logger.info(
            "QdrantVDB.%s: deleted %d points from %s",
            what,
            deleted,
            self.collection_name,
        )
        return deleted

    def _nothing_to_delete(self, what: str) -> int:
        logger.info(
            "QdrantVDB.%s: collection %s not found, nothing to delete",
            what,
            self.collection_name,
        )
        return 0

    def delete_by_database(self, database_name: str) -> int:
        """Delete every point of *database_name*; return the count."""
        return self._delete_matching(
            models.Filter(must=[_condition(_DATABASE_KEY, database_name)]),
            "delete_by_database",
        )

    def delete_by_id(self, node_id: str) -> int:
        """Delete every point whose metadata ``id`` is *node_id*."""
        return self.delete_by_ids([node_id])

    def delete_by_ids(self, node_ids: Iterable[str]) -> int:
        """Delete every point whose metadata ``id`` is in *node_ids*, in one call."""
        ids = list(dict.fromkeys(node_ids))
        if not ids:
            return 0
        return self._delete_matching(
            models.Filter(must=[_condition(_ID_KEY, ids)]), "delete_by_ids"
        )

    def delete_all(self) -> int:
        """Delete every point and keep the collection; return the count.

        A filter with no conditions matches every point: the server checks
        ``must``, ``should`` and ``must_not`` only when they are set.
        """
        return self._delete_matching(models.Filter(), "delete_all")

    def close(self) -> None:
        """Close the client when this store created it.

        A client passed in through ``client`` belongs to the caller and stays
        open. The store reconnects on its next call either way.
        """
        with self._client_lock:
            if not self._owns_client or self._client is None:
                return
            client, self._client = self._client, None
        try:
            client.close()
        except Exception:
            logger.debug("QdrantVDB.close: client close failed", exc_info=True)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
