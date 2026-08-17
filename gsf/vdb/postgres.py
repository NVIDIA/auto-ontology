# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Postgres + pgvector implementation of the NV-Ingest ``VDB`` operator.

The vectors live on the entity rows themselves -- ``catalog_table.embedding``,
``term.embedding`` and so on -- rather than in a separate collection. All of the
SQL for that is in :mod:`gsf.vdb.entity_store`; this class is the adapter that
keeps the ``VDB`` interface the library and every GSF call site expect.

It is deliberately a thin shim. ``Retriever``, ``search_semantic_index`` and the
service layer all speak in terms of collections, labels and metadata filters, so
the translation from "collection" to "the set of tables owning these labels"
happens here, once.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from langchain_core.embeddings import Embeddings

from nemo_retriever.common.vdb.adt_vdb import VDB

from gsf.vdb import entity_store


def _requested_labels(where: dict, available: tuple[str, ...]) -> tuple[str, ...]:
    """Labels named by a filter, narrowed to those this collection owns.

    The filter's ``label`` is either absent, one string, or a list. An unknown
    label yields an empty tuple, which is the honest answer -- searching every
    table instead would silently return matches the caller filtered out.
    """
    requested = where.get("label")
    if requested is None:
        return available
    wanted = {requested} if isinstance(requested, str) else set(requested)
    return tuple(label for label in available if label in wanted)


logger = logging.getLogger(__name__)

_DATABASE_METADATA_COLUMN = "database_name"
_LABEL_METADATA_COLUMN = "label"


class _UnusableEmbeddings(Embeddings):
    """Placeholder used when no query-side embedder is supplied.

    Nothing on GSF's paths calls it: ``retrieval`` receives query vectors that
    were embedded upstream, and the write path receives embeddings alongside the
    records. It exists so that a caller who *does* expect the VDB to embed for
    them gets a clear error instead of an AttributeError.
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError(
            "PostgresVDB has no embeddings function configured. "
            "Pass `embeddings=<Embeddings>` to the constructor to enable retrieval."
        )

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class PostgresVDB(VDB):
    """Concrete :class:`VDB` backed by Postgres + pgvector via LangChain v2.

    Each NV-Ingest record becomes a :class:`Document` whose ``page_content`` is
    the searchable text and whose ``metadata`` carries the original record
    metadata plus ``document_type``. Embeddings produced upstream by the NIM
    pipeline are written straight onto the entity rows, so the embedder is
    never re-run on the write path.

    The metadata field ``database_name`` is promoted to a real column on the
    underlying table so it can be used as a filter for bulk deletes and
    similarity searches.
    """

    # Tells upstream ``search_semantic_index`` to build per-query metadata
    # filters as a flat ``{column: value | [values]}`` mapping instead of a
    # SQL ``LIKE`` predicate over the JSON metadata column. The dict is fed
    # resolved against the entity tables by ``entity_store.search``.
    metadata_filter_format = "dict"

    def __init__(self, *, reset: bool = False, **kwargs: Any) -> None:
        connection_string = kwargs.get("connection_string")
        if not connection_string:
            raise ValueError(
                "PostgresVDB requires a 'connection_string' kwarg "
                "(e.g. postgresql://user:pass@host:5432/dbname)."
            )
        self.connection_string: str = connection_string

        self.collection_name: str = kwargs.get(
            "collection_name", kwargs.get("index_name", "data_objects_layer")
        )
        self.schema_name: str = kwargs.get("schema_name", "public")
        self.embeddings: Embeddings = kwargs.get("embeddings") or _UnusableEmbeddings()

        # Accepted and ignored: the column width is fixed by the migration, so
        # a caller asking for a different one would be silently wrong. Kept in
        # the signature because the ingestion operator passes it.
        self.vector_size: Optional[int] = kwargs.get(
            "vector_size", entity_store.EMBEDDING_DIMENSIONS
        )
        # Resetting the database embeddings prior to ingestion
        # In order to support without recreate:
        # 1. The ingestion should return which tables/columns were added/updated/deleted
        # 2. The implemtation should support be fault tolerant and support incremental ingestion, which is challenging.
        self.database_name = kwargs.get("database_name")
        if self.database_name and reset:
            ids = self.delete_by_database(self.database_name)
            logger.info(
                "PostgresVDB.delete_by_database: deleted %d rows for database %s",
                len(ids),
                self.database_name,
            )

        super().__init__(**kwargs)

    # ------------------------------------------------------------------
    # Labels this instance is responsible for
    # ------------------------------------------------------------------

    @property
    def labels(self) -> tuple[str, ...]:
        """The labels this collection name stands for.

        ``data_objects_layer`` means Table and Column; ``semantic_layer`` means
        the five semantic kinds. Callers still think in collections, so the
        translation lives here rather than leaking into every service.
        """
        return entity_store.labels_for_collection(self.collection_name)

    # ------------------------------------------------------------------
    # Write / read
    # ------------------------------------------------------------------

    def create_index(self, **kwargs: Any) -> str:
        """No-op: the tables and their vector columns are owned by Alembic.

        Kept because ``run()`` and the ingestion operator call it. Creating
        storage from the write path is exactly what this migration removed --
        the schema is a migration's job, so that a fresh database has one
        definition of its shape rather than two.
        """
        return self.collection_name

    def write_to_index(
        self,
        records: list,
        batch_size: int = 500,
        **kwargs: Any,
    ) -> int:
        """Attach embeddings to their entity rows. Returns rows updated.

        ``batch_size`` is accepted for interface compatibility and ignored: the
        write is one statement per label, so there is nothing to chunk.
        """
        written = entity_store.write_embeddings(records, allowed_labels=self.labels)
        logger.info(
            "PostgresVDB.write_to_index: %d row(s) embedded for %s",
            written,
            self.collection_name,
        )
        return written

    def delete_by_database(self, database_name: str) -> list[str]:
        """Clear every embedding belonging to *database_name*."""
        return entity_store.clear_by_database(database_name, labels=self.labels)

    def delete_all(self) -> int:
        """Clear every embedding this collection covers."""
        deleted = entity_store.clear_all(labels=self.labels)
        logger.info(
            "PostgresVDB.delete_all: cleared %d embedding(s) from %s",
            deleted,
            self.collection_name,
        )
        return deleted

    def delete_by_id(self, node_id: str) -> int:
        """Clear one row's embedding.

        The row itself is left alone: this API only ever meant "remove this from
        the index". Deleting the entity is the DAL's job, and now takes the
        embedding with it automatically.
        """
        return entity_store.clear_by_id(node_id, labels=self.labels)

    def retrieval(
        self,
        queries: list,
        top_k: int = 10,
        **kwargs: Any,
    ) -> list[list[dict]]:
        """Cosine-similarity k-NN search for each query vector.

        ``kwargs["where"]`` is the flat mapping ``search_semantic_index`` builds
        (see ``metadata_filter_format``), e.g.
        ``{"label": "CustomAnalysis", "database_name": "prod"}``. ``label``
        selects which tables to search and ``database_name`` becomes a column
        predicate; both may be absent, in which case every label this collection
        owns is searched.
        """
        where = kwargs.get("where") or {}
        labels = _requested_labels(where, self.labels)
        database_name = where.get("database_name") or self.database_name
        # Anything the filter names beyond label/database_name is matched
        # against the stored metadata -- `is_unique`, for instance, which
        # semantic FK inference uses to keep candidates to unique columns.
        extra = {
            key: value
            for key, value in where.items()
            if key not in {"label", "database_name"}
        }

        return [
            entity_store.search(
                query,
                labels=labels,
                top_k=top_k,
                database_name=database_name,
                extra_filters=extra,
            )
            for query in queries
        ]

    def run(self, records: list) -> int:
        """Create the collection if needed, then write records to it."""
        self.create_index()
        return self.write_to_index(records)

    def close(self) -> None:
        """No-op: connections come from the shared DAL pool, not a private one.

        Retained because ``gsf.utils.retriever`` closes retrievers on shutdown.
        The pool's lifecycle belongs to ``gsf.dal.session.dispose_engine``.
        """
        return None

    # ------------------------------------------------------------------
    # Collection management — required by the ABC, unused by GSF
    # ------------------------------------------------------------------
    #
    # ``VDB`` gained a collection-management API (scopes, collections,
    # per-document CRUD) that models a multi-tenant document store. GSF has one
    # collection per tier, created by ``_ensure_schema``, and reaches the store
    # through ``run`` / ``write_to_index`` / ``retrieval`` and the three bulk
    # deletes above — none of these are on any path it takes.
    #
    # They are declared abstract, so leaving them out makes ``PostgresVDB``
    # itself abstract and every ``get_data_vdb()`` raise ``TypeError`` at
    # construction. They raise rather than returning a plausible empty value:
    # if GSF ever grows a caller, it should fail here and get a real
    # implementation, not silently read an empty collection.

    def _unsupported(self, method: str) -> NotImplementedError:
        return NotImplementedError(
            f"PostgresVDB does not implement {method}(): GSF uses a single "
            f"collection per tier and never calls the collection-management "
            f"API. Implement it here if that changes."
        )

    def create_collection(self, *, scope, request):
        raise self._unsupported("create_collection")

    def get_collection(self, *, scope, collection_name):
        raise self._unsupported("get_collection")

    def update_collection(self, *, scope, collection_name, request):
        raise self._unsupported("update_collection")

    def delete_collection(self, *, scope, collection_name, if_exists):
        raise self._unsupported("delete_collection")

    def list_collections(self, *, scope, limit, continuation_token):
        raise self._unsupported("list_collections")

    def retrieve_collection(
        self, vectors, *, scope, collection_name, query_texts, top_k, **kwargs
    ):
        raise self._unsupported("retrieve_collection")

    def write_collection(self, records, *, context):
        # Reached only when a caller passes `collection_context` to
        # IngestVdbOperator; GSF never does, so ingestion takes `run()`.
        raise self._unsupported("write_collection")

    def get_document(self, *, scope, collection_name, document_id):
        raise self._unsupported("get_document")

    def list_documents(self, *, scope, collection_name, limit, continuation_token):
        raise self._unsupported("list_documents")

    def delete_document(self, *, scope, collection_name, document_id, if_exists):
        raise self._unsupported("delete_document")
