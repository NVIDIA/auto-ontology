# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Postgres + pgvector implementation of the NV-Ingest ``VDB`` operator.

Backed by :class:`langchain_postgres.PGVectorStore` (the v2 vector store API).
Records are carried as :class:`langchain_core.documents.Document` objects
throughout. Per-database identification is stored in a real ``database_name``
column so it can be used as a delete/search filter.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, Iterable, Optional

import psycopg
from psycopg import sql
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_postgres import Column, PGEngine, PGVectorStore

from nemo_retriever.common.vdb.adt_vdb import VDB

logger = logging.getLogger(__name__)

_DATABASE_METADATA_COLUMN = "database_name"
_LABEL_METADATA_COLUMN = "label"


class _UnusableEmbeddings(Embeddings):
    """Placeholder used when no query-side embedder is supplied.

    Ingestion via :meth:`PGVectorStore.add_embeddings` does not call this — it
    only fires if someone tries to run :meth:`PostgresVDB.retrieval` without
    passing an ``embeddings`` instance to the constructor.
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError(
            "PostgresVDB has no embeddings function configured. "
            "Pass `embeddings=<Embeddings>` to the constructor to enable retrieval."
        )

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def _flatten(records: Iterable) -> Iterable[dict]:
    """Yield record dicts from possibly-nested NV-Ingest output."""
    for item in records:
        if isinstance(item, dict):
            yield item
        elif isinstance(item, list):
            yield from _flatten(item)


def _to_async_url(url: str) -> str:
    """Convert a libpq-style URL to an async-SQLAlchemy URL for ``PGEngine``."""
    if "+asyncpg" in url or "+psycopg" in url:
        return url
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+asyncpg://", 1)
    return url


class PostgresVDB(VDB):
    """Concrete :class:`VDB` backed by Postgres + pgvector via LangChain v2.

    Each NV-Ingest record becomes a :class:`Document` whose ``page_content`` is
    the searchable text and whose ``metadata`` carries the original record
    metadata plus ``document_type``. Embeddings produced upstream by the NIM
    pipeline are bulk-loaded into PGVectorStore via ``add_embeddings`` so we
    don't re-run the embedder on the write path.

    The metadata field ``database_name`` is promoted to a real column on the
    underlying table so it can be used as a filter for bulk deletes and
    similarity searches.
    """

    # Tells upstream ``search_semantic_index`` to build per-query metadata
    # filters as a flat ``{column: value | [values]}`` mapping instead of a
    # SQL ``LIKE`` predicate over the JSON metadata column. The dict is fed
    # straight into ``PGVectorStore.similarity_search_with_score_by_vector``.
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

        self._engine: Optional[PGEngine] = None
        self._store: Optional[PGVectorStore] = None
        self.vector_size: Optional[int] = kwargs.get("vector_size", 2048)
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
    # Engine / store lifecycle
    # ------------------------------------------------------------------

    def _get_engine(self) -> PGEngine:
        if self._engine is None:
            self._engine = PGEngine.from_connection_string(
                _to_async_url(self.connection_string)
            )
        return self._engine

    def _table_exists(self) -> bool:
        with psycopg.connect(self.connection_string) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT 1
                    FROM information_schema.tables
                    WHERE table_schema = %s AND table_name = %s
                    """,
                    (self.schema_name, self.collection_name),
                )
                return cur.fetchone() is not None

    def _ensure_schema(self) -> None:
        """Create the target schema if it doesn't exist.

        ``init_vectorstore_table`` creates the table (and the pgvector
        extension) but not its containing schema, so it must exist first.
        """
        with psycopg.connect(self.connection_string) as conn:
            with conn.cursor() as cur:
                # `schema_name` is internal config (defaults to 'public');
                # psycopg can't parameterise identifiers, so it's interpolated
                # via the identifier-safe quote. Not user input.
                cur.execute(
                    sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(
                        sql.Identifier(self.schema_name)
                    )
                )
            conn.commit()

    def _get_store(self) -> Optional[PGVectorStore]:
        """Return the vector store, creating the table on first write.

        If the table doesn't exist and ``vector_size`` is not provided (e.g.
        on read paths), returns ``None`` so callers can short-circuit.
        """
        if self._store is not None:
            return self._store

        engine = self._get_engine()
        if not self._table_exists():
            self._ensure_schema()
            engine.init_vectorstore_table(
                table_name=self.collection_name,
                vector_size=self.vector_size,
                schema_name=self.schema_name,
                metadata_columns=[
                    Column(_DATABASE_METADATA_COLUMN, "VARCHAR(100)", nullable=True),
                    Column(_LABEL_METADATA_COLUMN, "VARCHAR(100)", nullable=True),
                ],
            )
            self._create_vector_index()

        self._store = PGVectorStore.create_sync(
            engine=engine,
            embedding_service=self.embeddings,
            table_name=self.collection_name,
            schema_name=self.schema_name,
            metadata_columns=[_DATABASE_METADATA_COLUMN, _LABEL_METADATA_COLUMN],
        )
        return self._store

    def _create_vector_index(self) -> None:
        """Convert the embedding column to ``halfvec`` and index it with HNSW.

        Without an index every similarity search is a sequential scan over the
        whole collection -- measured at 175ms against 0.36ms indexed on 51,200
        rows, and it degrades linearly from there.

        **The column has to become ``halfvec`` for the index to be reachable**,
        which is not obvious and is worth the ALTER. pgvector caps HNSW at 2000
        dimensions for ``vector`` and the embedding model emits 2048, so the
        only index that can be built over a ``vector(2048)`` column is one over
        a cast expression -- and an expression index is matched *syntactically*,
        so it serves ``embedding::halfvec(2048) <=> $1`` and not the plain
        ``embedding <=> $1`` that ``PGVectorStore`` generates. Postgres then
        silently plans a sequential scan and nothing reports it. Typing the
        column ``halfvec`` instead lets a plain index serve the query the store
        actually writes: the *parameter* is cast (``embedding <=> ($1)::halfvec``),
        which costs nothing, rather than the indexed column.

        The tradeoff is that stored vectors are float16 rather than float32.
        Cosine ranking over 2048 dimensions tolerates that -- it is the same
        approximation the index itself would apply -- but it is a real precision
        change, not just an index.

        Best-effort: a failure here costs latency, not correctness, so it is
        logged and the store still works.
        """
        qualified = sql.Identifier(self.schema_name, self.collection_name)
        index_name = sql.Identifier(f"ix_hnsw_{self.collection_name}_embedding")
        try:
            with psycopg.connect(self.connection_string) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        sql.SQL(
                            "ALTER TABLE {table} "
                            "ALTER COLUMN embedding TYPE halfvec({dim})"
                        ).format(
                            table=qualified,
                            dim=sql.Literal(self.vector_size),
                        )
                    )
                    cur.execute(
                        sql.SQL(
                            "CREATE INDEX IF NOT EXISTS {name} ON {table} "
                            "USING hnsw (embedding halfvec_cosine_ops)"
                        ).format(name=index_name, table=qualified)
                    )
                conn.commit()
        except Exception:
            logger.warning(
                "PostgresVDB: could not build the HNSW index on %s.%s; "
                "similarity search will fall back to a sequential scan",
                self.schema_name,
                self.collection_name,
                exc_info=True,
            )

    def create_index(self, **kwargs: Any) -> str:
        """Ensure the pgvector extension and underlying table exist."""
        self._get_store()
        return self.collection_name

    # ------------------------------------------------------------------
    # Read / write
    # ------------------------------------------------------------------

    def write_to_index(
        self,
        records: list,
        batch_size: int = 500,
        **kwargs: Any,
    ) -> int:
        """Bulk-insert NV-Ingest records, returning the number of rows written."""
        documents: list[Document] = []
        embeddings: list[list[float]] = []
        skipped = 0

        for record in _flatten(records):
            metadata = record.get("metadata") or {}
            embedding = metadata.get("embedding")
            text = (
                record.get("text") or record.get("content") or metadata.get("content")
            )
            if not embedding:
                skipped += 1
                continue

            doc_metadata = {
                "document_type": record.get("document_type"),
                **{k: v for k, v in metadata.items() if k != "embedding"},
            }
            documents.append(Document(page_content=text, metadata=doc_metadata))
            embeddings.append([float(v) for v in embedding])

        if not documents:
            logger.info(
                "PostgresVDB.write_to_index: no rows to insert (skipped %d)", skipped
            )
            return 0

        store = self._get_store()
        assert store is not None  # vector_size was provided

        inserted = 0
        for start in range(0, len(documents), batch_size):
            chunk_docs = documents[start : start + batch_size]
            chunk_embs = embeddings[start : start + batch_size]
            store.add_embeddings(
                texts=[d.page_content for d in chunk_docs],
                embeddings=chunk_embs,
                metadatas=[d.metadata for d in chunk_docs],
            )
            inserted += len(chunk_docs)

        logger.info(
            "PostgresVDB.write_to_index: inserted %d rows into %s (skipped %d)",
            inserted,
            self.collection_name,
            skipped,
        )
        return inserted

    def delete_by_database(self, database_name: str) -> list[str]:
        """Delete all rows whose ``database_name`` column matches ``database_name``.

        Returns the list of deleted row IDs (empty if the table doesn't exist
        or no rows match).
        """
        store = self._get_store()
        if store is None:
            logger.info(
                "PostgresVDB.delete_by_database: collection %s not found, "
                "nothing to delete",
                self.collection_name,
            )
            return []

        filter: dict[str, Any] = {
            _DATABASE_METADATA_COLUMN: database_name,
        }

        existing = store.get(
            where=filter,
            include=[],
        )
        ids = list(existing.get("ids", []) or [])
        if not ids:
            return []

        store.delete(filter=filter)
        return ids

    def delete_all(self) -> int:
        """Delete every row in the collection.

        Returns the number of rows deleted (``0`` when the collection table
        doesn't exist yet).
        """
        if not self._table_exists():
            logger.info(
                "PostgresVDB.delete_all: collection %s not found, nothing to delete",
                self.collection_name,
            )
            return 0

        with psycopg.connect(self.connection_string) as conn:
            with conn.cursor() as cur:
                # `schema_name`/`collection_name` are internal config; psycopg
                # can't parameterise identifiers, so they're composed via the
                # identifier-safe API. Not user input.
                cur.execute(
                    sql.SQL("DELETE FROM {table}").format(
                        table=sql.Identifier(self.schema_name, self.collection_name)
                    )
                )
                deleted = cur.rowcount

        logger.info(
            "PostgresVDB.delete_all: deleted %d rows from %s",
            deleted,
            self.collection_name,
        )
        return deleted

    def delete_by_id(self, node_id: str) -> int:
        """Delete every row whose metadata ``id`` matches ``node_id``.

        ``id`` lives in the JSONB ``langchain_metadata`` column (it is not
        a promoted real column — only ``database_name`` and ``label`` are),
        so the match goes through ``langchain_metadata ->> 'id'`` rather
        than the typed-filter path used by :meth:`delete_by_database`.
        Going through ``PGVectorStore.delete(filter=...)`` would be a
        no-op for this case because the dict-format filter only resolves
        declared metadata columns.

        Returns the number of rows deleted (``0`` when the collection
        table doesn't exist yet or nothing matched).
        """
        if not self._table_exists():
            logger.info(
                "PostgresVDB.delete_by_id: collection %s not found, nothing to delete",
                self.collection_name,
            )
            return 0

        with psycopg.connect(self.connection_string) as conn:
            with conn.cursor() as cur:
                # `schema_name`/`collection_name` are internal config; psycopg
                # can't parameterise identifiers, so they're composed via the
                # identifier-safe API. Not user input.
                cur.execute(
                    sql.SQL(
                        """
                        DELETE FROM {table}
                        WHERE langchain_metadata ->> 'id' = %s
                        """
                    ).format(
                        table=sql.Identifier(self.schema_name, self.collection_name)
                    ),
                    (node_id,),
                )
                deleted = cur.rowcount

        logger.info(
            "PostgresVDB.delete_by_id: deleted %d rows from %s for id=%s",
            deleted,
            self.collection_name,
            node_id,
        )
        return deleted

    def retrieval(
        self,
        queries: list,
        top_k: int = 10,
        **kwargs: Any,
    ) -> list[list[dict]]:
        """Cosine-similarity k-NN search for each query vector.

        ``kwargs["where"]`` is a langchain-postgres filter dict
        (e.g. ``{"label": "CustomAnalysis", "database_name": "prod"}``)
        applied to the declared metadata columns; passed straight through to
        ``PGVectorStore.similarity_search_with_score_by_vector``.

        Requires an ``embeddings`` instance from the constructor.
        """
        store = self._get_store()
        if store is None:
            return [[] for _ in queries]

        results: list[list[dict]] = []
        for query in queries:
            try:
                hits = store.similarity_search_with_score_by_vector(
                    embedding=query,
                    filter=kwargs.get("where", None),
                    k=top_k,
                )
                results.append(
                    [
                        {
                            "text": doc.page_content,
                            "metadata": doc.metadata,
                            "_distance": float(score),
                        }
                        for doc, score in hits
                    ]
                )
            except Exception as e:
                logger.error(f"Error in retrieval: {e}")
                return [[] for _ in queries]
        return results

    def run(self, records: list) -> int:
        """Create the collection if needed, then write records to it."""
        self.create_index()
        return self.write_to_index(records)

    #: Seconds to wait for the pool to dispose before giving up on it.
    #:
    #: Disposal is a courtesy: the pool's sockets are closed by the OS when the
    #: process exits either way. Waiting longer than this buys nothing and risks
    #: the hang described on :meth:`close`.
    _DISPOSE_TIMEOUT = 5.0

    def close(self) -> None:
        """Dispose the engine's connection pool, without ever blocking forever.

        Three paths, because disposal has to cross a thread boundary and each
        way of doing that can wedge:

        * **On the engine's own loop thread** -- scheduled, never waited on, or
          it would join itself.
        * **Off it, with the loop alive** -- submitted and waited on with a
          timeout.
        * **Off it, with the loop gone** -- skipped entirely.

        That last case is the one that matters, and it is why this does not use
        ``engine._run_as_sync``: that helper waits on ``future.result()`` with
        no timeout, and at interpreter shutdown the loop thread is already gone,
        so the future is never resolved by anyone. Reached through
        :meth:`__del__`, it hung ``python -m gsf.semantic`` for 45+ minutes
        after the work had finished and committed, and the ``try/except`` around
        it cannot catch a hang.
        """
        self._store = None
        engine = self._engine
        if engine is None:
            return
        self._engine = None

        loop = getattr(engine, "_loop", None)
        thread = getattr(engine, "_thread", None)

        on_loop_thread = thread is not None and thread is threading.current_thread()
        if not on_loop_thread:
            try:
                running = asyncio.get_running_loop()
            except RuntimeError:
                running = None
            on_loop_thread = running is not None and running is loop

        if on_loop_thread:
            # Can't block on the loop from within it; schedule dispose
            # non-blocking so it runs once the loop is free again.
            if loop is not None and loop.is_running():
                try:
                    asyncio.run_coroutine_threadsafe(engine._pool.dispose(), loop)
                except Exception:
                    pass
            return

        # Nobody is left to run the coroutine: submitting it would wait on a
        # future that can never resolve.
        if loop is None or not loop.is_running():
            return
        if thread is not None and not thread.is_alive():
            return

        try:
            future = asyncio.run_coroutine_threadsafe(engine._pool.dispose(), loop)
            future.result(timeout=self._DISPOSE_TIMEOUT)
        except Exception:
            # Includes the timeout: the pool outliving the process by a few
            # seconds is not worth blocking a shutdown over.
            logger.debug("PostgresVDB.close: pool dispose did not complete")

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

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
