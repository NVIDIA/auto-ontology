# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The PGVectorStore holding unstructured chunk embeddings.

Ingest writes chunk vectors here and retrieve reads them back, so both go
through :func:`get_chunk_store` — the table name, schema and promoted metadata
columns have to agree exactly or the read side silently finds nothing.

Rows are keyed by ``unstructured_chunk.id``: ingest passes the same UUID as
``ids=`` when adding embeddings, which is what lets a vector hit be resolved
back to its relational chunk row.

Documents are embedded upstream by ``embed_client`` and written here with
``add_embeddings``; the only embedding this module performs is of the query at
search time.
"""

from __future__ import annotations

import logging

import psycopg
from langchain_core.embeddings import Embeddings
from langchain_postgres import Column, PGEngine, PGVectorStore
from psycopg import sql
from sqlalchemy import Connection, text

import unstructured_ao.config as config

logger = logging.getLogger(__name__)

#: ``document_id`` is promoted to a real column rather than left in the JSON
#: metadata blob because langchain-postgres only resolves search filters
#: against declared ``metadata_columns`` — the per-document filter in
#: ``retrieve.link_document_attributes`` cannot work without it.
DOCUMENT_ID_COLUMN = "document_id"


def _to_async_url(url: str) -> str:
    """Convert a libpq-style URL to the async-SQLAlchemy URL ``PGEngine`` needs."""
    if "+asyncpg" in url or "+psycopg" in url:
        return url
    for prefix in ("postgresql://", "postgres://"):
        if url.startswith(prefix):
            return f"postgresql+asyncpg://{url[len(prefix) :]}"
    return url


def build_embeddings() -> Embeddings:
    """Return the query-side embedding client for the ``EMBED_*`` triplet.

    Queries go through langchain because that is the interface
    ``PGVectorStore.similarity_search_with_score(query=...)`` requires.  It must
    stay on the same ``EMBED_*`` triplet as the ingest-side pipeline: a
    different model puts query and document vectors in different spaces.
    """
    from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

    from auto_ontology.utils.model_config import resolve

    return NVIDIAEmbeddings(
        model=resolve("EMBED", "MODEL"),
        api_key=resolve("EMBED", "API_KEY"),
        base_url=resolve("EMBED", "ENDPOINT"),
    )


def _table_exists(connection_string: str, schema: str, table: str) -> bool:
    with psycopg.connect(connection_string) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT 1
                FROM information_schema.tables
                WHERE table_schema = %s AND table_name = %s
                """,
                (schema, table),
            )
            return cur.fetchone() is not None


def _create_table(
    engine: PGEngine,
    connection_string: str,
    vector_size: int,
    schema: str,
    table: str,
    metadata_columns: list[str],
) -> None:
    """Create the schema and the vector table at *vector_size* columns."""
    with psycopg.connect(connection_string) as conn:
        with conn.cursor() as cur:
            # init_vectorstore_table creates the table but not its schema, and
            # psycopg cannot parameterise an identifier. Not user input.
            cur.execute(
                sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema))
            )
        conn.commit()

    logger.info(
        "Creating vector table %s.%s with vector_size=%d.",
        schema,
        table,
        vector_size,
    )
    engine.init_vectorstore_table(
        table_name=table,
        vector_size=vector_size,
        schema_name=schema,
        metadata_columns=[Column(c, "TEXT", nullable=True) for c in metadata_columns],
    )


#: Promoted on the attribute table for the same reason ``document_id`` is on
#: the chunk table: langchain-postgres only filters on declared columns, so a
#: search that wants "measures, in this report" needs both as real columns.
ATTRIBUTE_METADATA_COLUMNS = [DOCUMENT_ID_COLUMN, "term_id", "attribute_type"]

_store: PGVectorStore | None = None
_attribute_store: PGVectorStore | None = None


def delete_document_vectors(conn: Connection, document_id: str) -> None:
    """Delete *document_id*'s chunk and attribute vectors on *conn*.

    The vector tables sit outside the relational schema, so the cascade from
    ``unstructured_document`` never reaches them.  Deleting through the
    caller's connection keeps them in the same transaction as the rows they
    mirror.  Either table may not exist yet: both are created on first write.
    """
    _delete_vectors(conn, config.CHUNK_SCHEMA, config.CHUNK_TABLE, document_id)
    delete_document_attribute_vectors(conn, document_id)


def delete_document_attribute_vectors(conn: Connection, document_id: str) -> None:
    """Delete only *document_id*'s attribute vectors on *conn*, keeping its chunks'.

    The attribute store writes on its own connection, so vectors from a semantic
    pass whose transaction rolled back outlive the attributes they point at.
    """
    _delete_vectors(conn, config.ATTRIBUTE_SCHEMA, config.ATTRIBUTE_TABLE, document_id)


def _delete_vectors(
    conn: Connection, schema: str, table: str, document_id: str
) -> None:
    quote = conn.dialect.identifier_preparer.quote
    qualified = f"{quote(schema)}.{quote(table)}"
    if conn.execute(
        text("SELECT to_regclass(:qualified)"), {"qualified": qualified}
    ).scalar():
        conn.execute(
            text(
                f"DELETE FROM {qualified} "
                f"WHERE {quote(DOCUMENT_ID_COLUMN)} = :document_id"
            ),
            {"document_id": document_id},
        )


def get_attribute_store(*, vector_size: int | None = None) -> PGVectorStore:
    """Return the attribute vector store, creating its table on first write.

    Rows are keyed by ``unstructured_attribute.id``, the same way chunk vectors
    are keyed by ``unstructured_chunk.id``, so a hit resolves back to its
    relational row — and from there to its term, its document and, through
    ``unstructured_attribute__chunk``, the chunks that support it.

    Unlike the chunk store this is searched across every document: an attribute
    search asks which concepts in the corpus match, not which passages of one
    file do.
    """
    global _attribute_store
    if _attribute_store is not None:
        return _attribute_store

    from auto_ontology.infra.postgres import get_postgres_connection_string

    connection_string = get_postgres_connection_string()
    engine = PGEngine.from_connection_string(_to_async_url(connection_string))

    if not _table_exists(
        connection_string, config.ATTRIBUTE_SCHEMA, config.ATTRIBUTE_TABLE
    ):
        if vector_size is None:
            raise RuntimeError(
                f"Vector table {config.ATTRIBUTE_SCHEMA}.{config.ATTRIBUTE_TABLE} "
                f"does not exist — run the semantic stage before querying."
            )
        _create_table(
            engine,
            connection_string,
            vector_size,
            config.ATTRIBUTE_SCHEMA,
            config.ATTRIBUTE_TABLE,
            ATTRIBUTE_METADATA_COLUMNS,
        )

    _attribute_store = PGVectorStore.create_sync(
        engine=engine,
        embedding_service=build_embeddings(),
        table_name=config.ATTRIBUTE_TABLE,
        schema_name=config.ATTRIBUTE_SCHEMA,
        metadata_columns=ATTRIBUTE_METADATA_COLUMNS,
    )
    return _attribute_store


def get_chunk_store(*, vector_size: int | None = None) -> PGVectorStore:
    """Return the chunk vector store, creating its table on first write.

    *vector_size* is the width of the vectors about to be written, taken from
    the embeddings the ingest pipeline produced.  It fixes the column width at
    creation, so it is required the first time and ignored afterwards.  Read
    callers pass nothing and get an error if nothing has been ingested yet,
    which is a clearer failure than an empty result set.

    Cached per process: each store owns an async connection pool, so building
    one per call would leak pools.
    """
    global _store
    if _store is not None:
        return _store

    from auto_ontology.infra.postgres import get_postgres_connection_string

    connection_string = get_postgres_connection_string()
    engine = PGEngine.from_connection_string(_to_async_url(connection_string))

    if not _table_exists(connection_string, config.CHUNK_SCHEMA, config.CHUNK_TABLE):
        if vector_size is None:
            raise RuntimeError(
                f"Vector table {config.CHUNK_SCHEMA}.{config.CHUNK_TABLE} does "
                f"not exist — ingest at least one document before querying."
            )
        _create_table(
            engine,
            connection_string,
            vector_size,
            config.CHUNK_SCHEMA,
            config.CHUNK_TABLE,
            [DOCUMENT_ID_COLUMN],
        )

    # Only ever used to embed queries: documents arrive pre-embedded from
    # whichever extractor ran and go in through ``add_embeddings``.
    _store = PGVectorStore.create_sync(
        engine=engine,
        embedding_service=build_embeddings(),
        table_name=config.CHUNK_TABLE,
        schema_name=config.CHUNK_SCHEMA,
        metadata_columns=[DOCUMENT_ID_COLUMN],
    )
    return _store
