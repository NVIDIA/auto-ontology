# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ingest pipeline for unstructured documents.

Pipeline per file
-----------------
1. NeMo Retriever ``create_ingestor`` extracts text / tables / charts /
   infographics as chunks.
2. Each chunk is written to ``unstructured_chunk`` (relational) with a
   pre-generated UUID that is also used as the langchain_pg_embedding row key.
3. Chunk text is embedded in batch via ``batch_embed_chunks`` and uploaded to
   the ``CHUNK_COLLECTION`` PGVectorStore so attribute → chunk vector search
   works in the retrieve step.
4. ``extract.run_extraction`` calls the LLM to extract one term and its
   attributes, writing to ``unstructured_term`` and ``unstructured_attribute``.

Usage
-----
    from unstructured_ao import config  # must be imported first (Ray override)
    from unstructured_ao.ingest import ingest_directory

    with engine.begin() as conn:
        for doc_id in ingest_directory(config.DATA_DIR, conn):
            print(f"Ingested document {doc_id}")
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

from sqlalchemy import Connection

import unstructured_ao.config as config  # noqa: F401  — Ray override must run first
from unstructured_ao.dal import (
    get_document_by_path,
    insert_chunk,
    insert_document,
)

logger = logging.getLogger(__name__)

#: NeMo Retriever ingestor content types → our canonical ``content_type`` values.
_CONTENT_TYPE_MAP: dict[str, str] = {
    "text": "text",
    "structured": "table",
    "chart": "chart",
    "infographic": "infographic",
    "image": "infographic",
}


def _build_pgvector_store():  # type: ignore[return]
    """Return a PGVectorStore backed by the auto-ontology Postgres instance.

    The collection ``config.CHUNK_COLLECTION`` is created on first use.
    """
    from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
    from langchain_postgres import PGVectorStore

    from auto_ontology.infra.postgres import get_postgres_connection_string
    from auto_ontology.utils.model_config import resolve

    embed_model = resolve("EMBED", "MODEL")
    embed_api_key = resolve("EMBED", "API_KEY")
    embed_endpoint = resolve("EMBED", "ENDPOINT")

    embeddings = NVIDIAEmbeddings(
        model=embed_model,
        api_key=embed_api_key,
        base_url=embed_endpoint,
    )
    return PGVectorStore.from_params(
        embedding=embeddings,
        host=None,
        port=None,
        database=None,
        user=None,
        password=None,
        collection_name=config.CHUNK_COLLECTION,
        connection=get_postgres_connection_string(),
    )


def _extract_chunks_from_file(file_path: str) -> list[dict[str, Any]]:
    """Run NeMo Retriever extraction on *file_path* and return raw chunks.

    Each returned dict has at least:
    - ``text`` (str): the chunk text content
    - ``metadata`` (dict): NeMo Retriever metadata (page_number, content_metadata, …)

    The pipeline is configured for text/table/chart/infographic extraction only
    (no embed, no VDB upload — those happen separately so we control IDs).
    """
    from nemo_retriever import create_ingestor

    ingestor = (
        create_ingestor(run_mode="batch")
        .files([file_path])
        .extract(
            extract_text=True,
            extract_tables=True,
            extract_charts=True,
            extract_infographics=True,
        )
    )
    return ingestor.ingest()


def _chunk_text(chunks: list[dict[str, Any]]) -> str:
    """Concatenate all text chunks into a single string for LLM extraction."""
    return "\n\n".join(
        c.get("text", "") or ""
        for c in chunks
        if (c.get("metadata", {}) or {}).get("content_metadata", {}).get("type")
        == "text"
        or not (c.get("metadata", {}) or {}).get("content_metadata")
    ).strip()


def _resolve_content_type(chunk: dict[str, Any]) -> str:
    raw = (chunk.get("metadata") or {}).get("content_metadata", {}).get("type", "text")
    return _CONTENT_TYPE_MAP.get(raw, "text")


def _resolve_page_number(chunk: dict[str, Any]) -> int | None:
    meta = chunk.get("metadata") or {}
    page = meta.get("page_number")
    if page is not None:
        try:
            return int(page)
        except (ValueError, TypeError):
            pass
    return None


def ingest_file(file_path: str | Path, conn: Connection) -> str | None:
    """Ingest a single file into Postgres and the vector store.

    Returns the ``unstructured_document.id`` on success, or ``None`` when the
    file was already ingested (same absolute path already present).

    On re-ingest (``force=True`` is not yet implemented), call
    ``dal.delete_document`` first to cascade-delete all downstream rows before
    calling this function again.
    """
    path = str(Path(file_path).expanduser().resolve())
    name = Path(path).name
    file_type = Path(path).suffix.lstrip(".").lower()

    existing = get_document_by_path(conn, path)
    if existing:
        logger.info("File already ingested: %s (document_id=%s).", path, existing["id"])
        return existing["id"]

    logger.info("Extracting chunks from %s …", name)
    try:
        raw_chunks = _extract_chunks_from_file(path)
    except Exception:
        logger.exception("NeMo Retriever extraction failed for %s — skipping.", path)
        return None

    if not raw_chunks:
        logger.warning("No chunks extracted from %s.", path)
        return None

    # Insert document row first (chunks FK to it).
    document_id = insert_document(
        conn,
        name=name,
        path=path,
        file_type=file_type,
    )

    # Assign UUIDs up front so the same id lands in both the relational table
    # and the vector store.
    chunk_ids: list[str] = [str(uuid.uuid4()) for _ in raw_chunks]

    # --- Store chunks in the relational table ---
    for idx, (chunk, cid) in enumerate(zip(raw_chunks, chunk_ids)):
        insert_chunk(
            conn,
            document_id=document_id,
            content=chunk.get("text", ""),
            content_type=_resolve_content_type(chunk),
            page_number=_resolve_page_number(chunk),
            chunk_index=idx,
            source_metadata=chunk.get("metadata"),
            chunk_id=cid,
        )

    logger.info("Stored %d chunk(s) for %s.", len(raw_chunks), name)

    # --- Embed chunks and upload to PGVectorStore ---
    try:
        from langchain_core.documents import Document

        vdb = _build_pgvector_store()
        docs = [
            Document(
                page_content=c.get("text", ""),
                metadata={
                    "document_id": document_id,
                    "chunk_index": i,
                    "content_type": _resolve_content_type(c),
                    "page_number": _resolve_page_number(c),
                    "source": name,
                },
            )
            for i, c in enumerate(raw_chunks)
        ]
        vdb.add_documents(docs, ids=chunk_ids)
        logger.info(
            "Uploaded %d embedding(s) to collection %r.",
            len(docs),
            config.CHUNK_COLLECTION,
        )
    except Exception:
        logger.exception(
            "Embedding / VDB upload failed for %s — relational rows are intact "
            "but retrieve step will find no vectors.",
            name,
        )

    # --- LLM extraction of term + attributes ---
    from unstructured_ao.extract import run_extraction

    full_text = _chunk_text(raw_chunks)
    run_extraction(document_id, full_text, conn)

    return document_id


def ingest_directory(
    directory: str | Path | None = None,
    conn: Connection | None = None,
) -> list[str]:
    """Ingest all supported files in *directory* (default: ``config.DATA_DIR``).

    Returns the list of document ids successfully ingested.

    *conn* must be an open SQLAlchemy ``Connection`` inside a transaction.
    When ``None`` a new engine + connection is created from env vars.
    """
    from auto_ontology.dal.session import get_engine

    data_dir = Path(directory or config.DATA_DIR)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}. "
            "Copy PDFs into unstructured/data/vidore_v3_hr/ first."
        )

    supported = (".pdf", ".docx", ".pptx", ".txt", ".html")
    files = sorted(p for p in data_dir.rglob("*") if p.suffix.lower() in supported)
    if not files:
        logger.warning("No supported files found in %s.", data_dir)
        return []

    def _run(conn: Connection) -> list[str]:
        doc_ids = []
        for f in files:
            doc_id = ingest_file(f, conn)
            if doc_id:
                doc_ids.append(doc_id)
        return doc_ids

    if conn is not None:
        return _run(conn)

    engine = get_engine()
    with engine.begin() as c:
        return _run(c)


def main() -> None:
    """CLI entry point: ingest all files in the default data directory."""
    import argparse

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    parser = argparse.ArgumentParser(
        description="Ingest unstructured documents into Postgres."
    )
    parser.add_argument(
        "directory", nargs="?", help="Override the default data directory."
    )
    args = parser.parse_args()

    doc_ids = ingest_directory(args.directory)
    print(f"Ingested {len(doc_ids)} document(s): {doc_ids}")


if __name__ == "__main__":
    main()
