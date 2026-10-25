# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ingest pipeline for unstructured documents: files in, chunks and vectors out.

No LLM runs here.  Everything this stage writes follows mechanically from the
file, so it can be re-run over a corpus for the cost of the embeddings alone,
while the extraction pass (``unstructured_ao.extract``) works from the chunks it
leaves in Postgres.

Pipeline per file
-----------------
1. The extractor selected by ``config.EXTRACTOR`` turns the file into chunks of
   text, tables, charts and infographics.  ``auto`` routes PDFs to the hosted
   extractor, office/HTML files to Docling, and TXT files to the plain-text
   reader.
2. Each chunk is written to ``unstructured_chunk`` (relational) with a
   pre-generated UUID that is also used as the vector-store row key.
3. The embeddings are uploaded to the chunk vector store under those same ids,
   so attribute → chunk vector search works in the extraction step.

Usage
-----
    from unstructured_ao.ingest import ingest_directory

    with engine.begin() as conn:
        for doc_id in ingest_directory("/path/to/documents", conn):
            print(f"Ingested document {doc_id}")

Terms and attributes come afterwards, from
``unstructured_ao.extract.run_extraction``.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

from sqlalchemy import Connection, text

import unstructured_ao.config as config
from unstructured_ao.dal import (
    get_document,
    get_document_by_path,
    insert_chunk,
    insert_document,
    list_pending_documents,
    set_document_status,
)

logger = logging.getLogger(__name__)

#: Extractor element types → our canonical ``content_type`` values.
_CONTENT_TYPE_MAP: dict[str, str] = {
    "text": "text",
    "structured": "table",
    "table": "table",
    "chart": "chart",
    "infographic": "infographic",
    "image": "infographic",
}


def _extract_chunks_from_file(file_path: str) -> list[dict[str, Any]]:
    """Extract and embed *file_path* with whichever extractor is configured.

    ``config.EXTRACTOR`` either selects one extractor or lets ``auto`` route by
    extension.  Every extractor returns the same record shape with an embedding
    attached, so nothing downstream of here knows which one ran.
    """
    extractor = _extractor_for_file(file_path)
    if extractor in _LIBRARY_EXTRACTORS:
        return _extract_with_library(file_path, extractor)
    raise ValueError(
        f"Unknown EXTRACTOR {extractor!r}; expected one of "
        f"{', '.join(repr(name) for name in sorted(_LIBRARY_EXTRACTORS))}."
    )


_AUTO_EXTRACTORS: dict[str, str] = {
    ".pdf": "hosted",
    ".docx": "docling",
    ".pptx": "docling",
    ".html": "docling",
    ".txt": "text",
}

_EXTRACTOR_EXTENSIONS: dict[str, frozenset[str]] = {
    "auto": frozenset(_AUTO_EXTRACTORS),
    "hosted": frozenset({".pdf"}),
    "docling": frozenset({".pdf", ".docx", ".pptx", ".html"}),
}


#: The optional package each extractor imports, from the ``unstructured``
#: extra.  The text reader needs none.
_EXTRACTOR_REQUIREMENTS: dict[str, str | None] = {
    "docling": "docling",
    "hosted": "pypdfium2",
    "text": None,
}


def supported_extensions() -> frozenset[str]:
    """File extensions the configured extractor accepts, e.g. ``".pdf"``."""
    return _EXTRACTOR_EXTENSIONS.get(config.EXTRACTOR, frozenset())


def missing_dependency(file_path: str | Path) -> str | None:
    """The package *file_path*'s extractor needs but cannot import, if any.

    Lets a caller refuse a file up front instead of accepting it and failing
    in the background.
    """
    from importlib.util import find_spec

    required = _EXTRACTOR_REQUIREMENTS.get(_extractor_for_file(file_path))
    if required is None or find_spec(required) is not None:
        return None
    return required


def _extractor_for_file(file_path: str | Path) -> str:
    """Resolve and validate the extractor for *file_path*."""
    extension = Path(file_path).suffix.lower()
    configured = config.EXTRACTOR
    if configured == "auto":
        extractor = _AUTO_EXTRACTORS.get(extension)
        if extractor is None:
            raise ValueError(
                f"Unsupported file extension for auto mode: {extension!r}."
            )
        return extractor

    supported = _EXTRACTOR_EXTENSIONS.get(configured)
    if supported is None:
        expected = ", ".join(repr(name) for name in sorted(_EXTRACTOR_EXTENSIONS))
        raise ValueError(
            f"Unknown EXTRACTOR {configured!r}; expected one of {expected}."
        )
    if extension not in supported:
        raise ValueError(
            f"The {configured!r} extractor does not support {extension!r} files."
        )
    return configured


#: Extractors that hand back rows without embedding them, keyed to the module
#: holding their ``extract_chunks``. ``ingest`` embeds their output in a second
#: pass so extraction stays testable without an endpoint and a re-embed does not
#: re-extract.
_LIBRARY_EXTRACTORS: dict[str, str] = {
    "docling": "unstructured_ao.docling_extract",
    "hosted": "unstructured_ao.hosted_extract",
    "text": "unstructured_ao.text_extract",
}


def _extract_with_library(file_path: str, extractor: str) -> list[dict[str, Any]]:
    """Extract with a library extractor, then embed the rows as text."""
    from importlib import import_module

    from unstructured_ao.embed_client import get_embed_client

    module = import_module(_LIBRARY_EXTRACTORS[extractor])
    client = get_embed_client()
    chunks = module.extract_chunks(file_path, client=client)
    if not chunks:
        return []
    vectors = client.embed([chunk["text"] for chunk in chunks])
    for chunk, vector in zip(chunks, vectors):
        chunk["embedding"] = vector
    return chunks


def _resolve_content_type(chunk: dict[str, Any]) -> str:
    return _CONTENT_TYPE_MAP.get(chunk.get("element_type") or "text", "text")


def _source_metadata(chunk: dict[str, Any]) -> dict[str, Any]:
    """The pipeline's own fields for a chunk, minus the two we store in columns.

    ``text`` and ``embedding`` are dropped: both already have a home, and an
    embedding in a JSON column would be several times the size of the row.
    """
    return {k: v for k, v in chunk.items() if k not in ("text", "embedding")}


def _resolve_page_number(chunk: dict[str, Any]) -> int | None:
    page = chunk.get("page_number")
    if page is not None:
        try:
            return int(page)
        except (ValueError, TypeError):
            pass
    return None


def ingest_file(file_path: str | Path, conn: Connection) -> str | None:
    """Ingest a single file into Postgres and the vector store.

    Returns the ``unstructured_document.id`` of the new document, or of the
    existing one when the same absolute path was already ingested.  Returns
    ``None`` when the extractor fails or yields no chunks.

    To re-ingest, call ``dal.delete_document`` first to cascade-delete all
    downstream rows before calling this function again.
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
        logger.exception(
            "The %s extractor failed on %s — skipping.", config.EXTRACTOR, path
        )
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
    _store_chunks(conn, document_id, name, raw_chunks)
    return document_id


def ingest_document(
    document_id: str,
    file_path: str | Path,
    conn: Connection,
    *,
    then: str = "ready",
) -> int:
    """Extract *file_path* into the already-inserted document *document_id*.

    For uploads, whose row is inserted as ``processing`` before extraction
    starts.  Moves the document to status *then* — ``extracting`` when the
    semantic pass follows — and returns the chunk count; raises
    when the extractor fails, yields nothing, or yields more than
    ``config.MAX_DOCUMENT_WORDS`` words, leaving the status to the caller,
    whose transaction this rolls back.
    """
    path = str(Path(file_path).expanduser().resolve())
    name = Path(path).name

    logger.info("Extracting chunks from %s …", name)
    raw_chunks = _extract_chunks_from_file(path)
    if not raw_chunks:
        raise ValueError(f"No content could be extracted from {name}.")
    words = sum(len((chunk.get("text") or "").split()) for chunk in raw_chunks)
    if words > config.MAX_DOCUMENT_WORDS:
        raise ValueError(
            f"{name} has {words:,} words; the limit is {config.MAX_DOCUMENT_WORDS:,}."
        )

    _store_chunks(conn, document_id, name, raw_chunks)
    set_document_status(conn, document_id, then)
    return len(raw_chunks)


def extract_document_semantics(document_id: str, conn: Connection) -> None:
    """Extract *document_id*'s term and attributes and link them to chunks.

    The LLM pass that follows ingest: ``extract.run_extraction`` over the
    document's chunk windows, then ``retrieve.link_document_attributes`` for the
    vector links.  Marks the document ``ready``; raises when no term came out,
    since ``run_extraction`` only logs that and a document with no term would
    otherwise read as done.
    """
    from unstructured_ao.dal import get_document_chunks, get_document_term
    from unstructured_ao.extract import run_extraction
    from unstructured_ao.retrieve import link_document_attributes
    from unstructured_ao.windows import build_windows

    chunks = get_document_chunks(conn, document_id)
    windows = build_windows([c["id"] for c in chunks], [c["content"] for c in chunks])
    run_extraction(document_id, windows, conn)
    if get_document_term(conn, document_id) is None:
        raise RuntimeError("Term extraction failed; see the server log.")
    link_document_attributes(document_id, conn)
    set_document_status(conn, document_id, "ready")


def _acquire_document_lock(conn: Connection, document_id: str) -> bool:
    """Own this document's pipeline across transactions and server replicas."""
    acquired = conn.execute(
        text("SELECT pg_try_advisory_lock(hashtextextended(:document_id, 0))"),
        {"document_id": document_id},
    ).scalar_one()
    conn.commit()
    return bool(acquired)


def _release_document_lock(conn: Connection, document_id: str) -> None:
    conn.execute(
        text("SELECT pg_advisory_unlock(hashtextextended(:document_id, 0))"),
        {"document_id": document_id},
    )
    conn.commit()


def run_document_ingest(document_id: str, file_path: str | Path) -> None:
    """Ingest an uploaded document, then extract its semantics, recording failure.

    Meant for a background task or startup recovery: nothing waits on it, so a
    failure is written to the row as ``failed`` with the error instead of being
    raised.  ``processing`` resumes at file ingestion and ``extracting``
    resumes at semantic extraction.

    Two transactions, so the chunks are committed — and searchable — while the
    slower LLM pass runs, and the list can tell the two stages apart.  A failed
    semantic pass leaves the chunks in place.  A session advisory lock prevents
    a rolling deployment or multiple replicas from resuming the same row.
    """
    from auto_ontology.dal.session import get_engine

    engine = get_engine()
    with engine.connect() as conn:
        if not _acquire_document_lock(conn, document_id):
            logger.info("Document %s is already being ingested.", document_id)
            return
        try:
            document = get_document(conn, document_id)
            conn.commit()
            if document is None or document["status"] not in (
                "processing",
                "extracting",
            ):
                return

            if document["status"] == "processing":
                extract = config.EXTRACT_ON_UPLOAD
                try:
                    with conn.begin():
                        ingest_document(
                            document_id,
                            file_path,
                            conn,
                            then="extracting" if extract else "ready",
                        )
                except Exception as exc:
                    logger.exception("Ingest failed for document %s.", document_id)
                    with conn.begin():
                        set_document_status(conn, document_id, "failed", error=str(exc))
                    return
                if not extract:
                    return

            try:
                with conn.begin():
                    extract_document_semantics(document_id, conn)
            except Exception as exc:
                logger.exception(
                    "Semantic extraction failed for document %s.", document_id
                )
                with conn.begin():
                    set_document_status(
                        conn,
                        document_id,
                        "failed",
                        error=f"Term and attribute extraction failed: {exc}",
                    )
        finally:
            _release_document_lock(conn, document_id)


def recover_pending_document_ingests() -> None:
    """Resume every non-terminal upload found when the API starts."""
    from auto_ontology.dal.session import get_engine

    engine = get_engine()
    with engine.connect() as conn:
        pending = list_pending_documents(conn)
        conn.commit()
    if pending:
        logger.info("Recovering %d interrupted document ingest(s).", len(pending))
    for document in pending:
        run_document_ingest(document["id"], document["path"])


def _store_chunks(
    conn: Connection,
    document_id: str,
    name: str,
    raw_chunks: list[dict[str, Any]],
) -> None:
    """Write *raw_chunks* to ``unstructured_chunk`` and the chunk vector store."""
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
            source_metadata=_source_metadata(chunk),
            chunk_id=cid,
        )

    logger.info("Stored %d chunk(s) for %s.", len(raw_chunks), name)

    # --- Upload the pipeline's embeddings to the vector store ---
    from unstructured_ao.vectorstore import get_chunk_store

    # The embed stage never raises: a chunk it could not embed comes back with
    # an empty ``embedding`` instead, so the only way to tell a working run from
    # a dead endpoint is to count what arrived.
    texts: list[str] = []
    vectors: list[list[float]] = []
    metadatas: list[dict[str, Any]] = []
    vector_ids: list[str] = []
    for i, (chunk, cid) in enumerate(zip(raw_chunks, chunk_ids)):
        embedding = chunk.get("embedding")
        if not embedding:
            continue
        texts.append(chunk.get("text", "") or "")
        vectors.append([float(v) for v in embedding])
        metadatas.append(
            {
                "document_id": document_id,
                "chunk_index": i,
                "content_type": _resolve_content_type(chunk),
                "page_number": _resolve_page_number(chunk),
                "source": name,
            }
        )
        vector_ids.append(cid)

    # Raising rolls the chunk rows back with the caller's transaction. A
    # document whose chunks exist relationally but have no vectors is invisible
    # to retrieval, which is worse than not having ingested it at all.
    if not vectors:
        raise RuntimeError(
            f"Embed stage returned no embeddings for any of "
            f"{len(raw_chunks)} chunk(s) in {name} — check the EMBED_* triplet "
            f"and the endpoint's rate limit."
        )
    if len(vectors) < len(raw_chunks):
        logger.warning(
            "Only %d/%d chunk(s) of %s came back embedded.",
            len(vectors),
            len(raw_chunks),
            name,
        )

    store = get_chunk_store(vector_size=len(vectors[0]))
    store.add_embeddings(
        texts=texts, embeddings=vectors, metadatas=metadatas, ids=vector_ids
    )
    logger.info(
        "Uploaded %d embedding(s) to %s.%s.",
        len(vectors),
        config.CHUNK_SCHEMA,
        config.CHUNK_TABLE,
    )


def ingest_directory(
    directory: str | Path,
    conn: Connection | None = None,
) -> list[str]:
    """Ingest all supported files in *directory*.

    Returns the list of document ids successfully ingested.

    *conn* must be an open SQLAlchemy ``Connection`` inside a transaction.
    When ``None`` a new engine + connection is created from env vars.
    """
    from auto_ontology.dal.session import get_engine

    data_dir = Path(directory)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}.")

    supported = _EXTRACTOR_EXTENSIONS.get(config.EXTRACTOR)
    if supported is None:
        expected = ", ".join(repr(name) for name in sorted(_EXTRACTOR_EXTENSIONS))
        raise ValueError(
            f"Unknown EXTRACTOR {config.EXTRACTOR!r}; expected one of {expected}."
        )
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
    """CLI entry point: ingest every supported file in a directory."""
    import argparse

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    parser = argparse.ArgumentParser(
        description="Ingest unstructured documents into Postgres."
    )
    parser.add_argument("directory", help="Directory of documents to ingest.")
    args = parser.parse_args()

    doc_ids = ingest_directory(args.directory)
    print(f"Ingested {len(doc_ids)} document(s): {doc_ids}")


if __name__ == "__main__":
    main()
