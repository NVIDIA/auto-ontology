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
1. The extractor named by ``config.EXTRACTOR`` turns the file into chunks of
   text, tables, charts and infographics.  ``hosted`` and ``docling`` return
   text only and are embedded in step 3; ``nemo`` embeds inside its own graph
   pass and skips it.
2. Each chunk is written to ``unstructured_chunk`` (relational) with a
   pre-generated UUID that is also used as the vector-store row key.
3. The embeddings are uploaded to the chunk vector store under those same ids,
   so attribute → chunk vector search works in the extraction step.

Usage
-----
    from unstructured_ao import config  # must be imported first (Ray override)
    from unstructured_ao.ingest import ingest_directory

    with engine.begin() as conn:
        for doc_id in ingest_directory("/path/to/documents", conn):
            print(f"Ingested document {doc_id}")

The ``nemo`` extractor additionally needs ``RAY_AUTH_MODE=disabled`` in the
environment.  It runs its stages as Ray actors, and without that the raylet
spends ~80s on auth setup before giving up with "The current node timed out
during startup"; with it Ray is up in about 7.  It is not set in ``config.py``
on purpose — it turns off a security feature, which is a call for whoever runs
the pipeline to make, not a default to inherit.

Terms and attributes come afterwards, from
``unstructured_ao.extract.run_extraction``.
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

#: NeMo Retriever element types → our canonical ``content_type`` values.
_CONTENT_TYPE_MAP: dict[str, str] = {
    "text": "text",
    "structured": "table",
    "table": "table",
    "chart": "chart",
    "infographic": "infographic",
    "image": "infographic",
}


def _embed_params():
    """NeMo Retriever ``EmbedParams`` for the chunk embeddings, ``text_image``.

    Reads the same ``EMBED_*`` triplet as the tabular ingest and the retriever,
    through auto-ontology's ``resolve``, so one set of env vars configures
    every embed path.  It does not reuse ``get_embed_params``: since #376 that
    returns auto-ontology's own dataclass for its in-repo embed client, and the
    extract graph needs the library's model.

    ``text_image`` rather than ``text`` because half of what the extract stage
    yields here is charts, tables and infographics, and ``EMBED_MODEL`` is a
    vision-language model that can read them as images.

    Changing modality invalidates the vectors already in the store and requires
    re-ingesting the corpus.
    """
    from nemo_retriever.common.params.models import EmbedParams

    from auto_ontology.utils.model_config import resolve

    return EmbedParams(
        embed_invoke_url=resolve("EMBED", "ENDPOINT"),
        model_name=resolve("EMBED", "MODEL"),
        api_key=resolve("EMBED", "API_KEY"),
        embed_modality="text_image",
    )


def _extract_chunks_from_file(file_path: str) -> list[dict[str, Any]]:
    """Extract and embed *file_path* with whichever extractor is configured.

    ``config.EXTRACTOR`` picks between NeMo Retriever's graph, Docling, and the
    fully hosted path.  All three return the same record shape with an
    ``embedding`` attached, so nothing downstream of here knows which one ran.
    """
    if config.EXTRACTOR == "nemo":
        return _extract_with_nemo(file_path)
    if config.EXTRACTOR in _LIBRARY_EXTRACTORS:
        return _extract_with_library(file_path)
    raise ValueError(
        f"Unknown EXTRACTOR {config.EXTRACTOR!r}; expected one of "
        f"'nemo', {', '.join(repr(name) for name in sorted(_LIBRARY_EXTRACTORS))}."
    )


#: Extractors that hand back rows without embedding them, keyed to the module
#: holding their ``extract_chunks``.  Unlike NeMo's graph these do not embed, so
#: ``ingest`` does it in a second pass: extraction stays testable without an
#: endpoint, and a re-embed does not re-extract.
_LIBRARY_EXTRACTORS: dict[str, str] = {
    "docling": "unstructured_ao.docling_extract",
    "hosted": "unstructured_ao.hosted_extract",
}


def _extract_with_library(file_path: str) -> list[dict[str, Any]]:
    """Extract with a library extractor, then embed the rows.

    Embedding is ``text`` only — measured better than ``text_image`` on this
    corpus, see :func:`_embed_params`.
    """
    from importlib import import_module

    from unstructured_ao.embed_client import get_embed_client

    module = import_module(_LIBRARY_EXTRACTORS[config.EXTRACTOR])
    client = get_embed_client()
    chunks = module.extract_chunks(file_path, client=client)
    if not chunks:
        return []
    vectors = client.embed([chunk["text"] for chunk in chunks])
    for chunk, vector in zip(chunks, vectors):
        chunk["embedding"] = vector
    return chunks


def _extract_with_nemo(file_path: str) -> list[dict[str, Any]]:
    """Extract and embed *file_path* in one NeMo Retriever pass.

    Each returned dict is one element of a page, in the library's compact
    result shape — ``text``, ``source_id``, ``element_type``, ``page_number``
    — plus the ``embedding`` this function attaches.

    ``ingest`` itself hands back the full pipeline frame, one row per element
    and some thirty columns wide, most of which are stage bookkeeping and
    base64 page images.  ``dataframe_to_transport_records`` is the library's own
    projection down to the public fields, so the column names the stages happen
    to use stay the library's business rather than something we read directly.
    The one exception is the embedding, which that projection drops; see
    :func:`_embeddings_from_frame`.

    Embedding is part of the graph rather than a second pass so each chunk is
    embedded by the same VL model that read it, and so the retry and batching
    policy is the library's rather than ours.  ``vdb_upload`` is deliberately
    not in the chain: the vectors are written separately under ids we choose,
    which is what lets a search hit resolve back to its ``unstructured_chunk``
    row.
    """
    from nemo_retriever import create_ingestor
    from nemo_retriever.ingestor.results import dataframe_to_transport_records

    params = _embed_params()
    ingestor = (
        create_ingestor(run_mode="batch")
        .files([file_path])
        .extract(
            extract_text=True,
            extract_tables=True,
            extract_charts=True,
            extract_infographics=True,
        )
        .embed(params)
    )
    frame = ingestor.ingest()
    chunks = dataframe_to_transport_records(frame, result_schema="compact")
    for chunk, embedding in zip(chunks, _embeddings_from_frame(frame, params)):
        chunk["embedding"] = embedding
    return chunks


def _embeddings_from_frame(frame: Any, params: Any) -> list[list[float] | None]:
    """The embed stage's vectors, in the frame's row order.

    Taken off the frame rather than out of the transport records because
    ``compact_result_record`` has no embedding field at all: the
    ``return_embeddings`` flag applies only to ``result_schema="legacy"``.
    Legacy is not an alternative here — it keeps every column, base64 page
    images included, and ``_source_metadata`` would write those into Postgres.
    Column names come from the same params the stage was configured with, so a
    change of embed model does not strand this on a stale column.
    """
    column = params.output_column
    if column not in frame.columns:
        return [None] * len(frame)
    flag = params.has_embedding_column

    def vector(row: Any) -> list[float] | None:
        if flag in frame.columns and not row[flag]:
            return None
        cell = row[column]
        # The stage reports a per-row embed failure in ``info_msg`` alongside
        # the vector, so a dict here is the success shape, not a wrapper to
        # unwrap defensively.
        if isinstance(cell, dict):
            cell = cell.get("embedding")
        return None if cell is None else list(cell)

    return [vector(row) for _, row in frame.iterrows()]


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

    return document_id


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
