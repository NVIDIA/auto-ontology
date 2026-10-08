# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Retrieve: link each attribute to the chunks that support it.

For every attribute of a document's term:
1. Build a query string from the attribute name + description.
2. Run a vector similarity search against the ``CHUNK_COLLECTION`` PGVectorStore,
   filtered to this document's chunks by ``document_id`` metadata.
3. Store the top-k (attribute_id, chunk_id, score) rows in
   ``unstructured_attribute__chunk``.

This module is intentionally separate from ``ingest.py`` so the two steps can
be run independently (e.g. re-run retrieval after the embedding model changes,
without re-extracting text).

Usage
-----
    from unstructured.retrieve import link_document_attributes

    with engine.begin() as conn:
        link_document_attributes(document_id, conn)
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import Connection

import unstructured.config as config  # noqa: F401  — Ray override
from auto_ontology.dal.schema import unstructured_attribute__chunk
from unstructured.dal import (
    get_document_attributes,
    get_document_chunks,
    insert_attribute_chunk_link,
)

logger = logging.getLogger(__name__)


def _build_pgvector_store():  # type: ignore[return]
    """Return the same PGVectorStore used at ingest time."""
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


def _clear_existing_links(conn: Connection, attribute_id: str) -> None:
    conn.execute(
        unstructured_attribute__chunk.delete().where(
            unstructured_attribute__chunk.c.attribute_id == attribute_id
        )
    )


def link_document_attributes(
    document_id: str,
    conn: Connection,
    *,
    top_k: int = config.RETRIEVE_TOP_K,
    relink: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    """Search for supporting chunks and write attribute → chunk links.

    Parameters
    ----------
    document_id:
        The ``unstructured_document.id`` whose attributes to link.
    conn:
        Open SQLAlchemy ``Connection`` inside a transaction.
    top_k:
        Maximum chunks to link per attribute.
    relink:
        When ``True`` (default) clear existing links before re-inserting.

    Returns
    -------
    A dict mapping attribute_id → list of linked chunk records.
    """
    attributes = get_document_attributes(conn, document_id)
    if not attributes:
        logger.warning(
            "No attributes found for document %s — nothing to link.", document_id
        )
        return {}

    # Collect valid chunk ids for this document so we can filter VDB hits.
    doc_chunks = get_document_chunks(conn, document_id)
    valid_chunk_ids = {c["id"] for c in doc_chunks}
    if not valid_chunk_ids:
        logger.warning(
            "Document %s has no chunks — cannot build attribute links.", document_id
        )
        return {}

    try:
        vdb = _build_pgvector_store()
    except Exception:
        logger.exception("Failed to connect to PGVectorStore — skipping link step.")
        return {}

    results: dict[str, list[dict[str, Any]]] = {}

    for attr in attributes:
        attribute_id = attr["id"]
        query_text = f"{attr['name']}: {attr['description'] or ''}"

        if relink:
            _clear_existing_links(conn, attribute_id)

        try:
            # Filter by document_id metadata so we only get chunks from this file.
            hits = vdb.similarity_search_with_score(
                query=query_text,
                k=top_k,
                filter={"document_id": document_id},
            )
        except Exception:
            logger.exception(
                "Vector search failed for attribute %s (%r) — skipping.",
                attribute_id,
                attr["name"],
            )
            results[attribute_id] = []
            continue

        linked: list[dict[str, Any]] = []
        for doc, score in hits:
            # The langchain Document id comes from ``ids=chunk_ids`` passed at
            # add_documents time.  It lives in doc.id (langchain 0.2+) or
            # doc.metadata.get("id") on earlier versions.
            chunk_id = getattr(doc, "id", None) or doc.metadata.get("id")
            if not chunk_id:
                logger.debug("Hit missing id — skipping: %r", doc.metadata)
                continue
            if chunk_id not in valid_chunk_ids:
                # Sanity-check: the filter should have handled this, but in case
                # the metadata filter is not applied by this VDB adapter version.
                logger.debug(
                    "Chunk %s not in valid set for document %s — skipping.",
                    chunk_id,
                    document_id,
                )
                continue

            insert_attribute_chunk_link(
                conn,
                attribute_id=attribute_id,
                chunk_id=chunk_id,
                score=float(score),
            )
            linked.append({"chunk_id": chunk_id, "score": float(score)})

        results[attribute_id] = linked
        logger.info(
            "Linked attribute %r → %d chunk(s) (document %s).",
            attr["name"],
            len(linked),
            document_id,
        )

    return results


def link_all_documents(conn: Connection | None = None) -> None:
    """Run ``link_document_attributes`` for every ingested document.

    Useful as a batch job after changing the embedding model or top_k.
    """
    from sqlalchemy import select

    from auto_ontology.dal.schema import unstructured_document

    def _run(conn: Connection) -> None:
        doc_ids = [
            row[0]
            for row in conn.execute(select(unstructured_document.c.id)).fetchall()
        ]
        logger.info("Linking attributes for %d document(s).", len(doc_ids))
        for doc_id in doc_ids:
            link_document_attributes(doc_id, conn)

    if conn is not None:
        _run(conn)
        return

    from auto_ontology.dal.session import get_engine

    engine = get_engine()
    with engine.begin() as c:
        _run(c)


def main() -> None:
    """CLI entry point: link attributes for all ingested documents."""
    import argparse

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    parser = argparse.ArgumentParser(
        description="Link document attributes to supporting chunks via vector search."
    )
    parser.add_argument(
        "--document-id", help="Process a single document instead of all."
    )
    args = parser.parse_args()

    if args.document_id:
        from auto_ontology.dal.session import get_engine

        engine = get_engine()
        with engine.begin() as conn:
            result = link_document_attributes(args.document_id, conn)
        total = sum(len(v) for v in result.values())
        print(f"Linked {total} attribute–chunk pairs for document {args.document_id}.")
    else:
        link_all_documents()
        print("Done.")


if __name__ == "__main__":
    main()
