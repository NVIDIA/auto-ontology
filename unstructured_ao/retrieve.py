# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Retrieve: link each attribute to the chunks that support it.

Extraction already linked the chunks it cited.  This step adds the ones it
could not cite — the chunks in windows the attribute was not extracted from —
so the two together cover the whole document.

For every attribute of a document's term:
1. Build a query string from the attribute name + description.
2. Run a vector similarity search against the chunk vector store, filtered to
   this document's chunks by the ``document_id`` metadata column.
3. Store the top-k (attribute_id, chunk_id, score) rows in
   ``unstructured_attribute__chunk`` as ``source='vector'`` links.

This module is intentionally separate from ``ingest.py`` so the two steps can
be run independently (e.g. re-run retrieval after the embedding model changes,
without re-extracting text).

Usage
-----
    from unstructured_ao.retrieve import link_document_attributes

    with engine.begin() as conn:
        link_document_attributes(document_id, conn)
"""

from __future__ import annotations

import logging
from collections.abc import Collection
from typing import Any

from sqlalchemy import Connection

import unstructured_ao.config as config  # noqa: F401  — Ray override
from auto_ontology.dal.schema import unstructured_attribute__chunk
from unstructured_ao.dal import (
    get_document_attributes,
    get_document_chunks,
    insert_attribute_chunk_link,
)
from unstructured_ao.vectorstore import get_chunk_store

logger = logging.getLogger(__name__)


def _clear_existing_links(conn: Connection, attribute_id: str) -> None:
    """Drop this attribute's vector links so the search can rebuild them.

    Scoped to ``source='vector'``: the llm links record which chunks the
    extractor actually read, and re-running retrieval does not make that
    untrue.
    """
    conn.execute(
        unstructured_attribute__chunk.delete().where(
            unstructured_attribute__chunk.c.attribute_id == attribute_id,
            unstructured_attribute__chunk.c.source == "vector",
        )
    )


def _search_with_retry(
    vdb: Any, query_text: str, top_k: int, document_id: str
) -> list[tuple[Any, float]]:
    """``similarity_search_with_score`` with backoff on a saturated endpoint.

    Every search embeds its query, so linking a document costs one embedding
    request per attribute — hundreds of them, on top of the extraction calls
    and alongside whatever other documents are running.  That saturates the
    endpoint, and a refused request looks exactly like an attribute no chunk
    supports: both end up with zero links.  Retrying is what keeps a 429 from
    being recorded as an eval result.
    """
    import random
    import time

    from auto_ontology.utils.llm_invoke import RETRY_MAX_ATTEMPTS, _is_retryable

    for attempt in range(RETRY_MAX_ATTEMPTS):
        try:
            # Filter by document_id metadata so we only get chunks from this file.
            return vdb.similarity_search_with_score(
                query=query_text,
                k=top_k,
                filter={"document_id": document_id},
            )
        except Exception as exc:
            if not _is_retryable(exc) or attempt == RETRY_MAX_ATTEMPTS - 1:
                raise
            wait = 2 ** (attempt + 1) + random.uniform(0, 1)
            logger.warning(
                "Retryable vector-search error on attempt %d/%d — retrying in "
                "%.1fs: %s",
                attempt + 1,
                RETRY_MAX_ATTEMPTS,
                wait,
                str(exc)[:120],
            )
            time.sleep(wait)

    raise RuntimeError("unreachable")  # pragma: no cover


def link_document_attributes(
    document_id: str,
    conn: Connection,
    *,
    top_k: int = config.RETRIEVE_TOP_K,
    relink: bool = True,
    only_attribute_ids: Collection[str] | None = None,
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
    only_attribute_ids:
        Restrict to these attributes.  For repairing a run where some searches
        were lost to a saturated endpoint: re-searching only the attributes
        that came away empty avoids re-embedding the whole document, which is
        what saturated it in the first place.

    Returns
    -------
    A dict mapping attribute_id → list of linked chunk records.
    """
    attributes = get_document_attributes(conn, document_id)
    if only_attribute_ids is not None:
        wanted = set(only_attribute_ids)
        attributes = [a for a in attributes if a["id"] in wanted]
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

    vdb = get_chunk_store()

    results: dict[str, list[dict[str, Any]]] = {}
    # Attributes whose search never got an answer.  They are indistinguishable
    # in the link table from attributes the search found nothing for, so they
    # are counted here and reported at the end rather than left to look like a
    # genuine miss in the eval.
    unsearchable: list[str] = []

    for attr in attributes:
        attribute_id = attr["id"]
        query_text = f"{attr['name']}: {attr['description'] or ''}"

        if relink:
            _clear_existing_links(conn, attribute_id)

        try:
            hits = _search_with_retry(vdb, query_text, top_k, document_id)
        except Exception:
            logger.exception(
                "Vector search failed for attribute %s (%r) — skipping.",
                attribute_id,
                attr["name"],
            )
            results[attribute_id] = []
            unsearchable.append(attr["name"])
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
                source="vector",
            )
            linked.append({"chunk_id": chunk_id, "score": float(score)})

        results[attribute_id] = linked
        logger.info(
            "Linked attribute %r → %d chunk(s) (document %s).",
            attr["name"],
            len(linked),
            document_id,
        )

    if unsearchable:
        logger.warning(
            "%d of %d attribute(s) in document %s have no vector links because "
            "the search never answered, not because nothing matched: %s",
            len(unsearchable),
            len(attributes),
            document_id,
            ", ".join(repr(n) for n in unsearchable[:5])
            + (" …" if len(unsearchable) > 5 else ""),
        )

    return results


_MISSING_VECTOR_LINKS = """
SELECT t.document_id, a.id AS attribute_id
  FROM unstructured_attribute a
  JOIN unstructured_term t ON t.id = a.term_id
 WHERE NOT EXISTS (
         SELECT 1 FROM unstructured_attribute__chunk l
          WHERE l.attribute_id = a.id AND l.source = 'vector'
       )
 ORDER BY t.document_id
"""


def repair_missing_links(*, document_id: str | None = None) -> int:
    """Re-search the attributes that have no ``source='vector'`` links.

    For recovering a run where the embedding endpoint shed requests: an
    attribute whose search never answered is committed with its LLM-grounded
    links but no vector ones, and is indistinguishable in the link table from
    an attribute the search genuinely found nothing for.  So this re-searches
    both.  The genuine misses cost one embedding each and come back empty
    again, which is cheap next to re-linking whole documents.

    Only the linking step runs — no LLM extraction, so nothing already
    extracted is recomputed or at risk.  Returns the number re-linked.
    """
    from sqlalchemy import text as sql_text

    from auto_ontology.dal.session import get_engine

    engine = get_engine()
    with engine.connect() as conn:
        rows = list(conn.execute(sql_text(_MISSING_VECTOR_LINKS)).mappings())

    by_document: dict[str, list[str]] = {}
    for row in rows:
        if document_id and row["document_id"] != document_id:
            continue
        by_document.setdefault(row["document_id"], []).append(row["attribute_id"])

    if not by_document:
        logger.info("Every attribute already has vector links — nothing to repair.")
        return 0

    total = sum(len(v) for v in by_document.values())
    logger.info(
        "Repairing %d attribute(s) across %d document(s).", total, len(by_document)
    )

    repaired = 0
    for doc_id, attribute_ids in by_document.items():
        # One transaction per document, so a document that fails again leaves
        # the ones already repaired committed.
        with engine.begin() as conn:
            result = link_document_attributes(
                doc_id,
                conn,
                relink=False,
                only_attribute_ids=attribute_ids,
            )
        linked = sum(1 for v in result.values() if v)
        repaired += linked
        logger.info(
            "  document %s: %d/%d attribute(s) now linked.",
            doc_id,
            linked,
            len(attribute_ids),
        )

    logger.info(
        "Repaired %d of %d attribute(s); the rest genuinely match no chunk.",
        repaired,
        total,
    )
    return repaired


def search_attributes(
    query: str,
    *,
    k: int = 10,
    document_id: str | None = None,
    attribute_type: str | None = None,
) -> list[dict[str, Any]]:
    """Find the attributes across the corpus that match *query*.

    This is the counterpart to a chunk search, and answers a different
    question: chunk search returns the passages that mention something,
    attribute search returns the concepts themselves — each with the term and
    document it came from, and resolvable to its supporting chunks through
    ``unstructured_attribute__chunk``.

    Unfiltered it spans every ingested document, so one concept stated in
    several reports comes back once per report.  *document_id* narrows it to
    one file and *attribute_type* to one kind of attribute.
    """
    from sqlalchemy import select

    from auto_ontology.dal.schema import (
        unstructured_attribute,
        unstructured_document,
        unstructured_term,
    )
    from auto_ontology.dal.session import get_engine
    from unstructured_ao.vectorstore import get_attribute_store

    search_filter: dict[str, str] = {}
    if document_id:
        search_filter["document_id"] = document_id
    if attribute_type:
        search_filter["attribute_type"] = attribute_type

    hits = get_attribute_store().similarity_search_with_score(
        query=query, k=k, **({"filter": search_filter} if search_filter else {})
    )

    scores = {}
    for doc, score in hits:
        attribute_id = getattr(doc, "id", None) or doc.metadata.get("id")
        if attribute_id:
            scores[attribute_id] = float(score)
    if not scores:
        return []

    # One round trip for the names: the vector table stores the embedded text,
    # not the columns a caller wants to display.
    query = (
        select(
            unstructured_attribute.c.id,
            unstructured_attribute.c.name,
            unstructured_attribute.c.description,
            unstructured_attribute.c.attribute_type,
            unstructured_term.c.name.label("term"),
            unstructured_document.c.name.label("document"),
        )
        .select_from(
            unstructured_attribute.join(
                unstructured_term,
                unstructured_term.c.id == unstructured_attribute.c.term_id,
            ).join(
                unstructured_document,
                unstructured_document.c.id == unstructured_term.c.document_id,
            )
        )
        .where(unstructured_attribute.c.id.in_(list(scores)))
    )
    with get_engine().connect() as conn:
        rows = {r["id"]: dict(r) for r in conn.execute(query).mappings()}

    return sorted(
        ({**rows[i], "score": s} for i, s in scores.items() if i in rows),
        key=lambda r: r["score"],
    )


def index_missing_attributes(batch_size: int = 256) -> int:
    """Embed and index every attribute not yet in the attribute vector store.

    Extraction indexes as it goes, so this is for attributes written before
    indexing existed and for re-indexing after an embedding model change.
    Returns the number of attributes added.
    """
    from sqlalchemy import text as sql_text

    from auto_ontology.dal.session import get_engine
    from unstructured_ao.extract import attribute_text
    from unstructured_ao.vectorstore import build_embeddings, get_attribute_store

    with get_engine().connect() as conn:
        indexed: set[str] = set()
        if conn.execute(
            sql_text(
                "SELECT to_regclass(:t)",
            ),
            {"t": f"{config.ATTRIBUTE_SCHEMA}.{config.ATTRIBUTE_TABLE}"},
        ).scalar():
            indexed = {
                r[0]
                for r in conn.execute(
                    sql_text(
                        f"SELECT langchain_id::text FROM "  # noqa: S608 — names are constants
                        f"{config.ATTRIBUTE_SCHEMA}.{config.ATTRIBUTE_TABLE}"
                    )
                )
            }
        pending = [
            dict(r)
            for r in conn.execute(
                sql_text(
                    "SELECT a.id, a.name, a.description, a.attribute_type,"
                    "       a.term_id, t.document_id"
                    "  FROM unstructured_attribute a"
                    "  JOIN unstructured_term t ON t.id = a.term_id"
                )
            ).mappings()
            if r["id"] not in indexed
        ]

    if not pending:
        logger.info("Attribute index is up to date (%d indexed).", len(indexed))
        return 0

    logger.info("Indexing %d attribute(s) …", len(pending))
    embeddings = build_embeddings()
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        texts = [attribute_text(r["name"], r["description"]) for r in batch]
        vectors = embeddings.embed_documents(texts)
        get_attribute_store(vector_size=len(vectors[0])).add_embeddings(
            texts=texts,
            embeddings=vectors,
            metadatas=[
                {
                    "document_id": r["document_id"],
                    "term_id": r["term_id"],
                    "attribute_type": r["attribute_type"],
                }
                for r in batch
            ],
            ids=[r["id"] for r in batch],
        )
        logger.info("  %d/%d", min(start + batch_size, len(pending)), len(pending))

    return len(pending)


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
