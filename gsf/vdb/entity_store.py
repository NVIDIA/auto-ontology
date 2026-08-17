# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Vector storage on the entities themselves, rather than in a side collection.

Every embeddable thing in GSF is already a row: a Table is a ``catalog_table``
row, a Term is a ``term`` row. Previously each also had a *second* row in one of
two ``vdb`` collections, tied back only by an id inside a JSON blob. That
arrangement cost three things:

* **Orphans.** Deleting a catalog row left its vector behind, because no
  constraint connected them. A benchmark database was found holding 1920 vectors
  for a catalog of ~849 -- retrieval was scoring rows that no longer existed.
* **A label filter on every search.** One collection held every kind of entity,
  so each query carried ``WHERE label = ...`` and scanned the others anyway.
* **A second write path.** Rows were inserted, not updated, so re-embedding
  depended on deleting first.

Putting ``embedding`` on the entity fixes all three by construction: the vector
is a column of the row, so ``ON DELETE CASCADE`` reaches it, each label *is* a
table, and re-embedding is an ``UPDATE``.

This module owns the mapping and the SQL. :class:`gsf.vdb.postgres.PostgresVDB`
keeps its public interface and delegates here, so ``Retriever``,
``search_semantic_index`` and every service call site are unchanged.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable

from pgvector.sqlalchemy import HALFVEC, Vector
from sqlalchemy import (
    Text,
    cast,
    column,
    literal,
    func,
    literal_column,
    select,
    update,
    values,
)

from gsf.dal import schema as s
from gsf.dal.schema import EMBEDDING_DIMENSIONS
from gsf.dal.session import store

logger = logging.getLogger(__name__)

#: How many extra candidates the approximate stage fetches before exact
#: re-ranking. 4x is the usual starting point: enough that HNSW's recall gap and
#: fp16 rounding cannot push a genuine top-k row out of the shortlist, cheap
#: enough that the re-rank stays trivial.
_OVERSAMPLE = 4

#: Label -> the table that owns rows of that label.
#:
#: This *is* the collection split that used to be expressed as two tables plus a
#: ``label`` column. ``search_semantic_index`` already issues one query per
#: label, so every search now resolves to exactly one table and needs no label
#: predicate at all.
LABEL_TABLES: dict[str, Any] = {
    "Table": s.catalog_table,
    "Column": s.catalog_column,
    "Term": s.term,
    "ColumnAttribute": s.column_attribute,
    "SqlAttribute": s.sql_attribute,
    "CustomAnalysis": s.custom_analysis,
    "PqlAnalysis": s.pql_analysis,
}

#: Which labels each former collection was responsible for. Retained because
#: callers still ask for "the data layer" or "the semantic layer", and because
#: ``delete_all`` has to know what it is allowed to clear.
COLLECTION_LABELS: dict[str, tuple[str, ...]] = {
    "data_objects_layer": ("Table", "Column"),
    "semantic_layer": (
        "Term",
        "ColumnAttribute",
        "SqlAttribute",
        "CustomAnalysis",
        "PqlAnalysis",
    ),
}


def labels_for_collection(collection_name: str) -> tuple[str, ...]:
    """Labels belonging to *collection_name*, or every label if unknown."""
    return COLLECTION_LABELS.get(collection_name, tuple(LABEL_TABLES))


def _coerce_vector(embedding: Any) -> list[float] | None:
    """Return *embedding* as a list of floats of the expected width.

    A wrong-width vector is rejected here rather than by Postgres, because the
    database error names neither the row nor the label and a single bad record
    would otherwise abort a batch of thousands.
    """
    if embedding is None:
        return None
    try:
        # Not `values`: that name is the SQLAlchemy VALUES constructor imported
        # above, and shadowing it here would be a trap for the next edit.
        floats = [float(v) for v in embedding]
    except (TypeError, ValueError):
        return None
    if len(floats) != EMBEDDING_DIMENSIONS:
        logger.warning(
            "embedding has %d dimensions, expected %d -- skipping",
            len(floats),
            EMBEDDING_DIMENSIONS,
        )
        return None
    return floats


def _flatten(records: Iterable) -> Iterable[dict]:
    """Yield record dicts from possibly-nested NV-Ingest output."""
    for item in records:
        if isinstance(item, dict):
            yield item
        elif isinstance(item, list):
            yield from _flatten(item)


def _record_fields(record: dict) -> tuple[str | None, str | None, str, str | None]:
    """Pull ``(id, label, text, database_name)`` out of one NV-Ingest record.

    The shape is the one ``CatalogEmbeddingRowsOp`` and the semantic embedder
    produce: identity lives in ``metadata``, and the searchable text may arrive
    under any of three keys depending on which producer built the record.
    """
    metadata = record.get("metadata") or {}
    content_metadata = metadata.get("content_metadata") or {}

    node_id = metadata.get("id") or content_metadata.get("id")
    label = metadata.get("label") or content_metadata.get("label")
    database_name = metadata.get("database_name") or content_metadata.get(
        "database_name"
    )
    body = record.get("text") or record.get("content") or metadata.get("content") or ""
    return (
        str(node_id) if node_id is not None else None,
        str(label) if label is not None else None,
        body,
        str(database_name) if database_name is not None else None,
    )


def write_embeddings(records: list, *, allowed_labels: Iterable[str]) -> int:
    """Attach embeddings to their entity rows. Returns the number updated.

    An ``UPDATE``, not an insert: the entity already exists by the time it is
    embedded (``ingest_catalog`` writes the catalog first), and updating means a
    re-embed needs no delete pass.

    A record whose row is gone is counted as skipped rather than failing the
    batch -- entities can be deleted between the catalog write and the embed,
    and that is not an error.
    """
    allowed = set(allowed_labels)
    by_label: dict[str, list[dict]] = {}
    skipped = 0

    for record in _flatten(records):
        node_id, label, body, database_name = _record_fields(record)
        vector = _coerce_vector((record.get("metadata") or {}).get("embedding"))
        if node_id is None or label is None or vector is None:
            skipped += 1
            continue
        if label not in allowed or label not in LABEL_TABLES:
            skipped += 1
            continue
        by_label.setdefault(label, []).append(
            {
                "row_id": node_id,
                "embedding": vector,
                "embedding_text": body,
                "embedding_database_name": database_name,
            }
        )

    updated = 0
    missing = 0
    for label, rows in by_label.items():
        table = LABEL_TABLES[label]
        # One `UPDATE ... FROM (VALUES ...)` per label rather than executemany,
        # because RETURNING then reports the rows that actually matched. That
        # distinction matters: an id with no row is the signal that the entity
        # was deleted between the catalog write and the embed, and executemany
        # would have reported it as a successful write.
        source = (
            values(
                column("row_id", Text),
                column("embedding", Vector(EMBEDDING_DIMENSIONS)),
                column("embedding_text", Text),
                column("embedding_database_name", Text),
                name="incoming",
            )
            .data(
                [
                    (
                        row["row_id"],
                        row["embedding"],
                        row["embedding_text"],
                        row["embedding_database_name"],
                    )
                    for row in rows
                ]
            )
            .alias("incoming")
        )
        result = store().query_write(
            update(table)
            .where(table.c.id == source.c.row_id)
            .values(
                # Explicit cast: a VALUES column renders as an untyped parameter,
                # which Postgres infers as text and then refuses to assign to a
                # vector column ("expression is of type text").
                embedding=cast(source.c.embedding, Vector(EMBEDDING_DIMENSIONS)),
                embedding_text=source.c.embedding_text,
                embedding_database_name=source.c.embedding_database_name,
            )
            .returning(table.c.id)
        )
        updated += len(result)
        missing += len(rows) - len(result)

    if skipped:
        logger.info("write_embeddings: skipped %d record(s)", skipped)
    if missing:
        logger.info(
            "write_embeddings: %d record(s) had no matching row (entity deleted "
            "since the catalog write)",
            missing,
        )
    logger.info(
        "write_embeddings: updated %d row(s) across %d label(s)",
        updated,
        len(by_label),
    )
    return updated


def search_statement(
    vector: list[float],
    *,
    label: str,
    top_k: int,
    database_name: str | None,
):
    """The SELECT one label's search issues.

    Split out from :func:`search` so the query plan can be asserted without
    executing anything -- see ``gsf/vdb/tests/test_vector_index.py``. Whether
    the HNSW index is reachable is a property of this statement's shape, and
    nothing about the returned rows reveals it.

    Two stages, and both halves matter.

    *Inner*: order by the **halfvec cast**, because that is the expression the
    index is built on and Postgres matches expression indexes syntactically --
    ordering by the plain column here would silently plan a sequential scan.
    Take ``_OVERSAMPLE`` times more rows than asked for, since this ranking is
    approximate twice over (fp16, and HNSW itself).

    *Outer*: re-rank that shortlist on the exact ``vector`` column, so the
    caller's ordering and distances are the ones an unindexed exact search would
    have produced. The approximation is confined to which rows are considered,
    never to what is reported.
    """
    table = LABEL_TABLES[label]
    approx = cast(table.c.embedding, HALFVEC(EMBEDDING_DIMENSIONS)).cosine_distance(
        cast(
            literal(vector, Vector(EMBEDDING_DIMENSIONS)),
            HALFVEC(EMBEDDING_DIMENSIONS),
        )
    )
    shortlist = (
        select(
            table.c.id.label("id"),
            table.c.embedding_text.label("text"),
            table.c.embedding_database_name.label("database_name"),
            table.c.embedding.label("embedding"),
        )
        # Not merely an optimisation: an unembedded row would otherwise sort as
        # maximally distant but still occupy one of the *k* slots.
        .where(table.c.embedding.isnot(None))
        .order_by(approx)
        .limit(top_k * _OVERSAMPLE)
    )
    if database_name:
        shortlist = shortlist.where(table.c.embedding_database_name == database_name)
    shortlist = shortlist.subquery()

    # `<=>` is cosine distance; lower is better, which matches the ordering
    # `_hits_to_semantic_rows` assumes.
    exact = shortlist.c.embedding.cosine_distance(vector).label("_distance")
    return (
        select(shortlist.c.id, shortlist.c.text, shortlist.c.database_name, exact)
        .order_by(exact)
        .limit(top_k)
    )


def search(
    query_embedding: list[float],
    *,
    labels: Iterable[str],
    top_k: int = 10,
    database_name: str | None = None,
) -> list[dict]:
    """Cosine k-NN over the tables owning *labels*, best-first.

    Returns hits shaped as ``search_semantic_index`` expects: ``text``,
    ``metadata`` (carrying ``id`` and ``label``) and ``_distance``.

    ``embedding IS NOT NULL`` is not merely an optimisation -- an unembedded row
    would otherwise sort as maximally distant but still occupy one of the *k*
    slots.
    """
    vector = _coerce_vector(query_embedding)
    if vector is None:
        return []

    hits: list[dict] = []
    for label in labels:
        if label not in LABEL_TABLES:
            continue
        statement = search_statement(
            vector, label=label, top_k=top_k, database_name=database_name
        )
        for row in store().query_read(statement):
            hits.append(
                {
                    "text": row["text"] or "",
                    "metadata": {
                        "id": row["id"],
                        "label": label,
                        "database_name": row["database_name"],
                    },
                    "_distance": float(row["_distance"]),
                }
            )

    # One query per label was issued, so the merged list is only sorted within
    # each label. The caller caps per label, but sorting here keeps the
    # "already sorted by distance" contract that `_hits_to_semantic_rows`
    # documents.
    hits.sort(key=lambda hit: hit["_distance"])
    return hits


def clear_by_id(node_id: str, *, labels: Iterable[str]) -> int:
    """Drop the embedding of one row. Returns rows affected."""
    cleared = 0
    for label in labels:
        table = LABEL_TABLES.get(label)
        if table is None:
            continue
        result = store().query_write(
            update(table)
            .where(table.c.id == node_id, table.c.embedding.isnot(None))
            .values(embedding=None, embedding_text=None)
            .returning(table.c.id)
        )
        cleared += len(result)
    return cleared


def clear_by_database(database_name: str, *, labels: Iterable[str]) -> list[str]:
    """Drop every embedding for one database. Returns the cleared row ids."""
    ids: list[str] = []
    for label in labels:
        table = LABEL_TABLES.get(label)
        if table is None:
            continue
        result = store().query_write(
            update(table)
            .where(
                table.c.embedding_database_name == database_name,
                table.c.embedding.isnot(None),
            )
            .values(embedding=None, embedding_text=None)
            .returning(table.c.id)
        )
        ids.extend(str(row["id"]) for row in result)
    return ids


def clear_all(*, labels: Iterable[str]) -> int:
    """Drop every embedding for *labels*. Returns rows affected."""
    cleared = 0
    for label in labels:
        table = LABEL_TABLES.get(label)
        if table is None:
            continue
        result = store().query_write(
            update(table)
            .where(table.c.embedding.isnot(None))
            .values(embedding=None, embedding_text=None)
            .returning(table.c.id)
        )
        cleared += len(result)
    return cleared


def embedded_counts(*, labels: Iterable[str] | None = None) -> dict[str, int]:
    """Embedded row count per label. Used by tests and the backfill check."""
    counts: dict[str, int] = {}
    for label in labels or LABEL_TABLES:
        table = LABEL_TABLES.get(label)
        if table is None:
            continue
        rows = store().query_read(
            select(func.count(literal_column("1"))).where(table.c.embedding.isnot(None))
        )
        counts[label] = int(next(iter(rows[0].values()))) if rows else 0
    return counts


__all__ = [
    "COLLECTION_LABELS",
    "EMBEDDING_DIMENSIONS",
    "LABEL_TABLES",
    "clear_all",
    "clear_by_database",
    "clear_by_id",
    "embedded_counts",
    "labels_for_collection",
    "search",
    "search_statement",
    "write_embeddings",
]
