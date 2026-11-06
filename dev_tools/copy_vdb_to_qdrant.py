# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Copy the pgvector collections into Qdrant.

Use it once when an existing deployment switches to
``AUTO_ONTOLOGY_VDB_BACKEND=qdrant``. Ingestion only re-embeds tables whose
catalog changed, so without a copy the Qdrant collections stay empty until the
next full ingest and compilation.

The copy reads the ``POSTGRES_*`` and ``QDRANT_*`` variables and calls no
embedding endpoint. pgvector stores the embeddings as ``halfvec`` (float16),
so the copied vectors keep float16 precision. A full ingest and compilation
writes float32 vectors instead.

Usage::

    uv run --no-sync python -m dev_tools.copy_vdb_to_qdrant
    uv run --no-sync python -m dev_tools.copy_vdb_to_qdrant --replace
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Any, Iterable, Iterator

import psycopg
from psycopg import sql

from auto_ontology.infra.postgres import get_postgres_connection_string
from auto_ontology.vdb import (
    DATA_VDB_COLLECTION,
    SEMANTIC_VDB_COLLECTION,
    VDB_SCHEMA,
    get_qdrant_client,
)
from auto_ontology.vdb.qdrant import QdrantVDB

logger = logging.getLogger("dev_tools.copy_vdb_to_qdrant")

_BATCH_SIZE = 500

#: One pgvector row: content, embedding as text, langchain_metadata,
#: database_name, label. langchain-postgres moves the declared metadata
#: columns out of ``langchain_metadata``, so the copy puts them back.
PgRow = tuple[str | None, str | list[float], dict | str | None, str | None, str | None]


def pg_row_to_record(row: PgRow) -> dict[str, Any]:
    """The record :meth:`QdrantVDB.write_to_index` takes for one pgvector row."""
    content, embedding, metadata, database_name, label = row
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    if isinstance(embedding, str):
        embedding = json.loads(embedding)
    metadata = dict(metadata or {})
    if database_name is not None:
        metadata["database_name"] = database_name
    if label is not None:
        metadata["label"] = label
    metadata["embedding"] = embedding
    return {
        "text": content,
        "document_type": metadata.pop("document_type", None) or "text",
        "metadata": metadata,
    }


def copy_rows(rows: Iterable[PgRow], target: QdrantVDB) -> int:
    """Write *rows* to *target* in batches; return the number written."""
    written = 0
    batch: list[dict[str, Any]] = []
    for row in rows:
        batch.append(pg_row_to_record(row))
        if len(batch) >= _BATCH_SIZE:
            written += target.write_to_index(batch)
            batch = []
    if batch:
        written += target.write_to_index(batch)
    return written


def _pg_rows(connection_string: str, collection: str) -> Iterator[PgRow]:
    """Stream every row of ``vdb.<collection>``; nothing when it does not exist."""
    with psycopg.connect(connection_string) as conn:
        exists = conn.execute(
            "SELECT 1 FROM information_schema.tables "
            "WHERE table_schema = %s AND table_name = %s",
            (VDB_SCHEMA, collection),
        ).fetchone()
        if not exists:
            logger.info("pgvector collection %s not found, skipping", collection)
            return
        # A named cursor streams from the server instead of loading every
        # 2048-dimension vector into memory at once.
        with conn.cursor(name=f"copy_{collection}") as cur:
            cur.itersize = _BATCH_SIZE
            cur.execute(
                sql.SQL(
                    "SELECT content, embedding::text, langchain_metadata, "
                    "database_name, label FROM {table}"
                ).format(table=sql.Identifier(VDB_SCHEMA, collection))
            )
            yield from cur


def check_targets(targets: list[QdrantVDB], *, replace: bool) -> None:
    """Refuse before any copy starts when a target already holds points.

    Checking every target first means a refusal never leaves one collection
    copied and the next one untouched.
    """
    occupied: dict[str, int] = {}
    for target in targets:
        target.create_index()
        count = target.client.count(
            collection_name=target.collection_name, exact=True
        ).count
        if count:
            occupied[target.collection_name] = count
    if occupied and not replace:
        listing = ", ".join(f"{name} ({n} points)" for name, n in occupied.items())
        raise SystemExit(
            f"Qdrant collections already hold points: {listing}. "
            "Run again with --replace to clear them first."
        )


def copy_collection(
    connection_string: str, target: QdrantVDB, *, replace: bool = False
) -> int:
    """Copy one pgvector collection into the Qdrant collection of the same name."""
    if replace:
        target.delete_all()
    written = copy_rows(_pg_rows(connection_string, target.collection_name), target)
    logger.info(
        "Copied %d points into Qdrant collection %s", written, target.collection_name
    )
    return written


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Clear each Qdrant collection before the copy.",
    )
    args = parser.parse_args(argv)

    connection_string = get_postgres_connection_string()
    client = get_qdrant_client()
    targets = [
        QdrantVDB(client=client, collection_name=name)
        for name in (DATA_VDB_COLLECTION, SEMANTIC_VDB_COLLECTION)
    ]
    check_targets(targets, replace=args.replace)
    for target in targets:
        copy_collection(connection_string, target, replace=args.replace)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    main()
