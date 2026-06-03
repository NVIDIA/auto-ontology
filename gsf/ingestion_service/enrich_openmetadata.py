# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pull descriptions from OpenMetadata and apply them to the catalog graph.

OpenMetadata is treated as a metadata connector (see
:class:`gsf.connectors.openmetadata.OpenMetadataConnector`) rather than a SQL
source: we read table/column descriptions over its REST API and write them onto
the matching ``Table``/``Column`` nodes in Neo4j (matched by database/schema/
table/column name).

Usage::

    uv run python -m gsf.ingestion_service.enrich_openmetadata [options]

Options:
    --overwrite       Replace existing descriptions (default: only fill empty).
    --no-embeddings   Skip the pgvector embedding refresh.
    --dry-run         Print what would be applied; do not write to Neo4j.
"""

from __future__ import annotations

import argparse
import logging

from gsf.server.env import load_server_env

logger = logging.getLogger("gsf.ingestion_service.enrich_openmetadata")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    load_server_env()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing descriptions (default: only fill empty ones).",
    )
    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Skip refreshing pgvector embeddings for changed nodes.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print fetched descriptions without writing to the graph.",
    )
    args = parser.parse_args()

    from gsf.connectors.openmetadata import OpenMetadataConnector

    connector = OpenMetadataConnector.from_env()
    descriptions = connector.get_descriptions()

    table_count = int((descriptions["level"] == "table").sum())
    column_count = int((descriptions["level"] == "column").sum())
    logger.info(
        "fetched %d descriptions from OpenMetadata (%d table, %d column)",
        len(descriptions),
        table_count,
        column_count,
    )

    if args.dry_run:
        if descriptions.empty:
            logger.info("dry-run: no descriptions found")
        else:
            print(descriptions.to_string(index=False))
        return

    from gsf.server.datasources.dal import apply_descriptions

    summary = apply_descriptions(
        descriptions,
        overwrite=args.overwrite,
        refresh_embeddings=not args.no_embeddings,
    )
    logger.info("done: %s", summary)


if __name__ == "__main__":
    main()
