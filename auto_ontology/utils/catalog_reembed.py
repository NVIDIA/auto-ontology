# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Refresh the data-VDB rows of catalog Tables and Columns after an edit."""

from __future__ import annotations

import logging
from typing import Any

from auto_ontology.dal.datasources import fetch_tables_and_columns_by_node_ids
from auto_ontology.utils.embedding import embed_rows, get_embed_params
from auto_ontology.utils.embedding_rows import CatalogEmbeddingRowsOp
from auto_ontology.vdb import get_data_vdb
from auto_ontology.vdb.records import write_embedded_rows

logger = logging.getLogger(__name__)


def refresh_catalog_embeddings(node_ids: list[str]) -> None:
    """Delete stale VDB rows, re-embed, and append Table/Column rows."""
    unique_ids = set(dict.fromkeys(node_ids))

    tables_df, columns_df, database_name = fetch_tables_and_columns_by_node_ids(
        node_ids
    )
    if tables_df.empty and columns_df.empty:
        logger.info(
            "No Table/Column rows found for node_ids=%r; skipping VDB upsert.",
            node_ids,
        )
        return

    embed_df = CatalogEmbeddingRowsOp(database_name=database_name).process(
        (tables_df, columns_df),
    )
    if embed_df.empty:
        return

    def _row_id(row: dict[str, Any]) -> str | None:
        meta = row.get("metadata") or {}
        node_id = meta.get("id")
        return str(node_id) if node_id is not None else None

    records = embed_df.to_dict(orient="records")
    records = [r for r in records if _row_id(r) in unique_ids]
    if not records:
        return

    # Embedded before anything is deleted, so a failed embed leaves the old rows.
    rows = embed_rows(records, get_embed_params(), what="tabular")

    vdb = get_data_vdb()
    for nid in unique_ids:
        vdb.delete_by_id(nid)
    write_embedded_rows(vdb, rows)
