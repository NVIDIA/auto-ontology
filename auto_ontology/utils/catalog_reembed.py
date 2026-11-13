# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Refresh the data-VDB rows of catalog Tables and Columns after an edit."""

from __future__ import annotations

import logging
from typing import Any

from auto_ontology.dal.datasources import (
    fetch_parent_table_id_for_column,
    fetch_tables_and_columns_by_node_ids,
)
from auto_ontology.utils.embedding import embed_rows, get_embed_params
from auto_ontology.utils.embedding_rows import CatalogEmbeddingRowsOp
from auto_ontology.vdb import get_data_vdb
from auto_ontology.vdb.records import replace_embedded_rows

logger = logging.getLogger(__name__)


def refresh_catalog_embeddings(node_ids: list[str]) -> None:
    """Re-embed these Table/Column rows, replacing their previous VDB rows."""
    unique_ids = set(dict.fromkeys(node_ids))

    # CatalogEmbeddingRowsOp builds a column's row only under its table, so a
    # column requested alone needs that table fetched too.
    owning_table_ids = {
        table_id
        for node_id in unique_ids
        if (table_id := fetch_parent_table_id_for_column(node_id)) is not None
    }
    tables_df, columns_df, database_name = fetch_tables_and_columns_by_node_ids(
        [*node_ids, *(owning_table_ids - unique_ids)]
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

    # A failed embed must leave the old rows. On pgvector that is embedding
    # here, before replace_embedded_rows deletes anything; on Elasticsearch
    # embed_rows passes the text through and the store writes before it
    # deletes.
    rows = embed_rows(records, get_embed_params(), what="tabular")
    written = replace_embedded_rows(get_data_vdb(), unique_ids, rows)
    # Writing nothing already raises. A partial write returns, and on
    # Elasticsearch an id with a rejected row keeps its old one, so the edit
    # would look saved while search still finds the old text.
    if written < len(rows):
        raise RuntimeError(
            f"Refreshed {written}/{len(rows)} catalog rows for {sorted(unique_ids)}; "
            "the rest keep their previous embedding."
        )
