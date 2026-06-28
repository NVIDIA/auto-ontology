# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data Access Layer — catalog datasource queries and VDB re-embedding.

All direct Neo4j calls live in gsf/neo4j/datasources.py.
This module only keeps the VDB orchestration: update_node_properties
and its helpers that mix Neo4j reads with pgvector upserts.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from nemo_retriever.tabular_data.ingestion.model.reserved_words import Labels
from nemo_retriever.tabular_data.operators.tabular_fetch_embeddings_operator import (
    TabularFetchEmbeddingsOp,
)

from gsf.neo4j.datasources import (
    get_parent_table_id_for_column,
    get_tables_and_columns_by_node_ids,
    list_columns_for_table,
    list_databases,
    list_schemas_for_database,
    list_tables_for_schema,
    patch_catalog_node,
)

logger = logging.getLogger(__name__)

__all__ = [
    "list_databases",
    "list_schemas_for_database",
    "list_tables_for_schema",
    "list_columns_for_table",
    "get_parent_table_id_for_column",
    "update_node_properties",
]


def update_node_properties(
    node_id: str,
    properties: dict[str, Any],
) -> dict[str, Any] | None:
    """Update properties on any catalog node matched by ``id``.

    Returns ``{id, ...updated_fields}``. When the patch touches fields that
    feed retrieval text (``Table``/``Column`` description or column
    ``sample_values``), stale pgvector rows are deleted and re-appended.
    """
    if not properties:
        return None

    patched = patch_catalog_node(node_id, properties)
    if not patched:
        return None

    node_props = patched["props"]
    result = {"id": patched["id"], **{k: node_props.get(k) for k in properties}}

    reembed_ids = _get_node_ids_for_embedding_update(
        node_id=node_id,
        label=patched["label"],
        properties=properties,
    )
    if reembed_ids:
        _refresh_vdb_embeddings(reembed_ids)

    return result


def _get_node_ids_for_embedding_update(
    *,
    node_id: str,
    label: str,
    properties: dict[str, Any],
) -> list[str]:
    """Return Neo4j node ids whose pgvector rows must be refreshed for *properties*.

    * ``Column`` + ``description`` → column and parent ``Table`` (table text
      lists column descriptions).
    * ``Column`` + ``sample_values`` only → column only.
    * ``Table`` + ``description`` → table only.
    """
    has_description = "description" in properties
    has_sample_values = "sample_values" in properties

    if label == Labels.TABLE:
        return [node_id] if has_description else []

    if label != Labels.COLUMN:
        return []

    if has_description:
        table_id = get_parent_table_id_for_column(node_id)
        targets = [node_id]
        if table_id is not None:
            targets.append(table_id)
        else:
            logger.warning(
                "Column %r has no parent Table in Neo4j; re-embedding column only.",
                node_id,
            )
        return targets

    if has_sample_values:
        return [node_id]

    return []


def _refresh_vdb_embeddings(node_ids: list[str]) -> None:
    """Delete stale VDB rows, re-embed, and append Table/Column rows."""
    from gsf.utils import get_embed_params
    from gsf.vdb import get_data_vdb
    from nemo_retriever.text_embed.runtime import embed_text_main_text_embed
    from nemo_retriever.vdb import IngestVdbOperator

    unique_ids = set(dict.fromkeys(node_ids))

    tables_df, columns_df, database_name = get_tables_and_columns_by_node_ids(node_ids)
    if tables_df.empty and columns_df.empty:
        logger.info(
            "No Table/Column rows found for node_ids=%r; skipping VDB upsert.",
            node_ids,
        )
        return

    embed_df = TabularFetchEmbeddingsOp(database_name=database_name).process(
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

    embed_params = get_embed_params()
    embedded = embed_text_main_text_embed(
        pd.DataFrame(records),
        model_name=embed_params.model_name,
        embed_invoke_url=embed_params.embed_invoke_url,
        api_key=embed_params.api_key,
        embed_modality=embed_params.embed_modality,
    )
    rows = [
        row
        for row in embedded.to_dict(orient="records")
        if (row.get("metadata") or {}).get("embedding")
    ]
    if not rows:
        raise RuntimeError(
            f"Embedding step produced 0/{len(embedded)} tabular rows with embeddings."
        )

    vdb = get_data_vdb()
    for nid in unique_ids:
        vdb.delete_by_id(nid)
    IngestVdbOperator(vdb=vdb)(rows)
