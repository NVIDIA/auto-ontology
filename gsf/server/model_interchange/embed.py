# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Accumulate and flush VDB embed rows during GSF model import."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from nemo_retriever.models.inference.main_text_embed import (
    TextEmbeddingConfig,
    create_text_embeddings_for_df,
)
from nemo_retriever.operators.vdb import IngestVdbOperator
from gsf.catalog.constants import Labels
from gsf.utils.embedding_rows import (
    build_column_text,
    build_embed_row,
    build_table_text,
)

from gsf.semantic.constants import LABEL_SQL_ATTRIBUTE
from gsf.semantic.embed import _build_rows
from gsf.utils import get_embed_params
from gsf.utils.model_config import resolve
from gsf.vdb import get_data_vdb, get_semantic_vdb

logger = logging.getLogger(__name__)


@dataclass
class ImportEmbedBuffer:
    """Pre-embed rows accumulated during import; flushed once per VDB collection."""

    data_rows: list[dict[str, Any]] = field(default_factory=list)
    semantic_rows: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class ColumnCatalogMeta:
    """Catalog context for a YAML column id used when building embed docs."""

    name: str
    description: str
    data_type: str
    sample_values: list[Any]
    is_unique: bool
    table_yaml_id: str
    table_name: str
    schema_name: str
    database_name: str


def build_column_data_row(
    *,
    live_id: str,
    column_name: str,
    column_description: str,
    data_type: str,
    sample_values: list[Any],
    table_name: str,
    schema_name: str,
    database_name: str,
) -> dict[str, Any]:
    """Build a data-layer embed row for a Column node."""
    text = build_column_text(
        column_name=column_name,
        column_description=column_description,
        data_type=data_type,
        sample_values=sample_values[:5],
        table_name=table_name,
        schema_name=schema_name,
        database_name=database_name,
    )
    return build_embed_row(
        text=text,
        node_id=live_id,
        label=Labels.COLUMN,
        name=column_name,
        schema_name=schema_name,
        database_name=database_name,
    )


def build_table_data_row(
    *,
    live_id: str,
    table_name: str,
    table_description: str,
    schema_name: str,
    database_name: str,
    columns: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a data-layer embed row for a Table node."""
    text = build_table_text(
        table_name=table_name,
        table_description=table_description,
        columns=columns,
        schema_name=schema_name,
        database_name=database_name,
    )
    return build_embed_row(
        text=text,
        node_id=live_id,
        label=Labels.TABLE,
        name=table_name,
        schema_name=schema_name,
        database_name=database_name,
    )


def build_term_semantic_rows(
    *,
    database_name: str,
    live_id: str,
    name: str,
    description: str,
    schema_names: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Build semantic-layer embed rows for a Term."""
    return _build_rows(
        database_name,
        {
            "name": name,
            "description": description,
            "id": live_id,
            "schema_names": schema_names or [],
        },
        [],
    )


def build_column_attribute_semantic_rows(
    *,
    database_name: str,
    live_id: str,
    name: str,
    description: str,
    term_name: str,
    source_column: str,
    table_id: str,
    table_name: str,
    is_unique: bool,
    sample_values: list[Any] | None,
    schema_name: str | None = None,
) -> list[dict[str, Any]]:
    """Build semantic-layer embed rows for a ColumnAttribute."""
    return _build_rows(
        database_name,
        {},
        [
            {
                "name": name,
                "description": description,
                "term_name": term_name,
                "source_column": source_column,
                "table_id": table_id,
                "table_name": table_name,
                "is_unique": is_unique,
                "sample_values": sample_values,
                "id": live_id,
                "schema_name": schema_name,
            },
        ],
    )


def build_sql_attribute_semantic_row(
    *,
    live_id: str,
    name: str,
    description: str,
    term_name: str,
    sql: str,
    database_name: str | None,
) -> dict[str, Any]:
    """Build a semantic-layer embed row for a SqlAttribute."""
    text = f"sql_attribute: {name}"
    if description.strip():
        text += f", description: {description}"
    if term_name:
        text += f", term: {term_name}"
    text += f", sql: {sql}"
    return _doc_to_embed_row(
        {
            "text": text,
            "name": name,
            "label": LABEL_SQL_ATTRIBUTE,
            "id": live_id,
        },
        database_name,
    )


def build_custom_analysis_semantic_row(
    *,
    live_id: str,
    name: str,
    description: str,
    sql: str,
    database_name: str | None,
) -> dict[str, Any]:
    """Build a semantic-layer embed row for a CustomAnalysis."""
    text = f"custom_analysis: {name}"
    if description.strip():
        text += f", description: {description}"
    if sql.strip():
        text += f", sql: {sql}"
    return _doc_to_embed_row(
        {
            "text": text,
            "name": name,
            "label": Labels.CUSTOM_ANALYSIS,
            "id": live_id,
        },
        database_name,
    )


def _doc_to_embed_row(
    item: dict[str, Any],
    database_name: str | None,
) -> dict[str, Any]:
    node_id = item.get("id")
    path = f"gsf:{node_id}" if node_id is not None else "gsf:unknown"
    tabular_fields = {
        "id": node_id,
        "label": item.get("label", ""),
        "name": item.get("name", ""),
        "source_path": path,
        "database_name": database_name,
    }
    return {
        "text": (item.get("text") or "").strip(),
        "_embed_modality": "text",
        "path": path,
        "page_number": -1,
        "metadata": {**tabular_fields, "content_metadata": dict(tabular_fields)},
    }


def _embed_and_ingest_rows(rows: list[dict[str, Any]], vdb: Any) -> int:
    if not rows:
        return 0
    embed_params = get_embed_params()
    embedded, _ = create_text_embeddings_for_df(
        pd.DataFrame(rows),
        task_config={
            "api_key": embed_params.api_key,
            "endpoint_url": embed_params.embed_invoke_url,
            "model_name": embed_params.model_name,
        },
        transform_config=TextEmbeddingConfig(
            embed_modality=embed_params.embed_modality,
        ),
    )
    with_embeddings = [
        row
        for row in embedded.to_dict(orient="records")
        if (row.get("metadata") or {}).get("embedding")
    ]
    if not with_embeddings:
        raise RuntimeError(
            f"Embedding step produced 0/{len(embedded)} rows with embeddings.",
        )
    IngestVdbOperator(vdb=vdb)(with_embeddings)
    return len(with_embeddings)


def flush_import_embeddings(buffer: ImportEmbedBuffer) -> dict[str, Any]:
    """Embed and ingest accumulated rows into both VDB collections (no reset)."""
    if not resolve("EMBED", "API_KEY"):
        logger.warning(
            "EMBED API key not set — skipping VDB flush after model import",
        )
        return {
            "skipped": True,
            "reason": "embed_api_key_missing",
            "data_rows": 0,
            "semantic_rows": 0,
        }

    result: dict[str, Any] = {"skipped": False}
    if buffer.data_rows:
        result["data_rows"] = _embed_and_ingest_rows(
            buffer.data_rows,
            get_data_vdb(),
        )
    else:
        result["data_rows"] = 0

    if buffer.semantic_rows:
        result["semantic_rows"] = _embed_and_ingest_rows(
            buffer.semantic_rows,
            get_semantic_vdb(),
        )
    else:
        result["semantic_rows"] = 0

    return result
