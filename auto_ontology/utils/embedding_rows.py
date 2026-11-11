# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Turn ``(tables_df, columns_df)`` into embedding-ready rows.

This is the text that gets embedded and vector-searched, so the templates
below are effectively part of the retrieval contract: changing them changes
every stored embedding, and a query embedded against the new wording will
score poorly against rows ingested under the old one. Re-embed the catalog
after editing :func:`build_table_text` or :func:`build_column_text`.

The row dicts produced here are consumed by the embed step
(:func:`auto_ontology.utils.embedding.batch_embed_chunks` or
:func:`~auto_ontology.utils.embedding.embed_rows`) and then written by
:func:`auto_ontology.vdb.records.write_embedded_rows`.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable

import pandas as pd

from auto_ontology.catalog.constants import Labels

logger = logging.getLogger(__name__)

__all__ = [
    "CatalogEmbeddingRowsOp",
    "build_column_text",
    "build_embed_row",
    "build_table_text",
]


class CatalogEmbeddingRowsOp:
    """Build an embedding-ready DataFrame from ``(tables_df, columns_df)``.

    Expected input: a 2-tuple ``(tables_df, columns_df)``. Both DataFrames
    carry the post-ingest catalog UUIDs; ``tables_df`` is keyed by ``id``
    (table UUID) with at least ``table_name``, ``table_schema`` and
    ``description`` columns, and ``columns_df`` carries one row per column
    with ``id``, ``table_name``, ``column_name``, ``data_type``,
    ``description`` and ``sample_values``. Multiple schemas can be
    concatenated into the same pair — the ``table_schema`` column on each
    table row keeps them distinguishable.

    Output columns: ``text, path, page_number, metadata``.
    Two row types are produced:

    * one ``Table`` row per table, whose ``text`` joins the table description
      with a compact list of its columns; and
    * one ``Column`` row per column.
    """

    def __init__(self, *, database_name: str) -> None:
        self._database_name = database_name

    def __call__(self, data: Any) -> pd.DataFrame:
        return self.process(data)

    def process(self, data: Any) -> pd.DataFrame:
        if not (isinstance(data, tuple) and len(data) == 2):
            raise TypeError(
                f"CatalogEmbeddingRowsOp expected a (tables_df, columns_df) tuple for "
                f"database {self._database_name!r}, got {type(data).__name__}."
            )

        tables_df, columns_df = data
        rows = list(self._build_rows(tables_df, columns_df))
        if rows:
            return pd.DataFrame(rows)
        return pd.DataFrame(columns=["text", "path", "page_number", "metadata"])

    def _build_rows(
        self, tables_df: pd.DataFrame, columns_df: pd.DataFrame
    ) -> Iterable[dict[str, Any]]:
        # Index columns by (schema, table_name) so duplicate table names across
        # different schemas (e.g. two "users" tables in two schemas) don't get
        # their columns merged into one bucket — which would both bloat each
        # table's text and double-emit every column row per duplicate.
        columns_by_table: dict[tuple[str, str], list[Any]] = {}
        for _, col in columns_df.iterrows():
            key = (
                str(col.get("table_schema", "")).lower(),
                str(col.get("table_name", "")).lower(),
            )
            columns_by_table.setdefault(key, []).append(col)

        rows: list[dict[str, Any]] = []
        for _, table in tables_df.iterrows():
            table_id = str(table.get("id", ""))
            table_name = str(table.get("table_name", ""))
            table_description = (
                "" if pd.isna(v := table.get("description")) else str(v).strip()
            )
            schema_name = str(table.get("table_schema", ""))
            columns = columns_by_table.get(
                (schema_name.lower(), table_name.lower()), []
            )

            table_text = build_table_text(
                table_name=table_name,
                table_description=table_description,
                columns=columns,
                schema_name=schema_name,
                database_name=self._database_name,
            )
            rows.append(
                build_embed_row(
                    text=table_text,
                    node_id=table_id,
                    label=Labels.TABLE,
                    name=table_name,
                    schema_name=schema_name,
                    database_name=self._database_name,
                )
            )

            for column in columns:
                column_id = str(column.get("id", ""))
                column_name = str(column.get("column_name", ""))
                data_type = (
                    "" if pd.isna(v := column.get("data_type")) else str(v).strip()
                )
                column_description = (
                    "" if pd.isna(v := column.get("description")) else str(v).strip()
                )
                sample_values = (column.get("sample_values") or [])[:5]
                column_text = build_column_text(
                    column_name=column_name,
                    column_description=column_description,
                    data_type=data_type,
                    sample_values=sample_values,
                    schema_name=schema_name,
                    table_name=table_name,
                    database_name=self._database_name,
                )
                rows.append(
                    build_embed_row(
                        text=column_text,
                        node_id=column_id,
                        label=Labels.COLUMN,
                        name=column_name,
                        schema_name=schema_name,
                        database_name=self._database_name,
                    )
                )
        return rows


# ── Text templates ───────────────────────────────────────────────────────────


def build_table_text(
    *,
    table_name: str,
    table_description: str,
    columns: list[Any],
    schema_name: str,
    database_name: str,
) -> str:
    """Build the embedding text for a table.

    Returns just the text string; the caller is responsible for wrapping it
    in an embed-row dict via :func:`build_embed_row`.
    """
    text = (
        f"database_name: {database_name}"
        f", schema_name: {schema_name}"
        f", table_name: {table_name}"
    )
    if table_description:
        text += f", table_description: {table_description}"

    column_pieces: list[str] = []
    for column in columns:
        column_name = column.get("column_name", "")
        data_type = "" if pd.isna(v := column.get("data_type")) else str(v).strip()
        piece = f"{{name: {column_name}, data_type: {data_type}"

        column_description = (
            "" if pd.isna(v := column.get("description")) else str(v).strip()
        )
        if column_description:
            piece += f", description: {column_description}"
        piece += "}"
        column_pieces.append(piece)

    text += f", columns: {','.join(column_pieces)}"
    return text


def build_column_text(
    *,
    column_name: str,
    column_description: str,
    data_type: str,
    sample_values: list[Any],
    table_name: str,
    schema_name: str,
    database_name: str,
) -> str:
    """Build the embedding text for a column.

    Returns just the text string; the caller is responsible for wrapping it
    in an embed-row dict via :func:`build_embed_row`.
    """
    text = (
        f"database_name: {database_name}"
        f", schema_name: {schema_name}"
        f", table_name: {table_name}"
        f", column_name: {column_name}"
        f", data_type: {data_type}"
    )
    if column_description:
        text += f", column_description: {column_description}"
    if len(sample_values) > 0:
        text += f", sample_values: {', '.join(str(x) for x in sample_values)}"
    return text


def build_embed_row(
    *,
    text: str,
    node_id: str | None,
    label: str,
    name: str,
    schema_name: str,
    database_name: str | None,
) -> dict[str, Any]:
    """Wrap embedding *text* in the row shape the embed → VDB path expects."""
    path = f"auto_ontology:{node_id}" if node_id else "auto_ontology:unknown"
    # Retrieval reads the top-level copies: the store keeps the whole metadata
    # dict. The copies under content_metadata are kept so rows written now look
    # like the ones already stored.
    fields = {
        "id": node_id,
        "label": label,
        "name": name,
        "source_path": path,
        "schema_name": schema_name,
        "database_name": database_name,
    }
    return {
        "text": text.strip(),
        "path": path,
        "page_number": -1,
        "metadata": {
            **fields,
            "content_metadata": dict(fields),
        },
    }
