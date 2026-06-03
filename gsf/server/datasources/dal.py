# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data Access Layer — Neo4j driver, catalog graph, and datasource queries."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from nemo_retriever.graph.tabular_fetch_embeddings_operator import (
    TabularFetchEmbeddingsOp,
)
from nemo_retriever.tabular_data.ingestion.model.reserved_words import Edges, Labels
from nemo_retriever.tabular_data.neo4j import get_neo4j_conn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Graph queries (public API for routers / services)
# ---------------------------------------------------------------------------


def list_databases() -> list[dict[str, Any]]:
    """Return Database rows with schema counts only; ``schemas`` is empty for lazy trees."""
    neo4j_conn = get_neo4j_conn()

    rows = neo4j_conn.query_read(
        f"""
        MATCH (db:{Labels.DB})-[:{Edges.CONTAINS}]->(s:{Labels.SCHEMA})
        RETURN db.id AS id, db.name AS name, db.description AS description,
               count(s) AS schema_count
        ORDER BY name
        """,
    )

    return [
        {
            "id": r["id"],
            "name": r["name"],
            "description": r["description"],
            "num_of_schemas": int(r["schema_count"]),
            "schemas": [],
        }
        for r in rows
    ]


def list_schemas_for_database(db_id: str) -> dict[str, Any] | None:
    """Return schemas_count and a list of schema summaries for a database.

    Returns a dict with ``schemas_count`` and ``schemas`` — a list of
    ``{id, schema_name, tables_count}`` dicts.

    Returns ``None`` if no ``Database`` matches ``db_id``.
    """
    neo4j_conn = get_neo4j_conn()
    rows = neo4j_conn.query_read(
        f"""
        MATCH (db:{Labels.DB} {{id: $db_id}})-[:{Edges.CONTAINS}]->
              (s:{Labels.SCHEMA})-[:{Edges.CONTAINS}]->(t:{Labels.TABLE})
        WITH s.id AS id, s.name AS schema_name, s.description AS description,
             count(t) AS tables_count
        ORDER BY schema_name
        WITH collect({{id: id, schema_name: schema_name,
                      description: description,
                      tables_count: tables_count}}) AS schemas
        RETURN size(schemas) AS schemas_count, schemas
        """,
        {"db_id": db_id},
    )

    record = rows[0]
    return {
        "schemas_count": record["schemas_count"],
        "schemas": [dict(s) for s in record["schemas"]],
    }


def list_tables_for_schema(
    schema_id: str,
    *,
    database_name: str | None = None,
) -> list[dict[str, Any]]:
    """Return Table payloads with column counts for a given schema.

    Each table dict contains ``database_name``, ``schema_name``, ``name``,
    and ``columns_count``.
    """
    neo4j_conn = get_neo4j_conn()
    rows = neo4j_conn.query_read(
        f"""
        MATCH (db:{Labels.DB})-[:{Edges.CONTAINS}]->
              (s:{Labels.SCHEMA} {{id: $schema_id}})-[:{Edges.CONTAINS}]->
              (t:{Labels.TABLE})-[:{Edges.CONTAINS}]->(c:{Labels.COLUMN})
        RETURN t.id AS id,
               t.name AS name,
               db.name AS database_name,
               s.name AS schema_name, t.description AS description,
               count(c) AS columns_count
        ORDER BY name
        """,
        {
            "schema_id": schema_id,
            "database_name": database_name,
        },
    )

    return rows


def list_columns_for_table(table_id: str) -> dict[str, Any] | None:
    """Return a table dict with nested columns, or None if the table is missing.

    Returns ``table_name``, ``schema_name``, ``database_name`` (all from the Table
    node), ``columns_count``, and ``columns`` — a list of
    ``{ordinal_position, column_name, data_type}`` dicts.
    """
    neo4j_conn = get_neo4j_conn()
    rows = neo4j_conn.query_read(
        f"""
        MATCH (db:{Labels.DB})-[:{Edges.CONTAINS}]->(s:{Labels.SCHEMA})-[:{Edges.CONTAINS}]->
              (t:{Labels.TABLE} {{id: $table_id}})-[:{Edges.CONTAINS}]->(c:{Labels.COLUMN})
        WITH t, c, s, db ORDER BY c.ordinal_position
        WITH t, s, db, collect({{
                 id: c.id,
                 ordinal_position: c.ordinal_position,
                 column_name: c.name,
                 data_type: c.data_type,
                 description: c.description,
                 sample_values: c.sample_values
             }}) AS columns
        RETURN t.name AS table_name,
               s.name AS schema_name,
               db.name AS database_name,
               size(columns) AS columns_count,
               columns
        """,
        {"table_id": table_id},
    )

    if not rows:
        return None
    return rows[0]


def update_node_properties(
    node_id: str,
    properties: dict[str, Any],
) -> dict[str, Any] | None:
    """Update properties on any catalog node matched by ``id``.

    Returns ``{id, ...updated_fields}``. When the patch touches fields that
    feed retrieval text (``Table``/``Column`` description or column
    ``sample_values``), stale pgvector rows are deleted and re-appended
    (same pattern as :func:`custom_analyses.dal.update_custom_analysis`).
    """
    if not properties:
        return None

    neo4j_conn = get_neo4j_conn()

    rows = neo4j_conn.query_write(
        f"""
        MATCH (n:{Labels.DB}|{Labels.SCHEMA}|{Labels.TABLE}|{Labels.COLUMN}
              {{id: $node_id}})
        SET n += $props
        RETURN n.id AS id, labels(n)[0] AS label, properties(n) AS props
        """,
        {"node_id": node_id, "props": properties},
    )
    if not rows:
        return None

    node_props = dict(rows[0]["props"])
    result = {"id": rows[0]["id"], **{k: node_props.get(k) for k in properties}}

    reembed_ids = _get_node_ids_for_embedding_update(
        node_id=node_id,
        label=rows[0]["label"],
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


def get_parent_table_id_for_column(column_id: str) -> str | None:
    rows = get_neo4j_conn().query_read(
        f"""
        MATCH (t:{Labels.TABLE})-[:{Edges.CONTAINS}]->(c:{Labels.COLUMN} {{id: $column_id}})
        RETURN t.id AS table_id
        LIMIT 1
        """,
        {"column_id": column_id},
    )
    return rows[0]["table_id"] if rows else None


def _refresh_vdb_embeddings(node_ids: list[str]) -> None:
    """Delete stale VDB rows, re-embed, and append Table/Column rows."""
    from gsf.ingestion_service.ingest import EMBED_PARAMS
    from gsf.vdb import get_vdb
    from nemo_retriever.text_embed.runtime import embed_text_main_text_embed
    from nemo_retriever.vdb import IngestVdbOperator

    unique_ids = set(dict.fromkeys(node_ids))
    vdb = get_vdb()
    for nid in unique_ids:
        vdb.delete_by_id(nid)

    tables_df, columns_df, database_name = _get_tables_and_columns_by_node_ids(
        node_ids,
    )
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

    embedded = embed_text_main_text_embed(
        pd.DataFrame(records),
        model_name=EMBED_PARAMS.model_name,
        embed_invoke_url=EMBED_PARAMS.embed_invoke_url,
        api_key=EMBED_PARAMS.api_key,
        embed_modality=EMBED_PARAMS.embed_modality,
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

    IngestVdbOperator(vdb=vdb)(rows)


def _get_tables_and_columns_by_node_ids(
    node_ids: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Load Table/Column rows from Neo4j as dataframes for :class:`TabularFetchEmbeddingsOp`.

    When a table id is included, all of its columns are loaded so table
    embedding text matches full ingest (not only the column row being edited).
    """
    conn = get_neo4j_conn()
    columns_df = pd.DataFrame(
        conn.query_read(
            f"""
            UNWIND $ids AS id
            MATCH (t:{Labels.TABLE})-[:{Edges.CONTAINS}]->(c:{Labels.COLUMN})
            WHERE t.id = id OR c.id = id
            RETURN DISTINCT
                   c.id AS id,
                   t.name AS table_name,
                   t.schema_name AS table_schema,
                   c.name AS column_name,
                   c.data_type AS data_type,
                   c.description AS description,
                   c.sample_values AS sample_values,
                   t.db_name AS db_name
            """,
            {"ids": node_ids},
        ),
    )
    tables_df = pd.DataFrame(
        conn.query_read(
            f"""
            UNWIND $ids AS id
            MATCH (t:{Labels.TABLE} {{id: id}})
            RETURN t.id AS id,
                   t.name AS table_name,
                   t.schema_name AS table_schema,
                   t.description AS description,
                   t.db_name AS db_name
            """,
            {"ids": node_ids},
        ),
    )

    database_name = ""
    if not tables_df.empty:
        database_name = str(tables_df.iloc[0].get("db_name") or "")
    elif not columns_df.empty:
        database_name = str(columns_df.iloc[0].get("db_name") or "")

    return tables_df, columns_df, database_name
