# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Neo4j data access for catalog nodes: Database, Schema, Table, Column.

Contains only functions that call ``get_neo4j_conn()`` directly.

All read functions use the ``fetch_*`` prefix.
Write functions use ``patch_*``, ``store_*``, or ``apply_*``.

Non-Neo4j helpers that call these functions remain in their original locations:
  - get_schemas_by_ids  →  retrieval/data_access/graph_schemas.py
  - build_tables_index  →  semantic/loaders.py
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pandas as pd

from nemo_retriever.tabular_data.ingestion.model.reserved_words import Edges, Labels
from nemo_retriever.tabular_data.neo4j import get_neo4j_conn

from gsf.semantic.constants import REL_REPRESENTS

logger = logging.getLogger(__name__)

_ALLOWED_NODE_LABELS = frozenset(Labels.LIST_OF_ALL)


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------


def fetch_databases() -> list[dict[str, Any]]:
    """Return Database rows with schema counts only; ``schemas`` is empty for lazy trees."""
    rows = get_neo4j_conn().query_read(
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


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def fetch_schemas_for_database(db_id: str) -> dict[str, Any] | None:
    """Return schemas_count and a list of schema summaries for a database.

    Returns a dict with ``schemas_count`` and ``schemas`` — a list of
    ``{id, schema_name, tables_count}`` dicts.

    Returns ``None`` if no ``Database`` matches ``db_id``.
    """
    rows = get_neo4j_conn().query_read(
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
    if not rows:
        return None
    record = rows[0]
    return {
        "schemas_count": record["schemas_count"],
        "schemas": [dict(s) for s in record["schemas"]],
    }


def fetch_all_schema_ids() -> list[str]:
    """Return all Schema node IDs."""
    return [
        r["schema_id"]
        for r in get_neo4j_conn().query_read(
            f"MATCH (s:{Labels.SCHEMA}) RETURN s.id AS schema_id",
        )
    ]


def fetch_schemas_by_ids(
    relevant_schemas_ids: list | None = None,
) -> list[dict[str, str]]:
    """Return column-level rows for the given schema IDs (all schemas when empty)."""
    schema_ids = relevant_schemas_ids or []
    result = get_neo4j_conn().query_read(
        f"""
        MATCH (db:{Labels.DB})-[:{Edges.CONTAINS}]->(schema:{Labels.SCHEMA})
              -[:{Edges.CONTAINS}]->(table:{Labels.TABLE})
              -[:{Edges.CONTAINS}]->(column:{Labels.COLUMN})
        WHERE size($schema_ids) = 0
           OR schema.id IN $schema_ids
        RETURN collect({{
            column_name:   column.name,
            column_id:     column.id,
            table_name:    table.name,
            table_id:      table.id,
            database_name: db.name,
            table_schema:  schema.name,
            data_type:     column.data_type
        }}) AS data
        """,
        {"schema_ids": schema_ids},
    )
    return result[0]["data"] if result else []


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------

_FETCH_TABLES_QUERY = f"""
MATCH (s:{Labels.SCHEMA})-[:{Edges.CONTAINS}]->(t:{Labels.TABLE})
OPTIONAL MATCH (t)<-[:{Edges.SQL}]-(sql:{Labels.SQL})
WITH t, s, count(DISTINCT sql) AS query_count
RETURN t.id AS id,
       t.name AS name,
       s.name AS schema_name,
       t.description AS description,
       t.pk as pk,
       query_count
ORDER BY query_count DESC
"""

_FETCH_TABLE_BY_ID = f"""
MATCH (t:{Labels.TABLE} {{id: $table_id}})
OPTIONAL MATCH (s:{Labels.SCHEMA})-[:{Edges.CONTAINS}]->(t)
RETURN t.id AS id,
       t.name AS name,
       coalesce(s.name, '') AS schema_name,
       t.description AS description,
       t.pk as pk
"""

_FETCH_TABLE_BY_NAME = f"""
MATCH (t:{Labels.TABLE} {{name: $name}})
OPTIONAL MATCH (s:{Labels.SCHEMA})-[:{Edges.CONTAINS}]->(t)
RETURN t.id AS id,
       t.name AS name,
       coalesce(s.name, '') AS schema_name,
       t.description AS description,
       t.pk as pk
LIMIT 1
"""

_FETCH_JOIN_NEIGHBORS = f"""
MATCH (t:{Labels.TABLE} {{id: $table_id}})-[:{Edges.JOIN}]-(other:{Labels.TABLE})
RETURN DISTINCT other.id AS id,
                other.name AS name,
                other.description AS description
"""

_FETCH_JOINS_QUERY = f"""
MATCH (t1:{Labels.TABLE})-[j:{Edges.JOIN}]->(t2:{Labels.TABLE})
RETURN t1.name AS source_table,
       t1.id AS source_table_id,
       t2.name AS target_table,
       t2.id AS target_table_id,
       j.join_columns AS join_columns
"""

_FETCH_TABLES_BY_IDS = f"""
UNWIND $table_ids AS tid
MATCH (tbl:{Labels.TABLE} {{id: tid}})
OPTIONAL MATCH (tbl)<-[:{Edges.CONTAINS}]-(sch:{Labels.SCHEMA})
OPTIONAL MATCH (tbl)-[:{Edges.CONTAINS}]->(col:{Labels.COLUMN})
WITH tbl, sch, collect({{name: col.name, data_type: col.data_type,
                         description: col.description}}) AS cols
RETURN tbl.id AS id, tbl.name AS name, tbl.description AS description,
       sch.name AS schema_name, cols
"""

_APPLY_TABLE_METADATA = f"""
UNWIND $rows AS row
MATCH (d:{Labels.DB} {{name: $database_name}})-[:{Edges.CONTAINS}]->
      (:{Labels.SCHEMA})-[:{Edges.CONTAINS}]->(t:{Labels.TABLE} {{name: row.table_name}})
SET t.description = coalesce(row.description, t.description)
"""

_APPLY_COLUMN_METADATA = f"""
UNWIND $rows AS row
MATCH (d:{Labels.DB} {{name: $database_name}})-[:{Edges.CONTAINS}]->
      (:{Labels.SCHEMA})-[:{Edges.CONTAINS}]->(t:{Labels.TABLE} {{name: row.table_name}})
      -[:{Edges.CONTAINS}]->(c:{Labels.COLUMN} {{name: row.column_name}})
SET c.description = coalesce(row.description, c.description),
    c.sample_values = coalesce(row.sample_values, c.sample_values)
"""


def fetch_tables_for_schema(
    schema_id: str,
    *,
    database_name: str
    | None = None,  # accepted for API compat; schema_id is globally unique
) -> list[dict[str, Any]]:
    """Return Table payloads with column counts for a given schema."""
    return get_neo4j_conn().query_read(
        f"""
        MATCH (db:{Labels.DB})-[:{Edges.CONTAINS}]->
              (s:{Labels.SCHEMA} {{id: $schema_id}})-[:{Edges.CONTAINS}]->
              (t:{Labels.TABLE})-[:{Edges.CONTAINS}]->(c:{Labels.COLUMN})
        RETURN t.id AS id,
               t.name AS name,
               t.table_type AS table_type,
               db.name AS database_name,
               s.name AS schema_name, t.description AS description,
               count(c) AS columns_count
        ORDER BY name
        """,
        {"schema_id": schema_id},
    )


def fetch_sorted_tables() -> list[dict[str, Any]]:
    """Return all tables ordered by query_count descending."""
    rows = get_neo4j_conn().query_read(_FETCH_TABLES_QUERY)
    return [
        {
            "id": r["id"],
            "name": r["name"],
            "schema_name": r["schema_name"],
            "description": r.get("description") or "",
            "query_count": int(r.get("query_count") or 0),
            "pk": r.get("pk") or [],
        }
        for r in rows
    ]


def fetch_table_by_id(table_id: str) -> dict[str, Any] | None:
    """Return a single Table row by id, or None if not found."""
    rows = get_neo4j_conn().query_read(_FETCH_TABLE_BY_ID, {"table_id": table_id})
    return rows[0] if rows else None


def fetch_table_by_name(name: str) -> dict[str, Any] | None:
    """Return the first Table row matching *name*, or None if not found."""
    rows = get_neo4j_conn().query_read(_FETCH_TABLE_BY_NAME, {"name": name})
    return rows[0] if rows else None


def fetch_tables_by_ids(table_ids: list[str]) -> list[dict[str, Any]]:
    """Return Table rows with nested column summaries for the given IDs."""
    if not table_ids:
        return []
    try:
        rows = get_neo4j_conn().query_read(
            _FETCH_TABLES_BY_IDS, {"table_ids": table_ids}
        )
    except Exception:
        logger.warning("fetch_tables_by_ids: Neo4j query failed", exc_info=True)
        return []
    tables = []
    for row in rows:
        tid = row.get("id")
        if not tid:
            continue
        cols = [c for c in (row.get("cols") or []) if c.get("name")]
        tables.append(
            {
                "id": tid,
                "name": row.get("name") or "",
                "description": row.get("description") or "",
                "schema_name": row.get("schema_name") or "",
                "label": "Table",
                "columns": cols,
            }
        )
    return tables


def fetch_all_tables_without_term() -> list[dict[str, Any]]:
    """Return Table nodes that have not yet been assigned a Term."""
    from nemo_retriever.tabular_data.ingestion.model.reserved_words import Edges

    return get_neo4j_conn().query_read(
        f"""
        MATCH (t:{Labels.TABLE})
        WHERE NOT (t)-[:{REL_REPRESENTS}]->()
        OPTIONAL MATCH (t)<-[:{Edges.CONTAINS}]-(sch:{Labels.SCHEMA})
        RETURN t.id AS id, t.name AS name, t.description AS description,
               sch.name AS schema_name
        ORDER BY t.name
        """
    )


def fetch_join_neighbors(table_id: str) -> list[dict[str, Any]]:
    """Return JOIN-adjacent tables (undirected), one row per neighbour."""
    return get_neo4j_conn().query_read(_FETCH_JOIN_NEIGHBORS, {"table_id": table_id})


def fetch_join_edges() -> list[dict[str, Any]]:
    """Return all JOIN edges between tables."""
    return get_neo4j_conn().query_read(_FETCH_JOINS_QUERY)


# ---------------------------------------------------------------------------
# Column
# ---------------------------------------------------------------------------

_FETCH_COLUMNS_QUERY = f"""
MATCH (t:{Labels.TABLE} {{id: $table_id}})
OPTIONAL MATCH (t)-[:{Edges.CONTAINS}]->(c:{Labels.COLUMN})
OPTIONAL MATCH (c)-[fk:{Edges.FOREIGN_KEY}]->(:{Labels.COLUMN})
RETURN c.id AS id,
       c.name AS name,
       c.data_type AS data_type,
       c.description AS description,
       c.ordinal_position AS ordinal_position,
       c.sample_values AS sample_values,
       fk IS NOT NULL AS is_foreign_key
ORDER BY c.ordinal_position
"""

_FETCH_FKS_QUERY = f"""
MATCH (t:{Labels.TABLE} {{id: $table_id}})-[:{Edges.CONTAINS}]->(src:{Labels.COLUMN})
      -[:{Edges.FOREIGN_KEY}]->(tgt:{Labels.COLUMN})<-[:{Edges.CONTAINS}]-
      (tgt_table:{Labels.TABLE})
RETURN src.name AS source_column,
       tgt.name AS target_column,
       tgt_table.name AS target_table,
       tgt_table.id AS target_table_id
"""

_FETCH_COL_TABLE_CONTEXTS = f"""
UNWIND $col_ids AS col_id
MATCH (col:{Labels.COLUMN} {{id: col_id}})<-[:{Edges.CONTAINS}]-(tbl:{Labels.TABLE})
      <-[:{Edges.CONTAINS}]-(sch:{Labels.SCHEMA})
RETURN col.id AS col_id, tbl.name AS table_name, sch.name AS schema_name
"""


def fetch_columns_for_table(table_id: str) -> dict[str, Any] | None:
    """Return a table dict with nested columns, or None if the table is missing."""
    rows = get_neo4j_conn().query_read(
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
               t.table_type AS table_type,
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


def fetch_parent_table_id_for_column(column_id: str) -> str | None:
    """Return the id of the Table that contains this Column, or None."""
    rows = get_neo4j_conn().query_read(
        f"""
        MATCH (t:{Labels.TABLE})-[:{Edges.CONTAINS}]->(c:{Labels.COLUMN} {{id: $column_id}})
        RETURN t.id AS table_id
        LIMIT 1
        """,
        {"column_id": column_id},
    )
    return rows[0]["table_id"] if rows else None


def fetch_table_context(table_id: str) -> dict[str, Any]:
    """Return columns and FK edges for one table."""
    conn = get_neo4j_conn()
    rows = conn.query_read(_FETCH_COLUMNS_QUERY, {"table_id": table_id})
    columns = [
        {
            "id": r["id"],
            "name": r["name"],
            "data_type": r["data_type"],
            "description": r.get("description"),
            "ordinal_position": r.get("ordinal_position"),
            "sample_values": r.get("sample_values"),
        }
        for r in rows
        if r.get("id") is not None
    ]
    fks = conn.query_read(_FETCH_FKS_QUERY, {"table_id": table_id})
    return {"columns": columns, "fks": fks}


def fetch_col_table_contexts(col_ids: list[str]) -> dict[str, dict[str, str]]:
    """Batch lookup: Column id → {table_name, schema_name}."""
    if not col_ids:
        return {}
    try:
        rows = get_neo4j_conn().query_read(
            _FETCH_COL_TABLE_CONTEXTS, {"col_ids": col_ids}
        )
    except Exception:
        logger.warning("fetch_col_table_contexts: Neo4j query failed", exc_info=True)
        return {}
    return {
        r["col_id"]: {
            "table_name": r.get("table_name") or "",
            "schema_name": r.get("schema_name") or "",
        }
        for r in rows
        if r.get("col_id")
    }


def store_column_sample_values(table_id: str, samples: dict[str, list]) -> None:
    """Write sample_values JSON onto Column nodes for a given table.

    Skips silently when *samples* is empty.
    """
    if not samples:
        return
    entries = [
        {"column_name": col, "sample_values": json.dumps(vals)}
        for col, vals in samples.items()
    ]
    get_neo4j_conn().query_write(
        f"""
        MATCH (t:{Labels.TABLE} {{id: $table_id}})-[:{Edges.CONTAINS}]->(col:{Labels.COLUMN})
        WHERE col.name IN [e IN $entries | e.column_name]
        WITH col,
             [e IN $entries WHERE e.column_name = col.name | e.sample_values][0]
             AS sv
        WHERE sv IS NOT NULL
        SET col.sample_values = sv
        """,
        {"table_id": table_id, "entries": entries},
    )


# ---------------------------------------------------------------------------
# Cross-entity (Table + Column batch operations)
# ---------------------------------------------------------------------------


def fetch_tables_and_columns_by_node_ids(
    node_ids: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Load Table/Column rows from Neo4j as dataframes for TabularFetchEmbeddingsOp."""
    conn = get_neo4j_conn()
    columns_df = pd.DataFrame(
        conn.query_read(
            f"""
            UNWIND $ids AS id
            MATCH (db:{Labels.DB})-[:{Edges.CONTAINS}]->(s:{Labels.SCHEMA})
                  -[:{Edges.CONTAINS}]->(t:{Labels.TABLE})-[:{Edges.CONTAINS}]->(c:{Labels.COLUMN})
            WHERE t.id = id OR c.id = id
            RETURN DISTINCT
                   c.id AS id,
                   t.name AS table_name,
                   s.name AS table_schema,
                   c.name AS column_name,
                   c.data_type AS data_type,
                   c.description AS description,
                   c.sample_values AS sample_values,
                   db.name AS database_name
            """,
            {"ids": node_ids},
        ),
    )
    tables_df = pd.DataFrame(
        conn.query_read(
            f"""
            UNWIND $ids AS id
            MATCH (db:{Labels.DB})-[:{Edges.CONTAINS}]->(s:{Labels.SCHEMA})
                  -[:{Edges.CONTAINS}]->(t:{Labels.TABLE} {{id: id}})
            RETURN t.id AS id,
                   t.name AS table_name,
                   s.name AS table_schema,
                   t.table_type AS table_type,
                   t.description AS description,
                   db.name AS database_name
            """,
            {"ids": node_ids},
        ),
    )
    database_name = ""
    if not tables_df.empty:
        database_name = str(tables_df.iloc[0].get("database_name") or "")
    elif not columns_df.empty:
        database_name = str(columns_df.iloc[0].get("database_name") or "")
    return tables_df, columns_df, database_name


def apply_metadata_batch(
    database_name: str,
    table_rows: list[dict],
    column_rows: list[dict],
) -> None:
    """Batch-write description / sample_values onto Table and Column nodes.

    *table_rows* — list of ``{table_name, description}``.
    *column_rows* — list of ``{table_name, column_name, description, sample_values}``.
    Skips silently when either list is empty.
    """
    conn = get_neo4j_conn()
    if table_rows:
        conn.query_write(
            _APPLY_TABLE_METADATA,
            {"rows": table_rows, "database_name": database_name},
        )
    if column_rows:
        conn.query_write(
            _APPLY_COLUMN_METADATA,
            {"rows": column_rows, "database_name": database_name},
        )


# ---------------------------------------------------------------------------
# Any catalog node
# ---------------------------------------------------------------------------


def patch_catalog_node(
    node_id: str,
    properties: dict[str, Any],
) -> dict[str, Any] | None:
    """Write ``properties`` onto any catalog node matched by ``id``.

    Returns ``{id, label, props}`` or ``None`` when no node matches.
    This is the pure Cypher write; callers are responsible for triggering
    any downstream VDB re-embedding.
    """
    rows = get_neo4j_conn().query_write(
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
    return {
        "id": rows[0]["id"],
        "label": rows[0]["label"],
        "props": dict(rows[0]["props"]),
    }


def fetch_node_properties_by_id(id: str, label: str | list[str]) -> dict | None:
    """Return all properties of the node with the given id and label, or None.

    Rejects unknown labels and returns None with a warning instead of raising.
    """
    labels_list = label if isinstance(label, list) else [label]
    for lbl in labels_list:
        if lbl not in _ALLOWED_NODE_LABELS:
            logger.warning(
                "Rejecting unknown label %r in fetch_node_properties_by_id", lbl
            )
            return None
    label_filter = "|".join(labels_list)
    props = get_neo4j_conn().query_read_only(
        f"""
        MATCH (n:{label_filter} {{id: $id}})
        RETURN apoc.map.setKey(properties(n), "label", labels(n)[0]) AS props
        """,
        parameters={"id": id},
    )
    return props[0]["props"] if props else None


def fetch_item_by_id(item_id: str, label: str | list[str]) -> dict | None:
    """Like ``fetch_node_properties_by_id`` but logs an error when the node is missing."""
    result = fetch_node_properties_by_id(item_id, label)
    if result is None:
        logger.error("Required item with id %r not found in graph.", item_id)
    return result
