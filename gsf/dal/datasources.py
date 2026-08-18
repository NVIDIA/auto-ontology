# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Neo4j data access for catalog nodes: Database, Schema, Table, Column.

Contains only functions that call ``graph()`` directly.

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

from gsf.dal.cypher_fragments import (
    column_description_expr,
    paging_clause,
    table_description_expr,
)
from gsf.dal.neo4j_tx import graph
from gsf.dal.users import resolve_accessible_catalog_ids, resolve_table_filter
from gsf.semantic.constants import (
    LABEL_COLUMN_ATTRIBUTE,
    LABEL_SQL_ATTRIBUTE,
    LABEL_TERM,
    REL_HAS_ATTRIBUTE,
    REL_PROPERTY_OF,
    REL_REPRESENTS,
    REL_SEMANTIC_FK,
    SQL_ATTR_SOURCE_BRIDGE,
)
from gsf.utils.join_columns import parse_join_columns
from gsf.utils.sample_values import parse_sample_values

logger = logging.getLogger(__name__)

_ALLOWED_NODE_LABELS = frozenset(Labels.LIST_OF_ALL)


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------


def fetch_databases(zone_ids: list[str] | None = None) -> list[dict[str, Any]]:
    """Return Database rows with schema counts only; ``schemas`` is empty for lazy trees.

    When *zone_ids* is supplied the result is restricted to databases (and schema
    counts) reachable through those zones.  Pass ``None`` (or omit) to return
    the full unfiltered catalog (admin / internal callers).
    """
    data_ids_by_zone = resolve_accessible_catalog_ids(zone_ids)
    if data_ids_by_zone is not None:
        db_ids = list(data_ids_by_zone["db_ids"])
        schema_ids = list(data_ids_by_zone["schema_ids"])
        where_clause = "WHERE db.id IN $db_ids AND s.id IN $schema_ids"
        params: dict[str, Any] = {"db_ids": db_ids, "schema_ids": schema_ids}
    else:
        where_clause = ""
        params = {}

    rows = graph().query_read(
        f"""
        MATCH (db:{Labels.DB})-[:{Edges.CONTAINS}]->(s:{Labels.SCHEMA})
        {where_clause}
        RETURN db.id AS id, db.name AS name, db.description AS description,
               count(s) AS schema_count
        ORDER BY name
        """,
        params,
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


def fetch_schemas_for_database(
    db_id: str,
    zone_ids: list[str] | None = None,
) -> dict[str, Any] | None:
    """Return schemas_count and a list of schema summaries for a database.

    Returns a dict with ``schemas_count`` and ``schemas`` — a list of
    ``{id, schema_name, tables_count}`` dicts.

    Returns ``None`` if no ``Database`` matches ``db_id``.

    When *zone_ids* is supplied only schemas (and their table counts) reachable
    through those zones are returned.
    """
    data_ids_by_zone = resolve_accessible_catalog_ids(zone_ids)
    if data_ids_by_zone is not None:
        schema_ids = list(data_ids_by_zone["schema_ids"])
        table_ids = list(data_ids_by_zone["table_ids"])
        where_clause = "WHERE s.id IN $schema_ids AND t.id IN $table_ids"
        params: dict[str, Any] = {
            "db_id": db_id,
            "schema_ids": schema_ids,
            "table_ids": table_ids,
        }
    else:
        where_clause = ""
        params = {"db_id": db_id}

    rows = graph().query_read(
        f"""
        MATCH (db:{Labels.DB} {{id: $db_id}})-[:{Edges.CONTAINS}]->
              (s:{Labels.SCHEMA})-[:{Edges.CONTAINS}]->(t:{Labels.TABLE})
        {where_clause}
        WITH s.id AS id, s.name AS schema_name, s.description AS description,
             count(t) AS tables_count
        ORDER BY schema_name
        WITH collect({{id: id, schema_name: schema_name,
                      description: description,
                      tables_count: tables_count}}) AS schemas
        RETURN size(schemas) AS schemas_count, schemas
        """,
        params,
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
        for r in graph().query_read(
            f"MATCH (s:{Labels.SCHEMA}) RETURN s.id AS schema_id",
        )
    ]


def fetch_schema_ids_for_database(database_name: str) -> list[str]:
    """Return Schema node IDs belonging to a single database.

    Scopes the catalog build to one database so schema-name collisions
    (e.g. multiple SQLite DBs all using ``main``) don't overwrite each
    other in the assembled ``all_schemas`` map.
    """
    return [
        r["schema_id"]
        for r in graph().query_read(
            f"""
            MATCH (db:{Labels.DB} {{name: $database_name}})
                  -[:{Edges.CONTAINS}]->(s:{Labels.SCHEMA})
            RETURN s.id AS schema_id
            """,
            {"database_name": database_name},
        )
    ]


def fetch_schemas_by_ids(
    relevant_schemas_ids: list | None = None,
) -> list[dict[str, str]]:
    """Return column-level rows for the given schema IDs (all schemas when empty)."""
    schema_ids = relevant_schemas_ids or []
    result = graph().query_read(
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
MATCH (s:{Labels.SCHEMA})-[:{Edges.CONTAINS}]->(t)
RETURN t.id AS id,
       t.name AS name,
       s.name AS schema_name,
       t.description AS description,
       t.pk as pk
"""

_FETCH_TABLE_BY_NAME = f"""
MATCH (t:{Labels.TABLE} {{name: $name}})
MATCH (s:{Labels.SCHEMA})-[:{Edges.CONTAINS}]->(t)
RETURN t.id AS id,
       t.name AS name,
       s.name AS schema_name,
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
MATCH (db:{Labels.DB})-[:{Edges.CONTAINS}]->(sch:{Labels.SCHEMA})
      -[:{Edges.CONTAINS}]->(tbl:{Labels.TABLE} {{id: tid}})
MATCH (tbl)-[:{Edges.CONTAINS}]->(col:{Labels.COLUMN})
WITH db, tbl, sch, collect({{name: col.name, data_type: col.data_type,
                         description: {column_description_expr("col")}}}) AS cols
RETURN tbl.id AS id, tbl.name AS name, tbl.description AS description,
       db.name AS database_name, sch.name AS schema_name, tbl.pk AS pk, cols
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


# Shared middle segment of the Table Cypher queries below: given `db, s, t`
# in scope, computes `columns_count`, `sql_count` and `unique_term_ids` (a
# Table's terms via both REPRESENTS and the ColumnAttribute/SEMANTIC_FK
# path, deduplicated). Interpolate between a query's initial MATCH/WHERE and
# its RETURN — used by both ``fetch_tables_for_schema`` (below) and
# ``gsf.dal.exploration.fetch_data_exploration_graph`` (every visible table)
# so the two stay in sync instead of drifting as separately-maintained
# copies. Public (no leading underscore) so the Exploration DAL can import it.
TABLE_COUNTS_SUBQUERY = f"""
MATCH (t)-[:{Edges.CONTAINS}]->(c:{Labels.COLUMN})
WITH db, s, t, count(DISTINCT c) AS columns_count
OPTIONAL MATCH (t)<-[:{Edges.SQL}]-(sql:{Labels.SQL})
WITH db, s, t, columns_count, count(DISTINCT sql) AS sql_count
OPTIONAL MATCH (t)-[:{REL_REPRESENTS}]->(represented:{LABEL_TERM})
WITH db, s, t, columns_count, sql_count,
     collect(DISTINCT represented.id) AS represented_term_ids
OPTIONAL MATCH (t)-[:{Edges.CONTAINS}]->(:{Labels.COLUMN})
      -[:{REL_HAS_ATTRIBUTE}|{REL_SEMANTIC_FK}]->
      (:{LABEL_COLUMN_ATTRIBUTE})-[:{REL_PROPERTY_OF}]->
      (attribute_term:{LABEL_TERM})
WITH db, s, t, columns_count, sql_count,
     represented_term_ids,
     collect(DISTINCT attribute_term.id) AS attribute_term_ids
WITH db, s, t, columns_count, sql_count,
     represented_term_ids + attribute_term_ids AS all_term_ids
WITH db, s, t, columns_count, sql_count,
     reduce(unique_ids = [], term_id IN all_term_ids |
         CASE
             WHEN term_id IS NULL OR term_id IN unique_ids THEN unique_ids
             ELSE unique_ids + term_id
         END
     ) AS unique_term_ids
"""


def fetch_tables_for_schema(
    schema_id: str,
    *,
    database_name: str
    | None = None,  # accepted for API compat; schema_id is globally unique
    zone_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Return Table payloads with column, SQL, and Term counts for a schema.

    When *zone_ids* is supplied only tables reachable through those zones are
    returned.
    """
    where_clause, params = resolve_table_filter(
        zone_ids, "t.id", extra_params={"schema_id": schema_id}
    )

    return graph().query_read(
        f"""
        MATCH (db:{Labels.DB})-[:{Edges.CONTAINS}]->
              (s:{Labels.SCHEMA} {{id: $schema_id}})-[:{Edges.CONTAINS}]->
              (t:{Labels.TABLE})
        {where_clause}
        {TABLE_COUNTS_SUBQUERY}
        RETURN t.id AS id,
               t.name AS name,
               t.table_type AS table_type,
               db.name AS database_name,
               s.name AS schema_name,
               {table_description_expr("t")} AS description,
               coalesce(t.description_certified, false) AS description_certified,
               columns_count,
               sql_count,
               size(unique_term_ids) AS terms_count
        ORDER BY name
        """,
        params,
    )


def fetch_sorted_tables() -> list[dict[str, Any]]:
    """Return all tables ordered by query_count descending."""
    rows = graph().query_read(_FETCH_TABLES_QUERY)
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
    rows = graph().query_read(_FETCH_TABLE_BY_ID, {"table_id": table_id})
    return rows[0] if rows else None


def fetch_table_by_name(name: str) -> dict[str, Any] | None:
    """Return the first Table row matching *name*, or None if not found."""
    rows = graph().query_read(_FETCH_TABLE_BY_NAME, {"name": name})
    return rows[0] if rows else None


def fetch_tables_by_ids(table_ids: list[str]) -> list[dict[str, Any]]:
    """Return Table rows with nested column summaries for the given IDs."""
    if not table_ids:
        return []
    try:
        rows = graph().query_read(_FETCH_TABLES_BY_IDS, {"table_ids": table_ids})
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
                "database_name": row.get("database_name") or "",
                "schema_name": row.get("schema_name") or "",
                "label": "Table",
                # The prediction graph keys its entities on this: a table that
                # arrives without it reaches KumoRFM with no identity, which
                # costs it every edge and makes it unusable in `FOR EACH`.
                "pk": row.get("pk") or [],
                "columns": cols,
            }
        )
    return tables


def fetch_all_tables_without_term(
    database_name: str | None = None,
) -> list[dict[str, Any]]:
    """Return Table nodes that have not yet been assigned a Term.

    When *database_name* is provided, only tables belonging to that database
    are returned. Multiple databases can be co-resident in the same Neo4j
    graph (e.g. the BIRD benchmark), so scoping keeps each compile pass — and
    the ``database_name`` its embeddings are tagged with — isolated to a single
    database. When omitted, every term-less table in the graph is returned.
    """
    from nemo_retriever.tabular_data.ingestion.model.reserved_words import Edges

    if database_name is not None:
        return graph().query_read(
            f"""
            MATCH (d:{Labels.DB} {{name: $database_name}})-[:{Edges.CONTAINS}]->
                  (sch:{Labels.SCHEMA})-[:{Edges.CONTAINS}]->(t:{Labels.TABLE})
            WHERE NOT (t)-[:{REL_REPRESENTS}]->()
            RETURN t.id AS id, t.name AS name, t.description AS description,
                   sch.name AS schema_name
            ORDER BY t.name
            """,
            {"database_name": database_name},
        )

    return graph().query_read(
        f"""
        MATCH (t:{Labels.TABLE})
        WHERE NOT (t)-[:{REL_REPRESENTS}]->()
        MATCH (t)<-[:{Edges.CONTAINS}]-(sch:{Labels.SCHEMA})
        RETURN t.id AS id, t.name AS name, t.description AS description,
               sch.name AS schema_name
        ORDER BY t.name
        """
    )


def fetch_join_neighbors(table_id: str) -> list[dict[str, Any]]:
    """Return JOIN-adjacent tables (undirected), one row per neighbour."""
    return graph().query_read(_FETCH_JOIN_NEIGHBORS, {"table_id": table_id})


def fetch_join_edges() -> list[dict[str, Any]]:
    """Return all JOIN edges between tables.

    ``join_columns`` is stored as a JSON string (see
    ``gsf.utils.join_columns``), so it is parsed back to a list here.
    """
    rows = graph().query_read(_FETCH_JOINS_QUERY)
    for row in rows:
        row["join_columns"] = parse_join_columns(row.get("join_columns"))
    return rows


# ---------------------------------------------------------------------------
# Column
# ---------------------------------------------------------------------------

_FETCH_COLUMNS_QUERY = f"""
MATCH (t:{Labels.TABLE} {{id: $table_id}})
MATCH (t)-[:{Edges.CONTAINS}]->(c:{Labels.COLUMN})
RETURN c.id AS id,
       c.name AS name,
       c.data_type AS data_type,
       {column_description_expr("c")} AS description,
       c.ordinal_position AS ordinal_position,
       c.sample_values AS sample_values,
       EXISTS {{ (c)-[:{Edges.FOREIGN_KEY}]->(:{Labels.COLUMN}) }} AS is_foreign_key
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
      <-[:{Edges.CONTAINS}]-(db:{Labels.DB})
RETURN col.id AS col_id, tbl.name AS table_name, sch.name AS schema_name,
       db.name AS database_name
"""


def fetch_columns_for_table(
    table_id: str,
    *,
    skip: int = 0,
    limit: int | None = None,
) -> dict[str, Any] | None:
    """Return a table dict with nested columns, or None if the table is missing.

    Columns are ordered by ordinal position, so *skip* and *limit* read one
    page of that order; pair them with ``count_columns_for_table`` for the
    table's full column count, which no paged read can report. Omit *limit*
    for every column of the table, which is what the catalog tree and the
    text-to-SQL context need.

    Paging happens inside a subquery scoped to the table, so the table's own
    fields (``table_name``, ``schema_name``, ``database_name``) come back the
    same way whether the page holds rows or not — ``None`` means the table
    itself (or its Schema/Database path) is missing, never that *skip* landed
    past the last column.
    """
    params: dict[str, Any] = {"table_id": table_id}
    paging = paging_clause(skip, limit, params)
    rows = graph().query_read(
        f"""
        MATCH (db:{Labels.DB})-[:{Edges.CONTAINS}]->(s:{Labels.SCHEMA})-[:{Edges.CONTAINS}]->
              (t:{Labels.TABLE} {{id: $table_id}})
        CALL (t) {{
            MATCH (t)-[:{Edges.CONTAINS}]->(c:{Labels.COLUMN})
            WITH c ORDER BY c.ordinal_position
            {paging}
            RETURN collect({{
                       id: c.id,
                       ordinal_position: c.ordinal_position,
                       column_name: c.name,
                       data_type: c.data_type,
                       description: {column_description_expr("c")},
                       description_certified: coalesce(c.description_certified, false),
                       sample_values: c.sample_values
                   }}) AS columns
        }}
        RETURN t.name AS table_name,
               t.table_type AS table_type,
               s.name AS schema_name,
               db.name AS database_name,
               columns
        LIMIT 1
        """,
        params,
    )
    if not rows:
        return None
    table = rows[0]
    for column in table.get("columns") or []:
        column["sample_values"] = parse_sample_values(column.get("sample_values"))
    return table


def count_columns_for_table(table_id: str) -> int:
    """Return how many Columns a table has.

    Companion to ``fetch_columns_for_table`` when it is called with a *limit*:
    Neo4j won't report the unpaged size of a ``LIMIT``-ed result, so the
    caller's pager needs this second query.
    """
    rows = graph().query_read(
        f"""
        MATCH (:{Labels.TABLE} {{id: $table_id}})-[:{Edges.CONTAINS}]->(c:{Labels.COLUMN})
        RETURN count(c) AS total
        """,
        {"table_id": table_id},
    )
    return rows[0]["total"] if rows else 0


def fetch_parent_table_id_for_column(column_id: str) -> str | None:
    """Return the id of the Table that contains this Column, or None."""
    rows = graph().query_read(
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
    conn = graph()
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
    """Batch lookup: Column id → database/schema/table identity."""
    if not col_ids:
        return {}
    try:
        rows = graph().query_read(_FETCH_COL_TABLE_CONTEXTS, {"col_ids": col_ids})
    except Exception:
        logger.warning("fetch_col_table_contexts: Neo4j query failed", exc_info=True)
        return {}
    return {
        r["col_id"]: {
            "table_name": r.get("table_name") or "",
            "schema_name": r.get("schema_name") or "",
            "database_name": r.get("database_name") or "",
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
    graph().query_write(
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


def store_column_uniqueness(table_id: str, uniqueness: dict[str, bool]) -> None:
    """Write is_unique flags onto Column nodes for a given table.

    Skips silently when *uniqueness* is empty.
    """
    if not uniqueness:
        return
    entries = [
        {"column_name": col, "is_unique": bool(is_unique)}
        for col, is_unique in uniqueness.items()
    ]
    graph().query_write(
        f"""
        MATCH (t:{Labels.TABLE} {{id: $table_id}})-[:{Edges.CONTAINS}]->(col:{Labels.COLUMN})
        WHERE col.name IN [e IN $entries | e.column_name]
        WITH col,
             [e IN $entries WHERE e.column_name = col.name | e.is_unique][0]
             AS iu
        WHERE iu IS NOT NULL
        SET col.is_unique = iu
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
    conn = graph()
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
                   {column_description_expr("c")} AS description,
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
    conn = graph()
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
    rows = graph().query_write(
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
    props = graph().query_read(
        f"""
        MATCH (n:{label_filter} {{id: $id}})
        RETURN apoc.map.setKey(properties(n), "label", labels(n)[0]) AS props
        """,
        {"id": id},
    )
    return props[0]["props"] if props else None


def fetch_item_by_id(item_id: str, label: str | list[str]) -> dict | None:
    """Like ``fetch_node_properties_by_id`` but logs an error when the node is missing."""
    result = fetch_node_properties_by_id(item_id, label)
    if result is None:
        logger.error("Required item with id %r not found in graph.", item_id)
    return result


def fetch_bridge_table_candidates(database_name: str) -> list[dict[str, Any]]:
    """Return pure-FK junction tables eligible for bridge SqlAttribute creation.

    A table qualifies when it has at least two columns, no column has
    HAS_ATTRIBUTE, every column is linked via FOREIGN_KEY or SEMANTIC_FK,
    every column resolves to an FK pair, and no SqlAttribute with source
    ``bridgeTable`` already references the table through HAS_SQL -> Sql -> SQL.

    Self-referential bridges are allowed (multiple FK columns targeting the
    same table), e.g. ``also_buy(product_id, also_buy_product_id)``.
    """
    rows = graph().query_read(
        f"""
        MATCH (db:{Labels.DB} {{name: $database_name}})-[:{Edges.CONTAINS}]->
              (sch:{Labels.SCHEMA})-[:{Edges.CONTAINS}]->(t:{Labels.TABLE})
        MATCH (t)-[:{Edges.CONTAINS}]->(col:{Labels.COLUMN})
        WITH sch, t, collect(col) AS cols
        WHERE size(cols) >= 2
          AND NONE(c IN cols WHERE (c)-[:{REL_HAS_ATTRIBUTE}]->())
          AND ALL(
            c IN cols
            WHERE (c)-[:{Edges.FOREIGN_KEY}]->(:{Labels.COLUMN})
               OR (c)-[:{REL_SEMANTIC_FK}]->(:{LABEL_COLUMN_ATTRIBUTE})
          )
          AND NOT EXISTS {{
            (attr:{LABEL_SQL_ATTRIBUTE} {{source: $bridge_source}})
                  -[:{Edges.HAS_SQL}]->(:{Labels.SQL})-[:{Edges.SQL}]->(t)
          }}
        WITH sch, t, cols
        UNWIND cols AS col
        OPTIONAL MATCH (col)-[:{Edges.FOREIGN_KEY}]->(fk_tgt:{Labels.COLUMN})
              <-[:{Edges.CONTAINS}]-(fk_tbl:{Labels.TABLE})
              <-[:{Edges.CONTAINS}]-(fk_sch:{Labels.SCHEMA})
        OPTIONAL MATCH (col)-[:{REL_SEMANTIC_FK}]->(:{LABEL_COLUMN_ATTRIBUTE})
              <-[:{REL_HAS_ATTRIBUTE}]-(sem_tgt:{Labels.COLUMN})
              <-[:{Edges.CONTAINS}]-(sem_tbl:{Labels.TABLE})
              <-[:{Edges.CONTAINS}]-(sem_sch:{Labels.SCHEMA})
        WITH sch, t, cols, col,
             coalesce(fk_tbl, sem_tbl) AS tgt_tbl,
             coalesce(fk_sch, sem_sch) AS tgt_sch,
             coalesce(fk_tgt, sem_tgt) AS tgt_col
        WHERE tgt_tbl IS NOT NULL AND tgt_col IS NOT NULL
        WITH sch, t, cols,
             collect(DISTINCT {{
               source_column: col.name,
               target_table: tgt_tbl.name,
               target_schema: tgt_sch.name,
               target_column: tgt_col.name,
               target_table_id: tgt_tbl.id
             }}) AS fk_pairs
        WHERE size(fk_pairs) >= 2 AND size(fk_pairs) = size(cols)
        RETURN t.id AS table_id,
               t.name AS table_name,
               sch.name AS schema_name,
               t.description AS description,
               fk_pairs
        ORDER BY t.name
        """,
        {"database_name": database_name, "bridge_source": SQL_ATTR_SOURCE_BRIDGE},
    )
    return [dict(row) for row in rows]
